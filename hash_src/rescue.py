import math
import hashlib
import numpy as np
import cupy as cp
from sympy import isprime

import time
import random

class RescuePrime:
    """
    Rescue-Prime hash function implementation.

    Attributes
    -----------
    p : int
        Prime number for finite field Fp over which operations are defined.
    m : int
        State width (number of field elements in the state).
    c : int
        Capacity of the arithmetic sponge.
    s : int
        Target security level in bits.
    rp : int
        Rate of the arithmetic sponge (m - c).
        Determines the number of field elements absorbed between invokations of the Rescue-XLIX permutation.
    alpha : int
        S-box exponent.
    alpha_inv : int
        Inverse S-box exponent.
    gpu : bool
        Whether to use GPU acceleration.
    """
    def __init__(self, p: int, m: int, c: int, s: int, enable_gpu: bool=False):
        """
        Initializes Rescue-Prime hash instance.

        Parameters
        ----------
        p : int
            Prime number for finite field Fp over which operations are defined.
            Must be a prime number with a binary expansion of at least 32 bits.
        m : int
            State width (number of field elements in the state).
            Must be greater than 1.
        c : int
            Capacity of the arithmetic sponge.
        s : int
            Target security level in bits.
        dtype : type
            Data type for field elements.
        enable_gpu : bool, optional
            Whether to use GPU acceleration, by default False.
        """

        # Validate Parameters
        if not isinstance(p, int) or p < 2**32 or p > 2**63-1 or not isprime(p):
            raise ValueError("p must be a prime number with at least 32 bits.")
        if not isinstance(m, int) or m <= 1:
            raise ValueError("m must be greater than 1.")
        if not isinstance(c, int) or c < 0 or c > m:
            raise ValueError("Invalid capacity c. Must be an integer between 0 and m.")
        if not isinstance(s, int) or s < 80 or s > 512:
            raise ValueError("s must be an integer between 80 and 512.")
        
        # Set attributes
        self.p = p
        self.m = m
        self.c = c
        self.s = s

        self.rp = self.m - self.c # rate of the sponge

        self.gpu = enable_gpu

        # adjust framework for GPU vs CPU
        self.xp = cp if self.gpu else np
        self.dtype = self.xp.uint64

        self._mod_pow_kernel = None
        if self.gpu:
            self._mod_pow_kernel = cp.ElementwiseKernel(
                'uint64 x, uint64 exp, uint64 mod',
                'uint64 y',
                '''
                unsigned long long res = 1;
                unsigned long long base = x % mod;
                unsigned long long e = exp; // FIX: Copy to a mutable local variable
                
                while (e > 0) {
                    if (e & 1) res = field_multiply(res, base, mod);
                    base = field_multiply(base, base, mod);
                    e >>= 1; // FIX: Modifying local variable 'e' instead of 'exp'
                }
                y = res;
                ''',
                'mod_pow_kernel',
                preamble='''
                __device__ unsigned long long field_multiply(unsigned long long a, unsigned long long b, unsigned long long mod) {
                    unsigned long long res = 0;
                    a %= mod;
                    while (b > 0) {
                        if (b & 1) res = (res + a) % mod;
                        a = (a * 2) % mod;
                        b >>= 1;
                    }
                    return res;
                }
                '''
            )


        # Compute the S-box exponents alpha and alpha_inv
        self.alpha = self._get_alpha()
        self.alpha_inv = pow(self.alpha, -1, self.p - 1)

        # Compute the number of rounds N
        self.N = self._get_num_rounds()

        # Generate MDS Matrix
        self.MDS = self._generate_mds_matrix()

        # Generate Round Constants
        raw_constants = self._generate_constants()
        self.constants = self._array(raw_constants).reshape(2 * self.N, self.m)

    def _array(self, data):
        """Helper function to create either a NumPy or CuPy array based on GPU configuration."""
        return self.xp.array(data, dtype=self.dtype)
    def _zeros(self, shape):
        """Helper function to create an array of zeros"""
        return self.xp.zeros(shape, dtype=self.dtype)
    def _ones(self, shape):
        """Helper function to create an array of ones"""
        return self.xp.ones(shape, dtype=self.dtype)
    
    def _get_alpha(self) -> int:
        """Finds the smallest integer alpha that is coprime with p-1"""
        for alpha in range(3, self.p, 2): # check odd numbers
            if math.gcd(alpha, self.p - 1) == 1:
                return alpha
        
        raise ValueError("No suitable alpha found")

    def _get_num_rounds(self) -> int:
        """Computes the number of rounds for Groebner basis attack"""
        # Initialize N candidate
        l1 = 1

        # compute number of rounds
        while True:
            dcon = math.floor(0.5 * (self.alpha - 1) * self.m * (l1 - 1) +2)
            v = self.m * (l1 - 1) + self.rp
            target = 2 ** self.s

            if math.comb(dcon + v, v) ** 2 > target:
                break

            l1 += 1
        
        # set min val for sanity and add 50%
        return math.ceil(1.5 * max(5, l1))

    def _find_primitive_element(self) -> int:
        """Finds the smallest primitive element (generator) modulo p."""
        
        # Factorize p-1: get all the unique prime factors of p-1    
        factors = []
        n = self.p - 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)

        # raise each candidate g to the power of (p-1)//f for each factor 
        for g in range(2, self.p):
            if all(pow(g, (self.p - 1) // f, self.p) != 1 for f in factors):
                return g
            
        raise ValueError("Primitive root not found.")

    def _generate_mds_matrix(self):
        """Generates an m x m MDS matrix via Vandermonde row reduction."""
        g = self._find_primitive_element()
        
        # Construct m x 2m Vandermonde Matrix
        V = [[pow(g, i * j, self.p) for j in range(2 * self.m)] for i in range(self.m)]
        
        # Perform Gaussian elimination to transform V into Reduced Row Echelon Form: [I | M_transposed]
        for i in range(self.m):
            # Pivot search/scaling
            pivot = V[i][i]
            inv_pivot = pow(pivot, -1, self.p)
            for j in range(2 * self.m):
                V[i][j] = (V[i][j] * inv_pivot) % self.p
                
            for k in range(self.m):
                if k != i:
                    factor = V[k][i]
                    for j in range(2 * self.m):
                        V[k][j] = (V[k][j] - factor * V[i][j]) % self.p
                        
        # Extract Right hand m x m matrix and transpose it to get final MDS Matrix
        M_T = [row[self.m:] for row in V]
        MDS = [[M_T[j][i] for j in range(self.m)] for i in range(self.m)]
        return self._array(MDS)

    def _generate_constants(self):
        """Generates 2 * m * N pseudo-random round constants using SHAKE-256."""
        seed_string = f"Rescue-XLIX({self.p},{self.m},{self.c},{self.s})".encode('ascii')
        bytes_per_element = math.ceil(self.p.bit_length() / 8) + 1
        total_elements = 2 * self.m * self.N
        total_bytes = bytes_per_element * total_elements
        
        shake = hashlib.shake_256()
        shake.update(seed_string)
        byte_str = shake.digest(total_bytes)
        
        round_constants = []
        for i in range(total_elements):
            chunk = byte_str[bytes_per_element*i : bytes_per_element*(i+1)]
            # Decode little-endian bytes to integer
            val = int.from_bytes(chunk, byteorder='little')
            round_constants.append(val % self.p)
        return round_constants

    def _sbox_pow(self, base_array, exponent):
        """Applies modular exponentiation across an entire NumPy array element-wise."""
        if self.gpu:
            # CuPy ElementwiseKernel speeds up modular exponentiation directly inside the GPU VRAM
            return self._mod_pow_kernel(base_array, int(exponent), self.p)
        else:
            v_pow = np.vectorize(lambda base: pow(int(base), int(exponent), self.p), otypes=[self.dtype])
            return v_pow(base_array)
    
    def _mds_multiply(self, state):
        """Multiplies the state vector by the MDS matrix modulo p."""
        return (state @ self.MDS) % self.p
    
    def _add_round_constants(self, state, round_index):
        """Adds the appropriate round constants"""
        return (state + self.constants[round_index]) % self.p

    def rescue_xlix_permutation(self, state):
        """Executes the Rescue-XLIX permutation using optimized matrix math."""
        for i in range(self.N):
            # S-box Layer
            state = self._sbox_pow(state, self.alpha)
            # MDS Matrix Multiplication
            state = self._mds_multiply(state)
            # Add Round Constants
            state = self._add_round_constants(state, 2 * i)

            # Inverse S-box Layer
            state = self._sbox_pow(state, self.alpha_inv)
            # MDS Matrix Multiplication
            state = self._mds_multiply(state)
            # Add Round Constants
            state = self._add_round_constants(state, 2 * i + 1)
                
        return state

    def hash(self, message_elements, output_length=None):
        if output_length is None:
            output_length = self.rp
        # Prepare padded message on the active backend without transferring memory
        if self.gpu and isinstance(message_elements, cp.ndarray):
            arr = message_elements.astype(self.dtype)
            total_len = int(arr.size) + 1
            pad_needed = (-(total_len)) % self.rp
            if pad_needed == 0:
                tail = cp.array([1], dtype=self.dtype)
            else:
                tail = cp.concatenate([cp.array([1], dtype=self.dtype), cp.zeros(pad_needed, dtype=self.dtype)])
            padded_arr = cp.concatenate([arr, tail])
        else:
            # Use NumPy arrays for CPU or other iterable inputs
            arr = np.asarray(message_elements, dtype=self.dtype)
            total_len = int(arr.size) + 1
            pad_needed = (-(total_len)) % self.rp
            if pad_needed == 0:
                tail = np.array([1], dtype=self.dtype)
            else:
                tail = np.concatenate([np.array([1], dtype=self.dtype), np.zeros(pad_needed, dtype=self.dtype)])
            padded_arr = np.concatenate([arr, tail])

        # print(padded_arr)

        state = self._zeros(self.m)

        # Absorbing Phase (iterate over chunks directly on the active backend)
        for chunk_idx in range(0, padded_arr.size, self.rp):
            chunk = padded_arr[chunk_idx: chunk_idx + self.rp]
            # ensure chunk uses the active xp type
            if (self.gpu and cp is not None and isinstance(padded_arr, cp.ndarray)) or (not self.gpu and isinstance(padded_arr, np.ndarray)):
                state[:self.rp] = (state[:self.rp] + chunk) % self.p
            else:
                # convert to active backend
                state[:self.rp] = (state[:self.rp] + self._array(chunk)) % self.p
            state = self.rescue_xlix_permutation(state)
            
        # Squeezing Phase
        output = self._zeros(output_length)

        squeezed_count = 0
        while squeezed_count < output_length:
            for j in range(self.rp):
                if squeezed_count < output_length:
                    output[squeezed_count] = state[j]
                    squeezed_count += 1
            if squeezed_count < output_length:
                state = self.rescue_xlix_permutation(state)
                
        return output

    def hash_batch(self, batch_elements, output_length=None):
        """
        Hashes a batch of messages simultaneously.
        
        Parameters
        ----------
        batch_elements : list of lists, or 2D array
            An array/list of shape (batch_size, num_elements_per_message)
        """
        if (self.gpu and not isinstance(batch_elements, cp.ndarray)) or (not self.gpu and not isinstance(batch_elements, np.ndarray)):
            batch_elements = self.xp.array(batch_elements)
        #print(batch_elements)
        if output_length is None:
            output_length = self.rp
            
        batch_size = len(batch_elements)
        
        # Padding Phase (Executed uniformly for the batch)
        # Find the maximum message length in the batch to pad uniformly
        max_len = max(len(msg) for msg in batch_elements)
        padded_len = max_len + 1
        if padded_len % self.rp != 0:
            padded_len += self.rp - (padded_len % self.rp)
            
        # Build a uniform 2D padded matrix on the active backend without unintended transfers
        if self.gpu and isinstance(batch_elements, cp.ndarray):
            arr = batch_elements.astype(self.dtype)
            if arr.ndim == 1:
                arr = arr.reshape((arr.shape[0], 1))
            batch_size, orig_len = arr.shape
            ones = cp.ones((batch_size, 1), dtype=self.dtype)
            zeros_tail = cp.zeros((batch_size, padded_len - orig_len - 1), dtype=self.dtype) if (padded_len - orig_len - 1) > 0 else cp.zeros((batch_size, 0), dtype=self.dtype)
            padded_arr = cp.concatenate([arr, ones, zeros_tail], axis=1)
        else:
            # Accept lists of lists or NumPy arrays; build padded NumPy matrix to convert once if needed
            if isinstance(batch_elements, np.ndarray):
                arr = batch_elements.astype(self.dtype)
                if arr.ndim == 1:
                    arr = arr.reshape((arr.shape[0], 1))
            else:
                # list of lists or uneven lengths
                arr = np.zeros((batch_size, padded_len - 0), dtype=self.dtype)
                for i, msg in enumerate(batch_elements):
                    msg_arr = np.asarray(msg, dtype=self.dtype)
                    arr[i, :msg_arr.size] = msg_arr
            ones = np.ones((batch_size, 1), dtype=self.dtype)
            zeros_tail = np.zeros((batch_size, padded_len - arr.shape[1] - 1), dtype=self.dtype) if (padded_len - arr.shape[1] - 1) > 0 else np.zeros((batch_size, 0), dtype=self.dtype)
            padded_arr = np.concatenate([arr, ones, zeros_tail], axis=1)

        # Send the entire batch data to the target device (CPU or GPU) AT ONCE
        if self.gpu and isinstance(padded_arr, cp.ndarray):
            padded_arr_backend = padded_arr
        else:
            padded_arr_backend = self._array(padded_arr)
        
        # Initialize batch state: Shape (batch_size, m)
        state = self._zeros((batch_size, self.m))
        
        # 2. Absorbing Phase (Loops over message chunks, but processes ALL batches in parallel)
        for chunk_idx in range(0, padded_len, self.rp):
            # Slice out the same chunk slice across all batches simultaneously
            chunk = padded_arr_backend[:, chunk_idx : chunk_idx + self.rp]
            
            # Vectorized addition across the entire batch matrix slice
            state[:, :self.rp] = (state[:, :self.rp] + chunk) % self.p
            state = self.rescue_xlix_permutation(state)
            
        # 3. Squeezing Phase (Vectorized batch extraction)
        # We pre-allocate the output array structure directly on the active device
        outputs = self._zeros((batch_size, output_length))
        
        squeezed_count = 0
        while squeezed_count < output_length:
            for j in range(self.rp):
                if squeezed_count < output_length:
                    outputs[:, squeezed_count] = state[:, j]
                    squeezed_count += 1
            if squeezed_count < output_length:
                state = self.rescue_xlix_permutation(state)
                
        # Return the array directly in the active array backend.
        return outputs
# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_comprehensive_benchmarks():
    REPS=5
    p = 2**61 - 1
    m = 3
    c = 1
    s = 128
    output_len = 1

    random.seed(0)
    rp_cpu = RescuePrime(p=p, m=m, c=c, s=s, enable_gpu=False)
    rp_gpu = RescuePrime(p=p, m=m, c=c, s=s, enable_gpu=True)

    print("\n" + "="*90)
    print(" Experiment 1: Batched Hash Benchmark (Fixed Message Size: 8 Bytes)")
    print("="*90)
    hdr = f"  {'N Messages':<12} | {'CPU Time (ms)':<13} | {'GPU Batch (ms)':<14} | {'CPU us/hash':<12} | {'GPU us/hash':<12} | {'Speedup':<10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    MSG_SZ = 8
    

    for N in [16, 256, 4096, 8192]:
        batched_msgs = [[random.randint(0, 255) for _ in range(MSG_SZ)] for _ in range(N)]

        cpu_samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for msg in batched_msgs:
                rp_cpu.hash(msg, output_len)
            cpu_samples.append(time.perf_counter() - t0)
        cpu_ms = min(cpu_samples) * 1000.0
        cpu_us = (cpu_ms * 1000.0) / N

        gpu_bat_samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            rp_gpu.hash_batch(batched_msgs, output_len)
            cp.cuda.Stream.null.synchronize()
            gpu_bat_samples.append(time.perf_counter() - t0)
        gpu_ms = min(gpu_bat_samples) * 1000.0
        gpu_us = (gpu_ms * 1000.0) / N
        print(f"  {N:<12} | {cpu_ms:<13.2f} | {gpu_ms:<14.2f} | {cpu_us:<12.2f} | {gpu_us:<12.2f} | {cpu_ms/gpu_ms:.2f}x")

    print("\n" + "="*95)
    print(" Experiment 2: Variable Input Size Scaling (Single Message)")
    print("="*95)
    print(f"  {'Input Size':<12} | {'CPU Time (ms)':<13} | {'GPU Time (ms)':<13} | {'CPU Throughput':<16} | {'GPU Throughput':<16} | {'Speedup':<10}")
    print("  " + "-"*92)

    for label, size in [("8 B", 8),("512 B", 512), ("1 KB", 1024), ("4 KB", 4096), ("8 KB", 8192)]:
        single_msg  = [random.randint(0, 255) for _ in range(size)]
        cpu_samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            rp_cpu.hash(single_msg, output_len)
            cpu_samples.append(time.perf_counter() - t0)
        cpu_ms   = min(cpu_samples) * 1000.0
        cpu_kbps = (size / 1e3) / (cpu_ms / 1000.0)

        gpu_samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            rp_gpu.hash(single_msg, output_len)
            cp.cuda.Stream.null.synchronize()
            gpu_samples.append(time.perf_counter() - t0)
        gpu_ms   = min(gpu_samples) * 1000.0
        gpu_kbps = (size / 1e3) / (gpu_ms / 1000.0)
        print(f"  {label:<12} | {cpu_ms:<13.3f} | {gpu_ms:<13.3f} | {cpu_kbps:.2f} MB/s     | {gpu_kbps:.2f} MB/s       | {cpu_ms/gpu_ms:.2f}x")

    print("="*95 + "\n")


if __name__ == "__main__":
    run_comprehensive_benchmarks()