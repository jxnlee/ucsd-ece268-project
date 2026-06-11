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

    def rescue_xlix_permutation(self, state_array):
        """Executes the Rescue-XLIX permutation using optimized matrix math."""
        state = self._array(state_array)
        
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
            
        padded = list(message_elements) + [1]
        while len(padded) % self.rp != 0:
            padded.append(0)
            
        state = self._zeros(self.m)
        
        # Absorbing Phase
        for chunk_idx in range(0, len(padded), self.rp):
            chunk = self._array(padded[chunk_idx : chunk_idx + self.rp])
            state[:self.rp] = (state[:self.rp] + chunk) % self.p
            state = self.rescue_xlix_permutation(state)
            
        # Squeezing Phase
        output = []
        while len(output) < output_length:
            for j in range(self.rp):
                if len(output) < output_length:
                    output.append(int(state[j]))
            if len(output) < output_length:
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
        if output_length is None:
            output_length = self.rp
            
        batch_size = len(batch_elements)
        
        # Padding Phase (Executed uniformly for the batch)
        # Find the maximum message length in the batch to pad uniformly
        max_len = max(len(msg) for msg in batch_elements)
        padded_len = max_len + 1
        if padded_len % self.rp != 0:
            padded_len += self.rp - (padded_len % self.rp)
            
        # Build a uniform 2D padded matrix on the host side
        padded_matrix = []
        for msg in batch_elements:
            padded_msg = list(msg) + [1] + [0] * (padded_len - len(msg) - 1)
            padded_matrix.append(padded_msg)
            
        # Send the entire batch data to the target device (CPU or GPU) AT ONCE
        padded_arr = self._array(padded_matrix) # Shape: (batch_size, padded_len)
        
        # Initialize batch state: Shape (batch_size, m)
        state = self._zeros((batch_size, self.m))
        
        # 2. Absorbing Phase (Loops over message chunks, but processes ALL batches in parallel)
        for chunk_idx in range(0, padded_len, self.rp):
            # Slice out the same chunk slice across all batches simultaneously
            chunk = padded_arr[:, chunk_idx : chunk_idx + self.rp]
            
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
                
        # CRITICAL FIX: Pull the ENTIRE batch matrix back from the GPU to the CPU memory 
        # in one single bulk transfer, rather than looping item-by-item over PCIe.
        if self.gpu:
            return outputs.get().astype(int).tolist() # .get() safely pulls full VRAM to RAM
        else:
            return outputs.astype(int).tolist()
def run_differential_test():
    # ---------------------------------------------------------
    # 1. Parameter Setup
    # ---------------------------------------------------------
    p = 2**61 - 1  # 61-bit Mersenne Prime
    m = 3          # State width
    c = 1          # Capacity
    s = 128        # Security bits

    print("Initializing Rescue-Prime Test Instances...")
    print("==========================================")
    
    # Initialize CPU Instance
    start_init = time.perf_counter()
    cpu_rp = RescuePrime(p, m, c, s, enable_gpu=False)
    cpu_init_time = time.perf_counter() - start_init
    print(f"[-] CPU Instance Ready (Init took: {cpu_init_time:.4f}s)")

    # Initialize GPU Instance
    # Note: This will compile the ElementwiseKernel exactly once
    start_init = time.perf_counter()
    try:
        gpu_rp = RescuePrime(p, m, c, s, enable_gpu=True)
        gpu_init_time = time.perf_counter() - start_init
        print(f"[-] GPU Instance Ready (Init + Kernel Compile took: {gpu_init_time:.4f}s)")
        has_gpu = True
    except Exception as e:
        print(f"\n[!] GPU Initialization Failed! Ensure CuPy and CUDA are installed correctly.")
        print(f"    Error details: {e}")
        print("    Aborting differential benchmarking, running CPU check only.")
        has_gpu = False

    # ---------------------------------------------------------
    # 2. Test Vector Generation
    # ---------------------------------------------------------
    # We will generate a reasonably long sequence of random elements modulo p
    # to make sure the sponge absorption loop iterates multiple times.
    num_elements = 500  
    test_vector = [random.randint(0, p - 1) for _ in range(num_elements)]
    output_len = 10  # Squeeze out 10 field elements
    
    print(f"\nGenerated test vector containing {num_elements} field elements.")
    print("Beginning execution benchmarks...\n")

    # ---------------------------------------------------------
    # 3. CPU Hashing Execution
    # ---------------------------------------------------------
    # Warm-up pass to let python optimize lookup paths
    _ = cpu_rp.hash(test_vector, output_length=output_len)
    
    # Timed benchmarking loop
    iterations = 5
    cpu_start = time.perf_counter()
    for _ in range(iterations):
        cpu_hash_output = cpu_rp.hash(test_vector, output_length=output_len)
    cpu_total_time = time.perf_counter() - cpu_start
    cpu_avg_time = cpu_total_time / iterations

    print("--- CPU Performance Results ---")
    print(f"Total time ({iterations} passes): {cpu_total_time:.6f} seconds")
    print(f"Average time per hash    : {cpu_avg_time:.6f} seconds")

    if not has_gpu:
        print(f"Sample CPU Digest: {cpu_hash_output[:3]}...")
        return

    # ---------------------------------------------------------
    # 4. GPU Hashing Execution
    # ---------------------------------------------------------
    # Warm-up pass (Crucial for GPU! Compiles under-the-hood internal execution structures)
    _ = gpu_rp.hash(test_vector, output_length=output_len)
    cp.cuda.Stream.null.synchronize() # Wait for the GPU warm-up to completely finish

    gpu_start = time.perf_counter()
    for _ in range(iterations):
        gpu_hash_output = gpu_rp.hash(test_vector, output_length=output_len)
    
    # CRITICAL: CUDA execution is asynchronous. Python will continue running lines 
    # before the GPU hardware finishes. We MUST synchronize to get an accurate time measurement.
    cp.cuda.Stream.null.synchronize()
    gpu_total_time = time.perf_counter() - gpu_start
    gpu_avg_time = gpu_total_time / iterations

    print("\n--- GPU Performance Results ---")
    print(f"Total time ({iterations} passes): {gpu_total_time:.6f} seconds")
    print(f"Average time per hash    : {gpu_avg_time:.6f} seconds")

    # ---------------------------------------------------------
    # 5. Differential Validation & Verification
    # ---------------------------------------------------------
    print("\n--- Correctness and Validation Verification ---")
    print("==================================================")
    
    # Assert exact dimensional structures match
    if len(cpu_hash_output) != len(gpu_hash_output):
        print("[FAIL] Output length mismatch!")
        print(f"       CPU Length: {len(cpu_hash_output)}, GPU Length: {len(gpu_hash_output)}")
        return

    # Deep element-by-element equivalence evaluation
    match = True
    for idx in range(len(cpu_hash_output)):
        if cpu_hash_output[idx] != gpu_hash_output[idx]:
            print(f"[FAIL] Cryptographic divergence at field element index {idx}!")
            print(f"       CPU element: {cpu_hash_output[idx]}")
            print(f"       GPU element: {gpu_hash_output[idx]}")
            match = False
            break

    if match:
        print("[PASS] Cryptographic Equivalence Confirmed!")
        print("       The CPU array and GPU VRAM arrays computed completely identical digest elements.")
        print(f"Digest Sample: {cpu_hash_output[:3]}...")
        
        # Calculate speedup metric
        if gpu_avg_time > 0:
            speedup = cpu_avg_time / gpu_avg_time
            print(f"\n[Speedup Factor]: GPU is operating {speedup:.2f}x faster than the CPU variant on this payload.")
def run_batched_differential_test():
    p = 2**61 - 1
    m = 3
    c = 1
    s = 128
    output_len = 2

    print("Initializing Batch Instances...")
    cpu_rp = RescuePrime(p, m, c, s, enable_gpu=False)
    
    try:
        gpu_rp = RescuePrime(p, m, c, s, enable_gpu=True)
        has_gpu = True
    except Exception as e:
        print(f"GPU Init failed: {e}")
        has_gpu = False

    # ---------------------------------------------------------
    # Generate Batch Data: 10,000 messages, each with 10 elements
    # ---------------------------------------------------------
    batch_size = 10000
    elements_per_message = 10
    
    print(f"\nGenerating batch dataset: {batch_size} messages...")
    test_batch = [
        [random.randint(0, p - 1) for _ in range(elements_per_message)]
        for _ in range(batch_size)
    ]
    print("Dataset ready. Running benchmarks...\n")

    # ---------------------------------------------------------
    # CPU Batch Hashing
    # ---------------------------------------------------------
    print("Running CPU Batch...")
    # Warm-up
    _ = cpu_rp.hash_batch(test_batch[:10], output_length=output_len)
    
    start_cpu = time.perf_counter()
    cpu_results = cpu_rp.hash_batch(test_batch, output_length=output_len)
    cpu_time = time.perf_counter() - start_cpu
    print(f"[-] CPU Batch completed in: {cpu_time:.4f} seconds")

    if not has_gpu:
        return

    # ---------------------------------------------------------
    # GPU Batch Hashing
    # ---------------------------------------------------------
    print("\nRunning GPU Batch...")
    # Warm-up
    _ = gpu_rp.hash_batch(test_batch[:10], output_length=output_len)
    cp.cuda.Stream.null.synchronize()

    start_gpu = time.perf_counter()
    gpu_results = gpu_rp.hash_batch(test_batch, output_length=output_len)
    cp.cuda.Stream.null.synchronize() # Crucial synchronization check
    gpu_time = time.perf_counter() - start_gpu
    print(f"[-] GPU Batch completed in: {gpu_time:.4f} seconds")

    # ---------------------------------------------------------
    # Verification & Speedup Calculation
    # ---------------------------------------------------------
    print("\n--- Validation Results ---")
    if cpu_results == gpu_results:
        print("[PASS] Cryptographic Equivalence Maintained across all 10,000 hashes!")
        speedup = cpu_time / gpu_time
        print(f"[Speedup Factor]: GPU is {speedup:.2f}x FASTER than CPU.")
    else:
        print("[FAIL] Outputs diverged!")

if __name__ == "__main__":
    run_batched_differential_test()
