import numpy as np
import time

try:
    import cupy as cp
except ImportError:
    cp = None

# round constants

_RC = [
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

# rho can be precomputed due to calculations being constant

_RHO = [
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14,
]

_PI = [
     0,  6, 12, 18, 24,
     3,  9, 10, 16, 22,
     1,  7, 13, 19, 20,
     4,  5, 11, 17, 23,
     2,  8, 14, 15, 21,
]

# CUDA String Kernel — Keccak-p[1600, 12] Permutation

CUDA_KECCAK_SRC = r"""
extern "C" {
__device__ __inline__ unsigned long long rot_l64(unsigned long long a, int n) {
    n %= 64;
    if (n == 0) return a;
    return (a << n) | (a >> (64 - n));
}

__global__ void keccak_p_1600_fused(unsigned char* state) {
    unsigned long long lane[25];
    
    // Unpack bytes to 64-bit lanes (Little Endian)
    for (int i = 0; i < 25; i++) {
        lane[i] = ((unsigned long long)state[8*i + 0])       |
                  ((unsigned long long)state[8*i + 1] << 8)  |
                  ((unsigned long long)state[8*i + 2] << 16) |
                  ((unsigned long long)state[8*i + 3] << 24) |
                  ((unsigned long long)state[8*i + 4] << 32) |
                  ((unsigned long long)state[8*i + 5] << 40) |
                  ((unsigned long long)state[8*i + 6] << 48) |
                  ((unsigned long long)state[8*i + 7] << 56);
    }

    const unsigned long long RC[12] = {
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };

    for (int r = 0; r < 12; r++) {
        // === THETA ===
        unsigned long long C[5];
        for (int x = 0; x < 5; x++) {
            C[x] = lane[x] ^ lane[x+5] ^ lane[x+10] ^ lane[x+15] ^ lane[x+20];
        }
        for (int x = 0; x < 5; x++) {
            unsigned long long D = C[(x+4)%5] ^ rot_l64(C[(x+1)%5], 1);
            for (int y = 0; y < 5; y++) {
                lane[x + 5*y] ^= D;
            }
        }

        // === RHO + PI ===
        int x = 1, y = 0;
        unsigned long long current = lane[x + 5*y];
        for (int t = 0; t < 24; t++) {
            int new_x = y;
            int new_y = (2*x + 3*y) % 5;
            x = new_x; y = new_y;
            int rot = (t + 1) * (t + 2) / 2;
            unsigned long long temp = lane[x + 5*y];
            lane[x + 5*y] = rot_l64(current, rot);
            current = temp;
        }

        // === CHI ===
        for (int y = 0; y < 5; y++) {
            unsigned long long T[5];
            for (int x = 0; x < 5; x++) T[x] = lane[x + 5*y];
            for (int x = 0; x < 5; x++) {
                lane[x + 5*y] = T[x] ^ ((~T[(x+1)%5]) & T[(x+2)%5]);
            }
        }

        // === IOTA ===
        lane[0] ^= RC[r];
    }

    // Repack lanes back into global device memory bytes
    for (int i = 0; i < 25; i++) {
        unsigned long long v = lane[i];
        state[8*i + 0] = (unsigned char)(v & 0xFF);
        state[8*i + 1] = (unsigned char)((v >> 8) & 0xFF);
        state[8*i + 2] = (unsigned char)((v >> 16) & 0xFF);
        state[8*i + 3] = (unsigned char)((v >> 24) & 0xFF);
        state[8*i + 4] = (unsigned char)((v >> 32) & 0xFF);
        state[8*i + 5] = (unsigned char)((v >> 40) & 0xFF);
        state[8*i + 6] = (unsigned char)((v >> 48) & 0xFF);
        state[8*i + 7] = (unsigned char)((v >> 56) & 0xFF);
    }
}
}
"""

class Kangaroo12:
    def __init__(self, enable_gpu=False):
        if enable_gpu and cp is None:
            raise ImportError(
                "CuPy backend requested but not available in this environment."
            )

        self.xp = cp if enable_gpu else np
        self.enable_gpu = enable_gpu

        self.RC = self.xp.array(_RC, dtype=self.xp.uint64)
        self.RHO = self.xp.array(_RHO, dtype=self.xp.uint64)
        self.PI = self.xp.array(_PI, dtype=self.xp.uint64)

        if self.enable_gpu:
            self.mod = cp.RawModule(code=CUDA_KECCAK_SRC)
            self.cu_kernel = self.mod.get_function("keccak_p_1600_fused")

    def _rotate(self, a, n):
        n %= 64
        return int(((a << n) | (a >> (64 - n))) & 0xFFFFFFFFFFFFFFFF)

    def _keccak_p_1600(self, state: bytearray) -> bytes:
        if self.enable_gpu:
            gpu_state = cp.array(list(state), dtype=cp.uint8)
            self.cu_kernel((1,), (1,), (gpu_state,))
            return bytes(gpu_state.get().tolist())
            
        xp = self.xp
        lane = xp.array(
            [int.from_bytes(state[8*i: 8*i+8], 'little') for i in range(25)],
            dtype=xp.uint64,
        )

        for r in range(12):
            # === THETA ===
            C = xp.array([
                int(lane[x] ^ lane[x+5] ^ lane[x+10] ^ lane[x+15] ^ lane[x+20])
                for x in range(5)
            ], dtype=xp.uint64)

            D = xp.array([
                int(C[(x-1) % 5]) ^ self._rotate(int(C[(x+1) % 5]), 1)
                for x in range(5)
            ], dtype=xp.uint64)

            for x in range(5):
                for y in range(5):
                    lane[x + 5*y] = int(lane[x + 5*y]) ^ int(D[x])

            # === RHO + PI ===
            rotated_lanes = xp.array([
                self._rotate(int(lane[i]), int(self.RHO[i])) for i in range(25)
            ], dtype=xp.uint64)
            
            lane = rotated_lanes[self.PI]

            # === CHI ===
            for y_pos in range(5):
                T = [int(lane[x_pos + 5*y_pos]) for x_pos in range(5)]
                for x_pos in range(5):
                    lane[x_pos + 5*y_pos] = T[x_pos] ^ ((~T[(x_pos+1) % 5] & 0xFFFFFFFFFFFFFFFF) & T[(x_pos+2) % 5])

            # === IOTA ===
            lane[0] = int(lane[0]) ^ int(self.RC[r])

        out = bytearray(200)
        for i in range(25):
            out[8*i: 8*i+8] = int(lane[i]).to_bytes(8, 'little')
        return bytes(out)

    def _turbo_shake256(self, message: bytes, separation_byte: int, output_byte_len: int) -> bytes:
        RATE = 136
        state = bytearray(200)
        data = bytes(message) + bytes([separation_byte])
        offset = 0

        while offset + RATE <= len(data):
            for i in range(RATE):
                state[i] ^= data[offset + i]
            state = bytearray(self._keccak_p_1600(state))
            offset += RATE

        remainder = len(data) - offset
        for i in range(remainder):
            state[i] ^= data[offset + i]
        state[RATE - 1] ^= 0x80
        state = bytearray(self._keccak_p_1600(state))

        output = bytearray()
        while output_byte_len > RATE:
            output.extend(state[:RATE])
            output_byte_len -= RATE
            state = bytearray(self._keccak_p_1600(state))
        output.extend(state[:output_byte_len])
        return bytes(output)

    def _length_encode(self, val: int) -> bytes:
        if val == 0:
            return b'\x00'
        s = bytearray()
        while val > 0:
            s.insert(0, val % 256)
            val //= 256
        s.append(len(s))
        return bytes(s)

    def hash(self, input_message: bytes, custom_string: bytes, output_byte_len: int) -> bytes:
        CV_LEN = 64
        s = bytes(input_message) + bytes(custom_string) + self._length_encode(len(custom_string))

        if len(s) <= 8192:
            return self._turbo_shake256(s, 0x07, output_byte_len)

        fin = bytearray(s[:8192]) + bytearray([0x03] + [0x00] * 7)

        offset = 8192
        num_blocks = 0
        while offset < len(s):
            block_size = min(len(s) - offset, 8192)
            cv = self._turbo_shake256(s[offset: offset + block_size], 0x0B, CV_LEN)
            fin.extend(cv)
            num_blocks += 1
            offset += block_size

        fin.extend(self._length_encode(num_blocks))
        fin.extend([0xFF, 0xFF])

        return self._turbo_shake256(bytes(fin), 0x06, output_byte_len)

# Test Verification & Timing Harness

def run_tests():
    print("--- Executing Kangaroo12 Engine Validation Suite ---\n")

    kt256_empty_expected = "b23d2e9cea9f4904e02bec06817fc10ce38ce8e93ef4c89e6537076af8646404"
    kt256_custom_expected = "bbe187fa1c6d40617ef4023ec8ddb414d61d07a5f0db114015c11924faa24233"
    kt256_multi_expected = "d3b501e04a1c9a430f398e937ad57be691aa35b107d0ba4cdfdcaac3aa2ab534"

    # --- CPU Verification ---
    print("[*] Launching Kangaroo12 (CPU Path)...")
    k_cpu = Kangaroo12(enable_gpu=False)

    cpu_empty = k_cpu.hash(b"", b"", 32).hex()
    assert cpu_empty == kt256_empty_expected, "CPU empty text test FAILED"
    print("    [+] Empty Message: PASS")

    cpu_custom = k_cpu.hash(b"", b"KangarooTwelve", 32).hex()
    assert cpu_custom == kt256_custom_expected, "CPU custom string test FAILED"
    print("    [+] Custom String: PASS")

    cpu_multi = k_cpu.hash(bytes(range(256)) * 40, b"", 32).hex()
    assert cpu_multi == kt256_multi_expected, "CPU multi-block test FAILED"
    print("    [+] Tree-Topology Multi-block: PASS\n")

    # --- GPU Verification ---
    if cp is not None:
        print("[*] Launching Kangaroo12 (GPU Fused-Kernel Path)...")
        k_gpu = Kangaroo12(enable_gpu=True)

        gpu_empty = k_gpu.hash(b"", b"", 32).hex()
        assert gpu_empty == kt256_empty_expected, "GPU empty test FAILED"
        print("    [+] Empty Message: PASS")

        gpu_custom = k_gpu.hash(b"", b"KangarooTwelve", 32).hex()
        assert gpu_custom == kt256_custom_expected, "GPU custom string test FAILED"
        print("    [+] Custom String: PASS")

        gpu_multi = k_gpu.hash(bytes(range(256)) * 40, b"", 32).hex()
        assert gpu_multi == kt256_multi_expected, "GPU multi-block test FAILED"
        print("    [+] Tree-Topology Multi-block: PASS\n")
    else:
        print("--- GPU Path skipped: CuPy backend missing ---\n")

    print("=====================================================")
    print(" SUCCESS: All structural class implementations match!")
    print("=====================================================")

# Comprehensive Benchmark Harness (Batched vs. Single Large Messages)

def run_comprehensive_benchmarks():
    REPS = 5 # Number of loop iterations for statistical smoothing
    
    k_cpu = Kangaroo12(enable_gpu=False)
    k_gpu = Kangaroo12(enable_gpu=True) if cp is not None else None

    # EXPERIMENT 1: Batched Hash Performance (Fixed 8KB Message Size)

    print("\n" + "="*95)
    print(" Experiment 1: Batched Hash Benchmark Results (Fixed Message Size: 8 KB)")
    print("="*95)
    print(f"  {'N Messages':<12} | {'CPU Time (ms)':<13} | {'GPU Time (ms)':<13} | {'CPU (μs/hash)':<15} | {'GPU (μs/hash)':<15} | {'Speedup':<10}")
    print("  " + "─"*12 + "─┼─" + "─"*13 + "─┼─" + "─"*13 + "─┼─" + "─"*15 + "─┼─" + "─"*15 + "─┼─" + "─"*10)

    BATCH_SIZES = [16, 256, 4096]
    FIXED_SIZE = 8192
    batched_msg = bytes(range(256)) * (FIXED_SIZE // 256)

    for N in BATCH_SIZES:
        # Generate target batch payload strings
        batch = [batched_msg for _ in range(N)]

        # --- CPU Loop ---
        cpu_samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for msg in batch:
                k_cpu.hash(msg, b"", 32)
            cpu_samples.append(time.perf_counter() - t0)
        cpu_total_ms = min(cpu_samples) * 1000.0  # Best-of run to eliminate system noise
        cpu_us_per_hash = (cpu_total_ms * 1000.0) / N

        # --- GPU Loop ---
        if k_gpu is not None:
            gpu_samples = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                for msg in batch:
                    k_gpu.hash(msg, b"", 32)
                cp.cuda.Stream.null.synchronize()  # Hardware flush bound safely inside the clock
                gpu_samples.append(time.perf_counter() - t0)
            gpu_total_ms = min(gpu_samples) * 1000.0
            gpu_us_per_hash = (gpu_total_ms * 1000.0) / N
            speedup_str = f"{cpu_total_ms / gpu_total_ms:.2f}x"
            gpu_total_str = f"{gpu_total_ms:.2f}"
            gpu_us_str = f"{gpu_us_per_hash:.2f}"
        else:
            gpu_total_str, gpu_us_str, speedup_str = "N/A", "N/A", "N/A"

        print(f"  {N:<12} | {cpu_total_ms:<13.2f} | {gpu_total_str:<13} | {cpu_us_per_hash:<15.2f} | {gpu_us_str:<15} | {speedup_str:<10}")


    # EXPERIMENT 2: Single Message Scaling Profile (N = 1, Varying Sizes)

    print("\n" + "="*95)
    print(" Experiment 2: Variable Input Size Performance Scaling (Single Message, N = 1)")
    print("="*95)
    print(f"  {'Input Size':<12} | {'CPU Time (ms)':<13} | {'GPU Time (ms)':<13} | {'CPU Throughput':<16} | {'GPU Throughput':<16} | {'Speedup':<10}")
    print("  " + "─"*12 + "─┼─" + "─"*13 + "─┼─" + "─"*13 + "─┼─" + "─"*16 + "─┼─" + "─"*16 + "─┼─" + "─"*10)

    VARYING_SIZES = [
        ("16 B", 16),
        ("8 KB", 8192),
        ("24 KB", 24576),
        ("48 KB", 49152),
        ("1 MB", 1048576),
        ("10 MB", 10485760),
    ]

    for label, size in VARYING_SIZES:
        # Build individual scaling message block
        single_msg = bytes(range(256)) * (size // 256) + bytes(range(size % 256))

        # --- CPU Single Hash ---
        cpu_samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            k_cpu.hash(single_msg, b"", 32)
            cpu_samples.append(time.perf_counter() - t0)
        cpu_ms = min(cpu_samples) * 1000.0
        cpu_mbps = (size / 1e6) / (cpu_ms / 1000.0)

        # --- GPU Single Hash ---
        if k_gpu is not None:
            gpu_samples = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                k_gpu.hash(single_msg, b"", 32)
                cp.cuda.Stream.null.synchronize()
                gpu_samples.append(time.perf_counter() - t0)
            gpu_ms = min(gpu_samples) * 1000.0
            gpu_mbps = (size / 1e6) / (gpu_ms / 1000.0)
            
            gpu_ms_str = f"{gpu_ms:.3f}"
            gpu_mbps_str = f"{gpu_mbps:.2f} MB/s"
            speedup_str = f"{cpu_ms / gpu_ms:.2f}x"
        else:
            gpu_ms_str, gpu_mbps_str, speedup_str = "N/A", "N/A", "N/A"

        print(f"  {label:<12} | {cpu_ms:<13.3f} | {gpu_ms_str:<13} | {cpu_mbps:.2f} MB/s     | {gpu_mbps_str:<16} | {speedup_str:<10}")
    print("="*95 + "\n")


if __name__ == "__main__":
    run_tests()
    run_comprehensive_benchmarks()