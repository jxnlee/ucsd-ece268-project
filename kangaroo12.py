import numpy as np
import time

try:
    import cupy as cp
except ImportError:
    cp = None

_RC = [
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

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

CUDA_KECCAK_SRC = r"""
extern "C" {

__device__ __inline__ unsigned long long rot_l64(unsigned long long a, int n) {
    n &= 63;
    if (n == 0) return a;
    return (a << n) | (a >> (64 - n));
}

__device__ __inline__ void keccak_p_1600_lanes(unsigned long long lane[25]) {
    const unsigned long long RC[12] = {
        0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
        0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
        0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
    };
    for (int r = 0; r < 12; r++) {
        unsigned long long C[5];
        for (int x = 0; x < 5; x++)
            C[x] = lane[x] ^ lane[x+5] ^ lane[x+10] ^ lane[x+15] ^ lane[x+20];
        for (int x = 0; x < 5; x++) {
            unsigned long long D = C[(x+4)%5] ^ rot_l64(C[(x+1)%5], 1);
            for (int y = 0; y < 5; y++) lane[x + 5*y] ^= D;
        }
        int x = 1, y = 0;
        unsigned long long current = lane[x + 5*y];
        for (int t = 0; t < 24; t++) {
            int new_x = y, new_y = (2*x + 3*y) % 5;
            x = new_x; y = new_y;
            int rot = (t + 1) * (t + 2) / 2;
            unsigned long long temp = lane[x + 5*y];
            lane[x + 5*y] = rot_l64(current, rot);
            current = temp;
        }
        for (int y = 0; y < 5; y++) {
            unsigned long long T[5];
            for (int x = 0; x < 5; x++) T[x] = lane[x + 5*y];
            for (int x = 0; x < 5; x++)
                lane[x + 5*y] = T[x] ^ ((~T[(x+1)%5]) & T[(x+2)%5]);
        }
        lane[0] ^= RC[r];
    }
}

__device__ void turboshake256_core(
    const unsigned char* __restrict__ msg,
    int msg_len,
    unsigned char* __restrict__ output,
    int output_len
) {
    const int RATE = 136;
    unsigned long long lane[25];
    for (int i = 0; i < 25; i++) lane[i] = 0ULL;

    int offset = 0;
    while (offset + RATE <= msg_len) {
        for (int i = 0; i < 17; i++) {
            unsigned long long word = 0ULL;
            for (int b = 0; b < 8; b++)
                word |= ((unsigned long long)msg[offset + 8*i + b]) << (8*b);
            lane[i] ^= word;
        }
        keccak_p_1600_lanes(lane);
        offset += RATE;
    }

    int rem = msg_len - offset;
    for (int k = 0; k < rem; k++) {
        int lane_idx = k / 8, shift = (k % 8) * 8;
        lane[lane_idx] ^= ((unsigned long long)msg[offset + k]) << shift;
    }
    lane[16] ^= 0x80ULL << (7 * 8);
    keccak_p_1600_lanes(lane);

    int produced = 0;
    while (produced < output_len) {
        int take = output_len - produced;
        if (take > RATE) take = RATE;
        for (int k = 0; k < take; k++) {
            int lane_idx = k / 8, shift = (k % 8) * 8;
            output[produced + k] = (unsigned char)((lane[lane_idx] >> shift) & 0xFF);
        }
        produced += take;
        if (produced < output_len) keccak_p_1600_lanes(lane);
    }
}

__global__ void turboshake256_batched(
    const unsigned char* __restrict__ msgs_flat,
    const int*           __restrict__ msg_offsets,
    const int*           __restrict__ msg_lens,
    unsigned char*       __restrict__ outputs_flat,
    const int*           __restrict__ out_offsets,
    int output_len,
    int n_msgs
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_msgs) return;
    turboshake256_core(
        msgs_flat    + msg_offsets[tid],
        msg_lens[tid],
        outputs_flat + out_offsets[tid],
        output_len
    );
}

} // extern "C"
"""


class Kangaroo12:
    def __init__(self, enable_gpu=False):
        if enable_gpu and cp is None:
            raise ImportError("CuPy backend requested but not available.")

        self.xp = cp if enable_gpu else np
        self.enable_gpu = enable_gpu

        self.RC  = self.xp.array(_RC,  dtype=self.xp.uint64)
        self.RHO = self.xp.array(_RHO, dtype=self.xp.uint64)
        self.PI  = self.xp.array(_PI,  dtype=self.xp.uint64)

        if self.enable_gpu:
            self.mod = cp.RawModule(code=CUDA_KECCAK_SRC)
            self.cu_turboshake_batch = self.mod.get_function("turboshake256_batched")

    # ------------------------------------------------------------------
    # CPU helpers
    # ------------------------------------------------------------------

    def _rotate(self, a, n):
        n %= 64
        return int(((a << n) | (a >> (64 - n))) & 0xFFFFFFFFFFFFFFFF)

    def _keccak_p_1600(self, state: bytearray) -> bytes:
        xp = self.xp
        lane = xp.array(
            [int.from_bytes(state[8*i: 8*i+8], 'little') for i in range(25)],
            dtype=xp.uint64,
        )
        for r in range(12):
            C = xp.array([
                int(lane[x] ^ lane[x+5] ^ lane[x+10] ^ lane[x+15] ^ lane[x+20])
                for x in range(5)
            ], dtype=xp.uint64)
            D = xp.array([
                int(C[(x-1)%5]) ^ self._rotate(int(C[(x+1)%5]), 1)
                for x in range(5)
            ], dtype=xp.uint64)
            for x in range(5):
                for y in range(5):
                    lane[x + 5*y] = int(lane[x + 5*y]) ^ int(D[x])
            rotated_lanes = xp.array([
                self._rotate(int(lane[i]), int(self.RHO[i])) for i in range(25)
            ], dtype=xp.uint64)
            lane = rotated_lanes[self.PI]
            for y_pos in range(5):
                T = [int(lane[x_pos + 5*y_pos]) for x_pos in range(5)]
                for x_pos in range(5):
                    lane[x_pos + 5*y_pos] = T[x_pos] ^ ((~T[(x_pos+1)%5] & 0xFFFFFFFFFFFFFFFF) & T[(x_pos+2)%5])
            lane[0] = int(lane[0]) ^ int(self.RC[r])
        out = bytearray(200)
        for i in range(25):
            out[8*i: 8*i+8] = int(lane[i]).to_bytes(8, 'little')
        return bytes(out)

    def _turbo_shake256_cpu(self, message: bytes, separation_byte: int, output_byte_len: int) -> bytes:
        RATE  = 136
        state = bytearray(200)
        data  = bytes(message) + bytes([separation_byte])
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

    # ------------------------------------------------------------------
    # TurboShake256 — unified entry point
    # Routes to batched GPU kernel (n=1) or CPU path.
    # ------------------------------------------------------------------

    def _turbo_shake256(self, message: bytes, separation_byte: int, output_byte_len: int) -> bytes:
        if self.enable_gpu:
            return self._turbo_shake256_batch_gpu([message], separation_byte, output_byte_len)[0]
        return self._turbo_shake256_cpu(message, separation_byte, output_byte_len)

    # ------------------------------------------------------------------
    # TurboShake256 — batched GPU path
    # ------------------------------------------------------------------

    def _turbo_shake256_batch_gpu(self, messages: list, separation_byte: int, output_byte_len: int) -> list:
        n = len(messages)
        if n == 0:
            return []

        payloads    = [bytes(m) + bytes([separation_byte]) for m in messages]
        msg_lens    = [len(p) for p in payloads]
        msg_offsets = [0] * n
        for i in range(1, n):
            msg_offsets[i] = msg_offsets[i-1] + msg_lens[i-1]

        out_offsets     = [i * output_byte_len for i in range(n)]
        total_msg_bytes = msg_offsets[-1] + msg_lens[-1]
        total_out_bytes = n * output_byte_len

        flat_msgs = bytearray(total_msg_bytes)
        for i, p in enumerate(payloads):
            flat_msgs[msg_offsets[i]: msg_offsets[i] + msg_lens[i]] = p

        gpu_msgs_flat   = cp.array(np.frombuffer(flat_msgs, dtype=np.uint8))
        gpu_msg_offsets = cp.array(np.array(msg_offsets, dtype=np.int32))
        gpu_msg_lens    = cp.array(np.array(msg_lens,    dtype=np.int32))
        gpu_out_flat    = cp.zeros(total_out_bytes, dtype=cp.uint8)
        gpu_out_offsets = cp.array(np.array(out_offsets, dtype=np.int32))

        threads = 128
        blocks  = (n + threads - 1) // threads
        self.cu_turboshake_batch(
            (blocks,), (threads,),
            (gpu_msgs_flat, gpu_msg_offsets, gpu_msg_lens,
             gpu_out_flat, gpu_out_offsets,
             np.int32(output_byte_len), np.int32(n)),
        )

        out_host = gpu_out_flat.get()
        return [bytes(out_host[out_offsets[i]: out_offsets[i] + output_byte_len]) for i in range(n)]

    # ------------------------------------------------------------------
    # Length encoding helper
    # ------------------------------------------------------------------

    def _length_encode(self, val: int) -> bytes:
        if val == 0:
            return b'\x00'
        s = bytearray()
        while val > 0:
            s.insert(0, val % 256)
            val //= 256
        s.append(len(s))
        return bytes(s)

    # ------------------------------------------------------------------
    # Single-message hash
    # ------------------------------------------------------------------

    def hash(self, input_message: bytes, custom_string: bytes=b"", output_byte_len: int=32) -> bytes:
        CV_LEN = 64
        s = bytes(input_message) + bytes(custom_string) + self._length_encode(len(custom_string))

        if len(s) <= 8192:
            return self._turbo_shake256(s, 0x07, output_byte_len)

        fin = bytearray(s[:8192]) + bytearray([0x03] + [0x00] * 7)
        offset, num_blocks = 8192, 0
        while offset < len(s):
            block_size = min(len(s) - offset, 8192)
            cv = self._turbo_shake256(s[offset: offset + block_size], 0x0B, CV_LEN)
            fin.extend(cv)
            num_blocks += 1
            offset += block_size
        fin.extend(self._length_encode(num_blocks))
        fin.extend([0xFF, 0xFF])
        return self._turbo_shake256(bytes(fin), 0x06, output_byte_len)

    # ------------------------------------------------------------------
    # Batched hash
    # ------------------------------------------------------------------

    def hash_batch(self, messages: list, custom_string: bytes=b"", output_byte_len: int=32) -> list:
        if not self.enable_gpu:
            return [self.hash(m, custom_string, output_byte_len) for m in messages]

        CV_LEN = 64
        short_indices, short_payloads, long_indices = [], [], []

        for idx, msg in enumerate(messages):
            s = bytes(msg) + custom_string + self._length_encode(len(custom_string))
            if len(s) <= 8192:
                short_indices.append(idx)
                short_payloads.append(s)
            else:
                long_indices.append(idx)

        results = [None] * len(messages)

        if short_payloads:
            digests = self._turbo_shake256_batch_gpu(short_payloads, 0x07, output_byte_len)
            for i, digest in zip(short_indices, digests):
                results[i] = digest

        for idx in long_indices:
            results[idx] = self.hash(messages[idx][0], messages[idx][1], output_byte_len)

        return results


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests():
    print("--- Executing Kangaroo12 Engine Validation Suite ---\n")

    kt256_empty_expected  = "b23d2e9cea9f4904e02bec06817fc10ce38ce8e93ef4c89e6537076af8646404"
    kt256_custom_expected = "bbe187fa1c6d40617ef4023ec8ddb414d61d07a5f0db114015c11924faa24233"
    kt256_multi_expected  = "d3b501e04a1c9a430f398e937ad57be691aa35b107d0ba4cdfdcaac3aa2ab534"

    print("[*] CPU Path...")
    k_cpu = Kangaroo12(enable_gpu=False)
    assert k_cpu.hash(b"", b"", 32).hex() == kt256_empty_expected,              "CPU empty FAILED"
    print("    [+] Empty Message: PASS")
    assert k_cpu.hash(b"", b"KangarooTwelve", 32).hex() == kt256_custom_expected, "CPU custom FAILED"
    print("    [+] Custom String: PASS")
    assert k_cpu.hash(bytes(range(256)) * 40, b"", 32).hex() == kt256_multi_expected, "CPU multi FAILED"
    print("    [+] Tree-Topology Multi-block: PASS\n")

    if cp is not None:
        print("[*] GPU Batched Path...")
        k_gpu    = Kangaroo12(enable_gpu=True)
        batch_in = [
            (b"",                    b""),
            (b"",                    b"KangarooTwelve"),
            (bytes(range(256)) * 40, b""),
        ]
        expected = [kt256_empty_expected, kt256_custom_expected, kt256_multi_expected]
        results  = k_gpu.hash_batch(batch_in, 32)
        labels   = ["Empty Message", "Custom String", "Tree-Topology Multi-block"]
        for lbl, res, exp in zip(labels, results, expected):
            assert res.hex() == exp, f"GPU batch {lbl} FAILED\n  got: {res.hex()}\n  exp: {exp}"
            print(f"    [+] {lbl}: PASS")
        print()
    else:
        print("--- GPU Path skipped: CuPy not available ---\n")

    print("=====================================================")
    print(" SUCCESS: All tests passed!")
    print("=====================================================")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def run_comprehensive_benchmarks():
    REPS  = 5
    k_cpu = Kangaroo12(enable_gpu=False)
    k_gpu = Kangaroo12(enable_gpu=True) if cp is not None else None

    print("\n" + "="*90)
    print(" Experiment 1: Batched Hash Benchmark (Fixed Message Size: 8 KB)")
    print("="*90)
    hdr = f"  {'N Messages':<12} | {'CPU Time (ms)':<13} | {'GPU Batch (ms)':<14} | {'CPU us/hash':<12} | {'GPU us/hash':<12} | {'Speedup':<10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    FIXED_SIZE  = 8192
    batched_msg = bytes(range(256)) * (FIXED_SIZE // 256)

    for N in [16, 256, 4096]:
        batch_pairs = [(batched_msg, b"")] * N

        cpu_samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            for msg, cust in batch_pairs:
                k_cpu.hash(msg, cust, 32)
            cpu_samples.append(time.perf_counter() - t0)
        cpu_ms = min(cpu_samples) * 1000.0
        cpu_us = (cpu_ms * 1000.0) / N

        if k_gpu is not None:
            gpu_bat_samples = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                k_gpu.hash_batch(batch_pairs, 32)
                cp.cuda.Stream.null.synchronize()
                gpu_bat_samples.append(time.perf_counter() - t0)
            gpu_ms = min(gpu_bat_samples) * 1000.0
            gpu_us = (gpu_ms * 1000.0) / N
            print(f"  {N:<12} | {cpu_ms:<13.2f} | {gpu_ms:<14.2f} | {cpu_us:<12.2f} | {gpu_us:<12.2f} | {cpu_ms/gpu_ms:.2f}x")
        else:
            print(f"  {N:<12} | {cpu_ms:<13.2f} | {'N/A':<14} | {cpu_us:<12.2f} | {'N/A':<12} | {'N/A':<10}")

    print("\n" + "="*95)
    print(" Experiment 2: Variable Input Size Scaling (Single Message)")
    print("="*95)
    print(f"  {'Input Size':<12} | {'CPU Time (ms)':<13} | {'GPU Time (ms)':<13} | {'CPU Throughput':<16} | {'GPU Throughput':<16} | {'Speedup':<10}")
    print("  " + "-"*92)

    for label, size in [("16 B", 16), ("8 KB", 8192), ("24 KB", 24576),
                         ("48 KB", 49152), ("1 MB", 1048576), ("10 MB", 10485760)]:
        single_msg  = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
        cpu_samples = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            k_cpu.hash(single_msg, b"", 32)
            cpu_samples.append(time.perf_counter() - t0)
        cpu_ms   = min(cpu_samples) * 1000.0
        cpu_mbps = (size / 1e6) / (cpu_ms / 1000.0)

        if k_gpu is not None:
            gpu_samples = []
            for _ in range(REPS):
                t0 = time.perf_counter()
                k_gpu.hash(single_msg, b"", 32)
                cp.cuda.Stream.null.synchronize()
                gpu_samples.append(time.perf_counter() - t0)
            gpu_ms   = min(gpu_samples) * 1000.0
            gpu_mbps = (size / 1e6) / (gpu_ms / 1000.0)
            print(f"  {label:<12} | {cpu_ms:<13.3f} | {gpu_ms:<13.3f} | {cpu_mbps:.2f} MB/s     | {gpu_mbps:.2f} MB/s       | {cpu_ms/gpu_ms:.2f}x")
        else:
            print(f"  {label:<12} | {cpu_ms:<13.3f} | {'N/A':<13} | {cpu_mbps:.2f} MB/s     | {'N/A':<16} | {'N/A':<10}")

    print("="*95 + "\n")


if __name__ == "__main__":
    run_tests()
    run_comprehensive_benchmarks()