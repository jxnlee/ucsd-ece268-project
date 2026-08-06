import time
from hash_src.merkle_tree import MerkleTree
from hash_src.kangaroo12 import Kangaroo12
import numpy as np
import time

try:
    import cupy as cp
except ImportError:
    cp = None

def _rotate64(a, n):
    n %= 64
    return ((a << n) | (a >> (64 - n))) & 0xFFFFFFFFFFFFFFFF

_REF_RC = [
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
    0x8000000000008003, 0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A, 0x8000000080008081,
    0x8000000000008080, 0x0000000080000001, 0x8000000080008008,
]

def _ref_keccak_p1600_12(state: bytearray) -> bytearray:
    lanes = [int.from_bytes(state[8*i:8*i+8], 'little') for i in range(25)]
    for rc in _REF_RC:
        C = [lanes[x] ^ lanes[x+5] ^ lanes[x+10] ^ lanes[x+15] ^ lanes[x+20]
             for x in range(5)]
        D = [C[(x-1) % 5] ^ _rotate64(C[(x+1) % 5], 1) for x in range(5)]
        lanes = [lanes[i] ^ D[i % 5] for i in range(25)]

        new_lanes = lanes[:]
        x, y = 1, 0
        cur = lanes[x + 5*y]
        for t in range(24):
            nx, ny = y, (2*x + 3*y) % 5
            new_lanes[nx + 5*ny] = _rotate64(cur, (t+1)*(t+2)//2)
            cur = lanes[nx + 5*ny]
            x, y = nx, ny
        lanes = new_lanes

        for y in range(5):
            row = [lanes[x + 5*y] for x in range(5)]
            for x in range(5):
                lanes[x + 5*y] = row[x] ^ ((~row[(x+1)%5] & 0xFFFFFFFFFFFFFFFF) & row[(x+2)%5])
        lanes[0] ^= rc

    out = bytearray(200)
    for i in range(25):
        out[8*i:8*i+8] = lanes[i].to_bytes(8, 'little')
    return out

def _ref_turboshake256(msg: bytes, sep: int, out_len: int) -> bytes:
    RATE  = 136
    state = bytearray(200)
    data  = msg + bytes([sep])
    off   = 0
    while off + RATE <= len(data):
        for i in range(RATE):
            state[i] ^= data[off + i]
        state = _ref_keccak_p1600_12(state)
        off  += RATE
    rem = len(data) - off
    for i in range(rem):
        state[i] ^= data[off + i]
    state[RATE - 1] ^= 0x80
    state = _ref_keccak_p1600_12(state)
    out = bytearray()
    while out_len > RATE:
        out.extend(state[:RATE])
        out_len -= RATE
        state = _ref_keccak_p1600_12(state)
    out.extend(state[:out_len])
    return bytes(out)

def _ref_length_encode(n: int) -> bytes:
    if n == 0:
        return b'\x00'
    s = bytearray()
    while n:
        s.insert(0, n & 0xFF)
        n >>= 8
    s.append(len(s))
    return bytes(s)

def ref_k12(msg: bytes, custom: bytes, out_len: int) -> bytes:
    """Reference KT256 from RFC 9497 — used as oracle only, never timed."""
    CV_LEN = 64
    CHUNK  = 8192
    s = msg + custom + _ref_length_encode(len(custom))
    if len(s) <= CHUNK:
        return _ref_turboshake256(s, 0x07, out_len)
    fin = bytearray(s[:CHUNK]) + bytearray([0x03] + [0x00] * 7)
    off, num_blocks = CHUNK, 0
    while off < len(s):
        fin.extend(_ref_turboshake256(s[off: off + CHUNK], 0x0B, CV_LEN))
        num_blocks += 1
        off += CHUNK
    fin.extend(_ref_length_encode(num_blocks))
    fin.extend([0xFF, 0xFF])
    return _ref_turboshake256(bytes(fin), 0x06, out_len)


# Build a reference Merkle root using ref_k12 as the hash function.

def ref_merkle_root(leaf_bytes):
    """Compute the Merkle root using the RFC 9497 reference K12."""
    def h(data):
        return ref_k12(data, b"", 32)

    current = [h(leaf) for leaf in leaf_bytes]
    while len(current) > 1:
        if len(current) % 2 != 0:
            current.append(current[-1])
        current = [h(current[i] + current[i+1]) for i in range(0, len(current), 2)]
    return current[0]


# Build + time one tree for a given backend

def benchmark_tree(leaf_bytes, k12_cpu, has_gpu):
    # --- CPU ---
    t0 = time.perf_counter()
    tree_cpu = MerkleTree(
        data=leaf_bytes,
        hash=lambda x: k12_cpu.hash(x, b"", 32),
        enable_gpu=False,
        is_batched=False,
    )
    cpu_ms = (time.perf_counter() - t0) * 1000.0

    if not has_gpu:
        return cpu_ms, None, tree_cpu.root

    # --- GPU (parallel kernel path) ---
    # MerkleTree.__init__ with enable_gpu=True uses the batched CUDA kernels
    # directly — no wrapper needed, no manual __new__ construction.
    t0 = time.perf_counter()
    tree_gpu = MerkleTree(
        data=leaf_bytes,
        enable_gpu=True,
    )
    gpu_ms = (time.perf_counter() - t0) * 1000.0

    return cpu_ms, gpu_ms, tree_cpu.root, tree_gpu.root


# Main

def main():
    # ── Sanity-check the reference implementation first ─────────────────────
    print("Verifying reference K12 against RFC 9497 test vectors...")
    ref_vectors = [
        (b"",                    b"",               32, "b23d2e9cea9f4904e02bec06817fc10ce38ce8e93ef4c89e6537076af8646404"),
        (b"",                    b"KangarooTwelve", 32, "bbe187fa1c6d40617ef4023ec8ddb414d61d07a5f0db114015c11924faa24233"),
        (bytes(range(256)) * 40, b"",               32, "d3b501e04a1c9a430f398e937ad57be691aa35b107d0ba4cdfdcaac3aa2ab534"),
    ]
    for msg, custom, olen, expected in ref_vectors:
        got = ref_k12(msg, custom, olen).hex()
        status = "PASS" if got == expected else "FAIL"
        print(f"  [{status}] {expected[:24]}...")
    print()

    LEAF_COUNTS = [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 63, 64,
                   100, 128, 255, 256, 1000, 1024, 2048, 64000]

    k12_cpu = Kangaroo12(enable_gpu=False)
    has_gpu = cp is not None

    # ── header ──────────────────────────────────────────────────────────────
    col_w = 93 if has_gpu else 70
    print("=" * col_w)
    print("  Merkle Tree Benchmark + Reference Root Comparison")
    print("=" * col_w)
    if has_gpu:
        print(f"  {'Leaves':>7}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  {'Speedup':>9}  {'vs Ref (CPU)':>13}  {'vs Ref (GPU)':>13}")
    else:
        print(f"  {'Leaves':>7}  {'CPU (ms)':>10}  {'Speedup':>9}  {'vs Ref (CPU)':>13}")
    print("  " + "─" * (col_w - 2))

    for n in LEAF_COUNTS:
        leaf_bytes = [bytes([i % 256]) * 32 for i in range(n)]

        # Reference root (not timed — this is the oracle)
        oracle_root = ref_merkle_root(leaf_bytes)

        result = benchmark_tree(leaf_bytes, k12_cpu, has_gpu)

        if has_gpu:
            cpu_ms, gpu_ms, cpu_root, gpu_root = result
            speedup_str  = f"{cpu_ms / gpu_ms:.2f}x" if gpu_ms and gpu_ms > 0 else "—"
            gpu_ms_str   = f"{gpu_ms:>10.2f}" if gpu_ms is not None else f"{'N/A':>10}"
            cpu_vs_ref   = "MATCH" if cpu_root == oracle_root else "DIFF"
            gpu_vs_ref   = "MATCH" if gpu_root == oracle_root else "DIFF"
            print(f"  {n:>7}  {cpu_ms:>10.2f}  {gpu_ms_str}  {speedup_str:>9}  {cpu_vs_ref:>13}  {gpu_vs_ref:>13}")
        else:
            cpu_ms, _, cpu_root = result
            cpu_vs_ref = "MATCH" if cpu_root == oracle_root else "DIFF"
            print(f"  {n:>7}  {cpu_ms:>10.2f}  {'N/A':>9}  {cpu_vs_ref:>13}")

    print("=" * col_w)


if __name__ == "__main__":
    main()