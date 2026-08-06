"""
benchmark.py — Comprehensive batched hash + Merkle Tree benchmark suite.

Sweeps over a grid of (batch_size, elements_per_message) for both
RescuePrime and KangarooTwelve, reporting CPU and GPU timing in ms.

Usage:
    python benchmark.py                     # full suite, both hash functions
    python benchmark.py --only rescue       # RescuePrime only
    python benchmark.py --only k12          # KangarooTwelve only
    python benchmark.py --warmup 2          # override warmup iterations (default 1)
    python benchmark.py --csv results.csv   # also write results to a CSV
"""

import argparse
import csv
import random
import sys
import time
from itertools import product
from typing import Callable, Optional

# ── Optional heavy imports (GPU may not be available) ──────────────────────────
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from hash_src.kangaroo12 import Kangaroo12
from hash_src.merkle_tree import MerkleTree
from hash_src.rescue import RescuePrime

# ── Benchmark grid ─────────────────────────────────────────────────────────────
# Tune these lists to taste.
BATCH_SIZES          = [64, 256, 512, 1024, 2048, 4096, 8192, 16384]
ELEMENTS_PER_MESSAGE = [4, 8, 16, 32, 64, 128]

# RescuePrime parameters
RP_P = 2**61 - 1
RP_M = 3
RP_C = 1
RP_S = 128
RP_OUTPUT_LEN = 1

RANDOM_SEED  = 42
WARMUP_ITERS = 1          # number of warm-up hashes before timing


# ── Helpers ────────────────────────────────────────────────────────────────────

def sync_gpu():
    """Block until all outstanding CUDA work is complete."""
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()


def ms(seconds: float) -> float:
    return round(seconds * 1000, 3)


def make_batch(batch_size: int, elements_per_message: int, seed: int = RANDOM_SEED):
    rng = random.Random(seed)
    return [
        [rng.randint(0, 255) for _ in range(elements_per_message)]
        for _ in range(batch_size)
    ]


def time_merkle(
    batch: list,
    hash_fn: Callable,
    enable_gpu: bool,
    warmup_batch: list,
    warmup_iters: int = WARMUP_ITERS,
) -> tuple[float, Optional[list]]:
    """
    Warm up, then time a single MerkleTree construction.
    Returns (elapsed_ms, root_as_plain_list).
    """
    # Warm-up (not timed)
    for _ in range(warmup_iters):
        _ = MerkleTree(warmup_batch, hash=hash_fn, enable_gpu=enable_gpu, is_batched=True)
    if enable_gpu:
        sync_gpu()

    t0 = time.perf_counter()
    tree = MerkleTree(batch, hash=hash_fn, enable_gpu=enable_gpu, is_batched=True)
    root = tree.root
    if enable_gpu:
        sync_gpu()
    elapsed = time.perf_counter() - t0

    # Normalise root to a plain Python list for comparison
    try:
        root_list = root.get().tolist() if HAS_CUPY and isinstance(root, cp.ndarray) else (
            root.tolist() if hasattr(root, "tolist") else list(root)
        )
    except Exception:
        root_list = None

    return ms(elapsed), root_list


def print_table_header(label: str):
    print()
    print(f"{'═'*80}")
    print(f"  {label}")
    print(f"{'═'*80}")
    print(
        f"  {'Batch':>8}  {'Elems/msg':>9}  {'CPU (ms)':>10}  {'GPU (ms)':>10}  "
        f"{'Speedup':>8}  {'Match':>6}"
    )
    print(f"  {'-'*8}  {'-'*9}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")


def print_row(batch_size, elems, cpu_ms, gpu_ms, match):
    if gpu_ms and gpu_ms > 0:
        speedup = f"{cpu_ms / gpu_ms:.2f}x"
    else:
        speedup = "N/A"
    match_str = "✓" if match else "✗" if match is False else "—"
    print(
        f"  {batch_size:>8,}  {elems:>9}  {cpu_ms:>10.1f}  "
        f"{str(gpu_ms) + '' if gpu_ms is None else f'{gpu_ms:>10.1f}'}  "
        f"{speedup:>8}  {match_str:>6}"
    )


def _fmt(val, width=10):
    if val is None:
        return f"{'N/A':>{width}}"
    return f"{val:>{width}.1f}"


def print_row_fmt(batch_size, elems, cpu_ms, gpu_ms, match):
    speedup = f"{cpu_ms / gpu_ms:.2f}x" if (gpu_ms and gpu_ms > 0) else "N/A"
    match_str = "✓" if match else ("✗" if match is False else "—")
    print(
        f"  {batch_size:>8,}  {elems:>9}  {_fmt(cpu_ms)}  {_fmt(gpu_ms)}  "
        f"{speedup:>8}  {match_str:>6}"
    )


# ── Main benchmark runners ─────────────────────────────────────────────────────

def bench_rescue(
    batch_sizes: list,
    elem_sizes: list,
    warmup_iters: int,
    csv_rows: list,
):
    print("\nInitialising RescuePrime instances …")
    cpu_rp = RescuePrime(RP_P, RP_M, RP_C, RP_S, enable_gpu=False)
    gpu_rp = None
    if HAS_CUPY:
        try:
            gpu_rp = RescuePrime(RP_P, RP_M, RP_C, RP_S, enable_gpu=True)
        except Exception as exc:
            print(f"  [!] GPU RescuePrime init failed: {exc}")

    print_table_header("RescuePrime — Batched Merkle Tree")

    for batch_size, elems in product(batch_sizes, elem_sizes):
        batch       = make_batch(batch_size, elems)
        warmup_sml  = make_batch(min(batch_size, 64), elems)

        cpu_ms, cpu_root = time_merkle(
            batch, cpu_rp.hash_batch, enable_gpu=False,
            warmup_batch=warmup_sml, warmup_iters=warmup_iters,
        )

        gpu_ms, gpu_root = None, None
        if gpu_rp is not None:
            gpu_ms, gpu_root = time_merkle(
                batch, gpu_rp.hash_batch, enable_gpu=True,
                warmup_batch=warmup_sml, warmup_iters=warmup_iters,
            )

        match = (cpu_root == gpu_root) if (cpu_root is not None and gpu_root is not None) else None
        print_row_fmt(batch_size, elems, cpu_ms, gpu_ms, match)

        csv_rows.append({
            "hash":             "RescuePrime",
            "batch_size":       batch_size,
            "elements_per_msg": elems,
            "cpu_ms":           cpu_ms,
            "gpu_ms":           gpu_ms if gpu_ms is not None else "",
            "speedup":          f"{cpu_ms/gpu_ms:.2f}" if gpu_ms else "",
            "match":            "" if match is None else ("PASS" if match else "FAIL"),
        })


def bench_k12(
    batch_sizes: list,
    elem_sizes: list,
    warmup_iters: int,
    csv_rows: list,
):
    print("\nInitialising KangarooTwelve instances …")
    cpu_k12 = Kangaroo12(enable_gpu=False)
    gpu_k12 = None
    if HAS_CUPY:
        try:
            gpu_k12 = Kangaroo12(enable_gpu=True)
        except Exception as exc:
            print(f"  [!] GPU KangarooTwelve init failed: {exc}")

    print_table_header("KangarooTwelve — Batched Merkle Tree")

    for batch_size, elems in product(batch_sizes, elem_sizes):
        batch      = make_batch(batch_size, elems)
        warmup_sml = make_batch(min(batch_size, 64), elems)

        cpu_ms, cpu_root = time_merkle(
            batch, cpu_k12.hash_batch, enable_gpu=False,
            warmup_batch=warmup_sml, warmup_iters=warmup_iters,
        )

        gpu_ms, gpu_root = None, None
        if gpu_k12 is not None:
            gpu_ms, gpu_root = time_merkle(
                batch, gpu_k12.hash_batch, enable_gpu=True,
                warmup_batch=warmup_sml, warmup_iters=warmup_iters,
            )

        match = (cpu_root == gpu_root) if (cpu_root is not None and gpu_root is not None) else None
        print_row_fmt(batch_size, elems, cpu_ms, gpu_ms, match)

        csv_rows.append({
            "hash":             "KangarooTwelve",
            "batch_size":       batch_size,
            "elements_per_msg": elems,
            "cpu_ms":           cpu_ms,
            "gpu_ms":           gpu_ms if gpu_ms is not None else "",
            "speedup":          f"{cpu_ms/gpu_ms:.2f}" if gpu_ms else "",
            "match":            "" if match is None else ("PASS" if match else "FAIL"),
        })


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batched hash + Merkle Tree benchmark suite")
    parser.add_argument(
        "--only", choices=["rescue", "k12"], default=None,
        help="Run only one hash function (default: both)",
    )
    parser.add_argument(
        "--warmup", type=int, default=WARMUP_ITERS,
        help=f"Warm-up iterations before timing (default: {WARMUP_ITERS})",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Optional path to write results as CSV (e.g. results.csv)",
    )
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=BATCH_SIZES,
        metavar="N",
        help="Override the list of batch sizes to benchmark",
    )
    parser.add_argument(
        "--elem-sizes", type=int, nargs="+", default=ELEMENTS_PER_MESSAGE,
        metavar="E",
        help="Override the list of elements-per-message to benchmark",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("  Batched Hash + Merkle Tree Benchmark Suite")
    print("=" * 80)
    print(f"  Batch sizes          : {args.batch_sizes}")
    print(f"  Elements per message : {args.elem_sizes}")
    print(f"  Warm-up iterations   : {args.warmup}")
    print(f"  GPU available        : {'Yes (CuPy)' if HAS_CUPY else 'No'}")
    if args.only:
        print(f"  Running              : {args.only} only")

    csv_rows: list[dict] = []

    if args.only != "k12":
        bench_rescue(args.batch_sizes, args.elem_sizes, args.warmup, csv_rows)

    if args.only != "rescue":
        bench_k12(args.batch_sizes, args.elem_sizes, args.warmup, csv_rows)

    print()
    print("=" * 80)
    print("  Benchmark complete.")
    print("=" * 80)

    if args.csv:
        fieldnames = ["hash", "batch_size", "elements_per_msg", "cpu_ms", "gpu_ms", "speedup", "match"]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n  Results written to: {args.csv}")


if __name__ == "__main__":
    main()