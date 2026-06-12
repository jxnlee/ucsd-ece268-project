import time
import random
import numpy as np
import matplotlib.pyplot as plt

try:
    import cupy as cp
except ImportError:
    cp = None

from kangaroo12 import Kangaroo12
from rescue import RescuePrime
from merkle_tree import MerkleTree

# -----------------------------------------------------------------------------
# Patch MerkleTree concatenation to behave consistently for CPU and GPU arrays.
# This keeps bytes concatenation unchanged while using proper array concatenation
# for NumPy/CuPy inputs.
# -----------------------------------------------------------------------------

def patch_merkle_concatenate():
    def _concatenate(self, left, right):
        if cp is not None and isinstance(left, cp.ndarray) and isinstance(right, cp.ndarray):
            return cp.concatenate([left, right])
        if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            return np.concatenate([left, right])
        return left + right

    MerkleTree._concatenate = _concatenate


# -----------------------------------------------------------------------------
# Benchmark helpers
# -----------------------------------------------------------------------------

def build_leaves(num_leaves, leaf_size):
    return [[random.randint(0, 255) for _ in range(leaf_size)] for _ in range(num_leaves)]


def normalize_root(root):
    if cp is not None and isinstance(root, cp.ndarray):
        return root.get().tolist()
    if isinstance(root, np.ndarray):
        return root.tolist()
    if isinstance(root, bytes):
        return list(root)
    return root


def make_hash_fn(name, instance, is_batched):
    if name == "kangaroo12":
        if is_batched:
            return instance.hash_batch
        return lambda leaf: instance.hash(bytes(leaf), b"", 32)
    if name == "rescueprime":
        if is_batched:
            return instance.hash_batch
        return lambda leaf: instance.hash(leaf)
    raise ValueError(f"Unknown algorithm: {name}")


def benchmark_merkle(name, enable_gpu, is_batched, leaf_count, leaf_size, reps=2):
    if name == "kangaroo12":
        instance = Kangaroo12(enable_gpu=enable_gpu)
    elif name == "rescueprime":
        p = 2**61 - 1
        m = 3
        c = 1
        s = 128
        instance = RescuePrime(p, m, c, s, enable_gpu=enable_gpu)
    else:
        raise ValueError(f"Unsupported algorithm: {name}")

    hash_fn = make_hash_fn(name, instance, is_batched)
    leaves = build_leaves(leaf_count, leaf_size)

    # Warmup
    tree = MerkleTree(leaves, hash=hash_fn, enable_gpu=enable_gpu, is_batched=is_batched)
    if enable_gpu and cp is not None:
        cp.cuda.Stream.null.synchronize()
    ref_root = normalize_root(tree.root)

    times = []
    for _ in range(reps):
        start = time.perf_counter()
        tree = MerkleTree(leaves, hash=hash_fn, enable_gpu=enable_gpu, is_batched=is_batched)
        if enable_gpu and cp is not None:
            cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - start)

        if normalize_root(tree.root) != ref_root:
            raise RuntimeError("Merkle tree root changed between runs; check deterministic hash behavior")

    return min(times)


# -----------------------------------------------------------------------------
# Plotting and experiment control
# -----------------------------------------------------------------------------

def run_benchmarks():
    patch_merkle_concatenate()

    sizes = [64, 128, 256, 512, 1024, 2048]
    leaf_size = 8
    reps = 2

    algorithms = ["kangaroo12", "rescueprime"]
    modes = [
        (False, False, "CPU node-by-node"),
        (False, True, "CPU batched level"),
    ]
    if cp is not None:
        modes += [
            (True, False, "GPU node-by-node"),
            (True, True, "GPU batched level"),
        ]

    results = {algo: {label: [] for _, _, label in modes} for algo in algorithms}

    for algo in algorithms:
        for enable_gpu, is_batched, label in modes:
            if enable_gpu and cp is None:
                continue

            for size in sizes:
                print(f"Benchmarking {algo} | {label} | leaves={size}")
                elapsed = benchmark_merkle(
                    name=algo,
                    enable_gpu=enable_gpu,
                    is_batched=is_batched,
                    leaf_count=size,
                    leaf_size=leaf_size,
                    reps=reps,
                )
                results[algo][label].append(elapsed)

    return sizes, results


def plot_results(sizes, results):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    fig.suptitle("Merkle Tree Build Performance: Kangaroo12 vs RescuePrime", fontsize=16)

    for ax, algo in zip(axes, results.keys()):
        for label, times in results[algo].items():
            ax.plot(sizes, times, marker="o", label=label)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Number of leaves")
        ax.set_title(algo.capitalize())
        ax.grid(True, which="both", linestyle="--", alpha=0.5)
        ax.legend()

    axes[0].set_ylabel("Build time (seconds)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("merkle_tree_benchmarks.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    random.seed(0)
    sizes, results = run_benchmarks()
    plot_results(sizes, results)
