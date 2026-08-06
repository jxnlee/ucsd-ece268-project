import random

try:
    import cupy as cp
except ImportError:
    cp = None

from hash_src.kangaroo12 import Kangaroo12
from hash_src.rescue import RescuePrime
from hash_src.merkle_tree import MerkleTree


def normalize_root(root):
    if cp is not None and isinstance(root, cp.ndarray):
        return root.get()
    return root


def make_hash_function(algo, instance, is_batched):
    if algo == "kangaroo12":
        if is_batched:
            return instance.hash_batch
        return lambda leaf: instance.hash(bytes(leaf), b"", 32)

    if algo == "rescueprime":
        if is_batched:
            return instance.hash_batch
        return lambda leaf: instance.hash(leaf)

    raise ValueError(f"Unsupported algorithm: {algo}")


def demo_proof(algo, enable_gpu, is_batched):
    print("=" * 80)
    print(f"Demo: {algo.capitalize()} | {'GPU' if enable_gpu else 'CPU'} | {'Batched' if is_batched else 'Per-node'}")

    leaves = [[random.randint(0, 255) for _ in range(8)] for _ in range(16)]

    if algo == "kangaroo12":
        instance = Kangaroo12(enable_gpu=enable_gpu)
    else:
        p = 2**61 - 1
        m = 3
        c = 1
        s = 128
        instance = RescuePrime(p, m, c, s, enable_gpu=enable_gpu)

    hash_fn = make_hash_function(algo, instance, is_batched)
    tree = MerkleTree(leaves, hash=hash_fn, enable_gpu=enable_gpu, is_batched=is_batched)
    root = normalize_root(tree.root)

    print(f"Root type: {type(tree.root).__name__}")
    print(f"Root value: {root if isinstance(root, (bytes, list, tuple)) else root.tolist() if hasattr(root, 'tolist') else root}")

    proof_index = 5
    proof = tree.get_proof(proof_index)
    print(f"Proof for leaf index {proof_index} (length {len(proof)}):")
    for i, (sibling, is_left) in enumerate(proof):
        print(f"  level {i}: sibling on {'left' if is_left else 'right'}, type {type(sibling).__name__}")

    valid = tree.verify_proof(leaves[proof_index], proof, tree.root)
    print(f"Proof valid: {valid}")

    # Show invalid proof if leaf changes
    bad_leaf = leaves[proof_index].copy()
    bad_leaf[0] = (bad_leaf[0] + 1) % 256
    invalid = tree.verify_proof(bad_leaf, proof, tree.root)
    print(f"Proof valid after changing leaf content: {invalid}")
    print()


if __name__ == "__main__":
    random.seed(0)
    for algo in ["kangaroo12", "rescueprime"]:
        for enable_gpu in [False, True]:
            if enable_gpu and cp is None:
                print(f"Skipping GPU demo for {algo}: CuPy not available")
                continue
            demo_proof(algo, enable_gpu, is_batched=False)
            demo_proof(algo, enable_gpu, is_batched=True)
