from hash_src.kangaroo12 import Kangaroo12
from hash_src.merkle_tree import MerkleTree
from hash_src.rescue import RescuePrime

import random
import time

p = 2**61 - 1
m = 3
c = 1
s = 128
output_len = 1

random.seed(0)

print("Now Running Rescue Prime Benchmarks!")

cpu_rp = RescuePrime(p, m , c, s, enable_gpu=False)
gpu_rp = RescuePrime(p, m, c, s, enable_gpu=True)

print("Generating Batched Merkle Tree Test")
batch_sz = 8192
elements_per_msg = 8
test_batch = [[random.randint(0, 255) for _ in range(elements_per_msg)] for _ in range(batch_sz)]

print(f"Starting CPU batched Merkle Tree Test")
start_cpu = time.perf_counter()
cpu_tree = MerkleTree(test_batch, hash=cpu_rp.hash_batch, enable_gpu=False, is_batched=True)
cpu_result = cpu_tree.root.tolist()
cpu_time = time.perf_counter() - start_cpu
print (f"CPU Batch Completed in: {cpu_time} seconds with hash {cpu_result}")

print(f"Starting GPU batched Merkle Tree Test")
start_gpu = time.perf_counter()
gpu_tree = MerkleTree(test_batch, hash=gpu_rp.hash_batch, enable_gpu=True, is_batched=True)
gpu_result = gpu_tree.root.get().tolist()
gpu_time = time.perf_counter() - start_gpu
print (f"GPU Batch Completed in: {gpu_time} seconds with hash {gpu_result}")

if cpu_result == gpu_result:
    print("[PASS] Cryptographic Equivalence Maintained across all hashes!")
    speedup = cpu_time / gpu_time
    print(f"[Speedup Factor]: GPU is {speedup:.2f}x FASTER than CPU.")

print("Now Running Kangaroo12 Benchmarks!")

test_batch_k12 = [(bytes(msg), b"") for msg in test_batch]
k12_cpu = Kangaroo12(enable_gpu=False)
k12_gpu = Kangaroo12(enable_gpu=True)

print(f"Starting CPU batched Merkle Tree Test")
start_cpu = time.perf_counter()
cpu_tree = MerkleTree(test_batch, hash=k12_cpu.hash_batch, enable_gpu=False, is_batched=True)
cpu_result = cpu_tree.root
cpu_time = time.perf_counter() - start_cpu
print (f"CPU Batch Completed in: {cpu_time} seconds with hash {cpu_result}")

print(f"Starting GPU batched Merkle Tree Test")
start_gpu = time.perf_counter()
gpu_tree = MerkleTree(test_batch, hash=k12_gpu.hash_batch, enable_gpu=True, is_batched=True)
gpu_result = gpu_tree.root
gpu_time = time.perf_counter() - start_gpu
print (f"GPU Batch Completed in: {gpu_time} seconds with hash {gpu_result}")

if cpu_result == gpu_result:
    print("[PASS] Cryptographic Equivalence Maintained across all hashes!")
    speedup = cpu_time / gpu_time
    print(f"[Speedup Factor]: GPU is {speedup:.2f}x FASTER than CPU.")
