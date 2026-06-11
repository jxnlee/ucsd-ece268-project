import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from kangaroo12 import Kangaroo12


class CpuHashBackend:

    def __init__(self, fn):
        self.fn = fn

    def hash_batch(self, blocks: list[bytes]) -> list[bytes]:
        return [self.fn(b) for b in blocks]


class K12CpuBackend:

    def __init__(self):
        self._k12 = Kangaroo12(enable_gpu=False)

    def hash_batch(self, blocks: list[bytes]) -> list[bytes]:
        return [self._k12.hash(b, b"", 32) for b in blocks]


class K12GpuBackend:

    def __init__(self):
        if cp is None:
            raise ImportError("CuPy is required for K12GpuBackend.")
        self._k12 = Kangaroo12(enable_gpu=True)

    def hash_batch(self, blocks: list[bytes]) -> list[bytes]:
        results = [self._k12.hash(b, b"", 32) for b in blocks]
        cp.cuda.Stream.null.synchronize()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# MerkleTree
# ─────────────────────────────────────────────────────────────────────────────

class MerkleTree:
    
    def __init__(self, data: list[bytes], backend=None,
                 hash=None, enable_gpu: bool = False, is_batched: bool = False):

        self.data = data

        if backend is not None:
            self.backend = backend
        elif enable_gpu:
            self.backend = K12GpuBackend()
        elif hash is not None:
            self.backend = CpuHashBackend(hash)
        else:
            self.backend = K12CpuBackend()

        # kept for callers that inspect these attributes
        self.gpu     = enable_gpu or isinstance(self.backend, K12GpuBackend)
        self.batched = is_batched
        self.xp      = cp if self.gpu else np

        self.levels = [self._hash_level(data)]
        self.root = self._build_tree()

    def _hash_level(self, level_data):
        if self.batched:
            return self.hash(level_data)
        else:
            return [self.hash(block) for block in level_data]
        
    def _concatenate(self, left, right):
        return left + right
        
    def _build_tree(self):
        current = self.levels[0]

    def _build(self) -> bytes:
        current = self.backend.hash_batch(self.data)
        self.levels.append(current)

        while len(current) > 1:
            if len(current) % 2:
                current = current + [current[-1]]   # duplicate last node if odd

            pairs   = [current[i] + current[i + 1]
                       for i in range(0, len(current), 2)]
            current = self.backend.hash_batch(pairs)
            self.levels.append(current)

            # move up to the next level
            current = next_level

        if self.gpu:
            cp.cuda.Stream.null.synchronize()
        
        # return root hash
        return current[0]


    def get_proof(self, index: int) -> list[tuple[bytes, bool]]:
        proof = []
        for level in self.levels[:-1]:
            is_right = (index % 2 == 1)
            if is_right:
                sib_idx         = index - 1
                is_left_sibling = True
            else:
                sib_idx         = index + 1 if index + 1 < len(level) else index
                is_left_sibling = False
            proof.append((level[sib_idx], is_left_sibling))
            index //= 2
        return proof


    def verify_proof(self, leaf: bytes, proof: list[tuple[bytes, bool]],
                     expected_root: bytes) -> bool:

        cur = self.backend.hash_batch([leaf])[0]
        for sib, is_left in proof:
            pair = (sib + cur) if is_left else (cur + sib)
            cur  = self.backend.hash_batch([pair])[0]
        return cur == expected_root


    def _h(self, data: bytes) -> bytes:
        return self.backend.hash_batch([data])[0]

    def _concatenate(self, left, right):
        return left + right

    def _hash_level(self, level_data):
        return self.backend.hash_batch(list(level_data))

    def _build_tree(self):
        current = self.levels[0]
        while len(current) > 1:
            if len(current) % 2:
                current.append(current[-1])
            pairs   = [current[i] + current[i + 1] for i in range(0, len(current), 2)]
            current = self._hash_level(pairs)
            self.levels.append(current)
        return current[0]