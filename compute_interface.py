import numpy as np
import cupy as cp

class XPyComputeInterface:
    def __init__(self, use_gpu=False):
        self.config(use_gpu)

    def config(self, use_gpu):
        self.use_gpu = use_gpu
        self.xp = cp if use_gpu else np
        self.dtype = cp.uint64 if use_gpu else np.uint64

    def array(self, data, dtype=None):
        return self.xp.array(data, dtype=self.dtype if dtype is None else dtype)
    
    def zeros(self, shape, dtype=None):
        return self.xp.zeros(shape, dtype=self.dtype if dtype is None else dtype)
    def ones(self, shape, dtype=None):
        return self.xp.ones(shape, dtype=self.dtype if dtype is None else dtype)
    def concatenate(self, arrays):
        return self.xp.concatenate(arrays)

    def mod(self, array, modulus):
        return self.xp.mod(array, modulus)

    def matmul(self, a, b):
        return self.xp.matmul(a, b)

    def pow(self, base, exp, mod=None):
        if mod is not None:
            return self.xp.power(base, exp) % mod
        else:
            return self.xp.power(base, exp)