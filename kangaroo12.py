# IMPORTS
import numpy as np
import cupy as cp

# TURBOSHAKE

# Round constants for Keccak-p[1600, 12]
RC = [0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
      0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
      0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]

def rotate(a, n):
    return ((a >> (64 - (n % 64))) + (a << (n % 64))) % (1 << 64)

def keccak_p_1600(state):

    # state
    lane = [[0 for i in range(5)] for j in range(5)]
    for i in range(5):
        for j in range(5):
            lane[i][j] = state[64 * (5 * j + i) : 64 * (5 * j + i) + 64]

    for rounds in range(12):

        # theta
        c = [0] * 5
        for i in range(5):
            c[i] = lane[i][0]
            c[i] ^= lane[i][1]
            c[i] ^= lane[i][2]
            c[i] ^= lane[i][3]
            c[i] ^= lane[i][4]
        
        d = [0] * 5
        for i in range(5):
            d[i] = c[(i - 1) % 5] ^ rotate(c[(i + 1) % 5], 1)

        for i in range(5):
            for j in range(5):
                lane[i][j] = lane[i][j] ^ d[i]

        # rho and pi
        (x, y) = (1, 0)
        current = lane[x][y]
        for t in range(24):
            (x, y) = (y, ((2 * x) + (3 * y)) % 5)
            (current, lane[x][y]) = (lane[x][y], rotate(current, (t + 1) * ( t + 2 ) // 2))


        # chi
        for j in range(5):
            T = [0]*5
            for i in range(5):
                T[i] = lane[i][j]
            for i in range(5):
                lane[i][j] = T[i] ^ ((~T[(i + 1) % 5]) & T[(i + 2) % 5])

        # iota
        lane[0][0] = lane[0][0] ^ RC[rounds]

    state = bytearray()
    for i in range(5):
        for j in range(5):
            state = state + lane[i][j]

    return state

def TurboSHAKE128():
    return None

def TurboSHAKE256():
    return None

def kangaroo12():
    return None