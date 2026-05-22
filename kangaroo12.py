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

    state = bytearray(state)

    # state
    lane = [[0 for i in range(5)] for j in range(5)]
    for i in range(5):
        for j in range(5):
            offset = 8 * (i + 5 * j)
            lane[i][j] = int.from_bytes(state[offset : offset + 8], 'little')

    for rounds in range(12):

        # theta
        c = [0] * 5
        for i in range(5):
            c[i] = lane[i][0] ^ lane[i][1] ^ lane[i][2] ^ lane[i][3] ^ lane[i][4]
        
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
            shift_amount = (t + 1) * (t + 2) // 2
            lane[x][y], current = rotate(current, shift_amount), lane[x][y]


        # chi
        for j in range(5):
            T = [0]*5
            for i in range(5):
                T[i] = lane[i][j]
            for i in range(5):
                lane[i][j] = T[i] ^ ((~T[(i + 1) % 5] & 0xFFFFFFFFFFFFFFFF) & T[(i + 2) % 5])

        # iota
        lane[0][0] = lane[0][0] ^ RC[rounds]

    out_state = bytearray()
    for j in range(5):
        for i in range(5):
            out_state.extend(lane[i][j].to_bytes(8, 'little'))

    return bytes(out_state)

def TurboSHAKE128(message, separationByte, outputByteLen):
    offset = 0
    state = [0x00] * 200
    input  = list(message) + [separationByte]

    # Absorb
    while offset < len(input) - 168:
        state ^= input[offset : offset + 168] + [0x00] * 32
        state = keccak_p_1600(state)
        offset += 168
    
    lastBlock = len(input) - offset
    for i in range(lastBlock):
        state[i] ^= input[offset + i]
        
    state[168 - 1] ^= 0x80
    state = bytearray(keccak_p_1600(state))

    # Squeeze
    output = bytearray()
    while outputByteLen > 168:
        output = output + state[0 : 168]
        outputByteLen -= 168
        state = keccak_p_1600(state)

    output = output + state[0 : outputByteLen]

    return output

def TurboSHAKE256(message, separationByte, outputByteLen):
    offset = 0
    state = [0x00] * 200
    input  = list(message) + [separationByte]

    # Absorb
    while offset < len(input) - 136:
        state ^= input[offset : offset + 136] + [0x00] * 64
        state = keccak_p_1600(state)
        offset += 136
    
    lastBlock = len(input) - offset
    for i in range(lastBlock):
        state[i] ^= input[offset + i]
        
    state[136 - 1] ^= 0x80
    state = bytearray(keccak_p_1600(state))

    # Squeeze
    output = bytearray()
    while outputByteLen > 136:
        output = output + state[0 : 136]
        outputByteLen -= 136
        state = keccak_p_1600(state)

    output = output + state[0 : outputByteLen]

    return output

def length_encode(string):
    s = bytearray()
    while(string > 0):
        s = bytes([string % 256] + s)
        s = s//256
    s = s + bytes([len(s)])
    return s

def kangaroo12_128(inputMessage, customString, outputByteLen):
    s = inputMessage + customString
    s = s + length_encode(customString)

    if len(s) <= 8192:
        return TurboSHAKE128(s, 0x07, outputByteLen)
    else:
        fin = s[0:8192] + bytearray(0x03 + [0x00] * 7)
        offset = 8192
        numBlock = 0
        while offset < len(s):
            blockSize = min(len(s) - offset, 8192)
            CV = TurboSHAKE128(s[offset : offset + blockSize], 0x0b, 32)
            fin += CV
            numBlock += 1
            offset += blockSize
        fin = fin + length_encode(numBlock) + bytearray([0xff, 0xff])
        return TurboSHAKE128(fin, 0x06, outputByteLen)

def kangaroo12_256(inputMessage, customString, outputByteLen):
    s = inputMessage + customString
    s = s + length_encode(customString)

    if len(s) <= 8192:
        return TurboSHAKE256(s, 0x07, outputByteLen)
    else:
        fin = s[0:8192] + bytearray(0x03 + [0x00] * 7)
        offset = 8192
        numBlock = 0
        while offset < len(s):
            blockSize = min(len(s) - offset, 8192)
            CV = TurboSHAKE128(s[offset : offset + blockSize], 0x0b, 64)
            fin += CV
            numBlock += 1
            offset += blockSize
        fin = fin + length_encode(numBlock) + bytearray([0xff, 0xff])
        return TurboSHAKE128(fin, 0x06, outputByteLen)