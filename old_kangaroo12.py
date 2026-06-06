import numpy as np

# Round constants for Keccak-p[1600, 12] (rounds 12-23 of the full Keccak-f[1600])
RC = np.array([
    0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
], dtype=np.uint64)

def rotate(a, n):
    n = n % 64
    return ((a >> (64 - n)) | (a << n)) & 0xFFFFFFFFFFFFFFFF

def keccak_p_1600(state):
    state = bytearray(state)

    lane = [[0]*5 for _ in range(5)]
    for x in range(5):
        for y in range(5):
            offset = 8 * (x + 5 * y)
            lane[x][y] = int.from_bytes(state[offset:offset+8], 'little')

    for r in range(12):
        # === THETA ===
        C = [lane[x][0]^lane[x][1]^lane[x][2]^lane[x][3]^lane[x][4] for x in range(5)]
        D = [C[(x-1)%5] ^ rotate(C[(x+1)%5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                lane[x][y] ^= D[x]

        # === RHO & PI (chained algorithm) ===
        x, y = 1, 0
        current = lane[x][y]
        for t in range(24):
            x, y = y, (2*x + 3*y) % 5
            shift_amount = (t + 1) * (t + 2) // 2
            lane[x][y], current = rotate(current, shift_amount), lane[x][y]

        # === CHI ===
        for y in range(5):
            T = [lane[x][y] for x in range(5)]
            for x in range(5):
                lane[x][y] = T[x] ^ ((~T[(x+1)%5] & 0xFFFFFFFFFFFFFFFF) & T[(x+2)%5])

        # === IOTA ===
        lane[0][0] ^= int(RC[r])

    out = bytearray(200)
    for x in range(5):
        for y in range(5):
            out[8*(x+5*y):8*(x+5*y)+8] = int(lane[x][y]).to_bytes(8, 'little')
    return bytes(out)

def TurboSHAKE256(message, separationByte, outputByteLen):
    RATE = 136
    offset = 0
    state = bytearray(200)
    inp = list(message) + [separationByte]

    # Absorb complete blocks
    while offset < len(inp) - RATE:
        for i in range(RATE):
            state[i] ^= inp[offset + i]
        state = bytearray(keccak_p_1600(state))
        offset += RATE

    # Absorb last (partial) block + padding
    lastBlockLen = len(inp) - offset
    for i in range(lastBlockLen):
        state[i] ^= inp[offset + i]
    state[RATE - 1] ^= 0x80
    state = bytearray(keccak_p_1600(state))

    # Squeeze
    output = bytearray()
    while outputByteLen > RATE:
        output.extend(state[0:RATE])
        outputByteLen -= RATE
        state = bytearray(keccak_p_1600(state))
    output.extend(state[0:outputByteLen])
    return bytes(output)

def length_encode(val):
    if val == 0:
        return b'\x00'
    s = bytearray()
    while val > 0:
        s.insert(0, val % 256)
        val //= 256
    s.append(len(s))
    return bytes(s)

def kangaroo12_256(inputMessage, customString, outputByteLen):
    s = bytes(inputMessage) + bytes(customString) + length_encode(len(customString))

    if len(s) <= 8192:
        return TurboSHAKE256(s, 0x07, outputByteLen)
    else:
        fin = bytearray(s[0:8192]) + bytearray([0x03] + [0x00] * 7)
        offset = 8192
        numBlock = 0
        while offset < len(s):
            blockSize = min(len(s) - offset, 8192)
            CV = TurboSHAKE256(s[offset:offset + blockSize], 0x0b, 32)
            fin.extend(CV)
            numBlock += 1
            offset += blockSize
        fin.extend(length_encode(numBlock))
        fin.extend([0xff, 0xff])
        return TurboSHAKE256(fin, 0x06, outputByteLen)

def run_tests():
    print("--- Executing Primitives Test Suite ---\n")
    msg_empty = b""

    # Test 1: TurboSHAKE128
    # Verified against XKCP reference: https://github.com/XKCP/K12/blob/master/Python/TurboSHAKE.py
    ts128_expected = "1e415f1c5983aff2169217277d17bb538cd945a397ddec541f1ce41af2c1b74c"
    ts128_actual = TurboSHAKE128(msg_empty, 0x1f, 32).hex()
    print(f"[*] TurboSHAKE128 (Empty String, 32 Bytes, Domain=0x1F):")
    print(f"    Expected: {ts128_expected}")
    print(f"    Actual:   {ts128_actual}")
    assert ts128_actual == ts128_expected, "TurboSHAKE128 test failed!"
    print("    [+] PASS\n")

    # Test 2: TurboSHAKE256
    # Verified against XKCP reference: https://github.com/XKCP/K12/blob/master/Python/TurboSHAKE.py
    ts256_expected = "367a329dafea871c7802ec67f905ae13c57695dc2c6663c61035f59a18f8e7db"
    ts256_actual = TurboSHAKE256(msg_empty, 0x1f, 32).hex()
    print(f"[*] TurboSHAKE256 (Empty String, 32 Bytes, Domain=0x1F):")
    print(f"    Expected: {ts256_expected}")
    print(f"    Actual:   {ts256_actual}")
    assert ts256_actual == ts256_expected, "TurboSHAKE256 test failed!"
    print("    [+] PASS\n")

    # Test 3: KangarooTwelve-128 (Short Message)
    # Verified against XKCP reference: https://github.com/XKCP/K12/blob/master/Python/KangarooTwelve.py
    k12_short_expected = "1ac2d450fc3b4205d19da7bfca1b37513c0803577ac7167f06fe2ce1f0ef39e5"
    k12_short_actual = kangaroo12_256(b"", b"", 32).hex()
    print(f"[*] KangarooTwelve-256 (Empty Message, Empty CustomString, 32 Bytes):")
    print(f"    Expected: {k12_short_expected}")
    print(f"    Actual:   {k12_short_actual}")
    assert k12_short_actual == k12_short_expected, "KangarooTwelve-128 Short test failed!"
    print("    [+] PASS\n")

    print("==========================================")
    print(" SUCCESS: All cryptographic tests passed!")
    print("==========================================")

if __name__ == "__main__":
    run_tests()