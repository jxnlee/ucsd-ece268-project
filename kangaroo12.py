import numpy as np
import cupy as cp
import time
import random

class Kangaroo12:
    def __init__(self, custom_string=b"", output_byte_len=32, enable_gpu=False):
        """
        Kangaroo12 hash implementation supporting CPU (NumPy) and GPU (CuPy) backends.
        """
        if enable_gpu and cp is None:
            raise ImportError("CuPy is enabled but could not be imported.")
            
        self.xp = cp if enable_gpu else np
        self.enable_gpu = enable_gpu
        self.custom_string = bytes(custom_string)
        self.output_byte_len = output_byte_len

        # Round constants configured for the internal backend
        self.RC = self.xp.array([
            0x000000008000808b, 0x800000000000008b, 0x8000000000008089, 0x8000000000008003,
            0x8000000000008002, 0x8000000000000080, 0x000000000000800a, 0x800000008000000a,
            0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
        ], dtype=self.xp.uint64)

    def _rotate(self, a, n):
        n = n % 64
        return ((a >> (64 - n)) | (a << n)) & 0xFFFFFFFFFFFFFFFF

    def _keccak_p_1600_batch(self, state_lanes):
        """
        Vectorized Keccak-p[1600, 12] permutation.
        state_lanes shape: (Batch_Size, 5, 5) of uint64
        """
        lane = state_lanes.copy()

        for r in range(12):
            # === THETA ===
            C = lane[:, :, 0] ^ lane[:, :, 1] ^ lane[:, :, 2] ^ lane[:, :, 3] ^ lane[:, :, 4]
            
            # Roll axes across X dimension for vectorized cross-lane parity
            C_minus = self.xp.roll(C, 1, axis=1)
            C_plus = self.xp.roll(C, -1, axis=1)
            D = C_minus ^ self._rotate(C_plus, 1)
            
            # Apply D to lanes across the batch
            lane ^= D[:, :, np.newaxis]

            # === RHO & PI ===
            # Due to dependent steps, we can compute the shifts predictably
            x, y = 1, 0
            current = lane[:, x, y].copy()
            for t in range(24):
                next_x, next_y = y, (2 * x + 3 * y) % 5
                shift_amount = (t + 1) * (t + 2) // 2
                
                tmp = lane[:, next_x, next_y].copy()
                lane[:, next_x, next_y] = self._rotate(current, shift_amount)
                current = tmp
                x, y = next_x, next_y

            # === CHI ===
            # Compute step logic cleanly using vectorized bitwise masking
            T = lane.copy()
            for x in range(5):
                lane[:, x, :] = T[:, x, :] ^ ((~T[:, (x + 1) % 5, :]) & T[:, (x + 2) % 5, :])

            # === IOTA ===
            lane[:, 0, 0] ^= self.RC[r]

        return lane

    def _turboshake256_batch(self, messages_bytes, separation_byte, output_len):
        """
        Batched implementation of TurboSHAKE256 using vectorized lanes.
        Expects a list of equal-length byte segments.
        """
        RATE = 136
        batch_size = len(messages_bytes)
        
        # Determine uniform padded message length across this execution step
        msg_len = len(messages_bytes[0])
        padded_inp_len = msg_len + 1 # include separation byte
        
        # Pack everything efficiently into a flat bytes array
        flat_inp = bytearray()
        for msg in messages_bytes:
            flat_inp.extend(msg)
            flat_inp.append(separation_byte)
            
        # Convert flat data array over to CPU/GPU 8-bit space
        inp_array = self.xp.frombuffer(flat_inp, dtype=self.xp.uint8).reshape(batch_size, padded_inp_len)
        
        # State represents 200 bytes per message initialized to zero (25 uint64 lanes)
        state_lanes = self.xp.zeros((batch_size, 5, 5), dtype=self.xp.uint64)
        
        offset = 0
        while offset < padded_inp_len:
            block_len = min(RATE, padded_inp_len - offset)
            
            # Extract block from input slice
            chunk = self.xp.zeros((batch_size, 200), dtype=self.xp.uint8)
            chunk[:, :block_len] = inp_array[:, offset:offset+block_len]
            
            # If last block, apply the 0x80 padding constraint directly onto chunks
            if offset + block_len == padded_inp_len:
                chunk[:, RATE - 1] ^= 0x80
                
            # View 200 byte state chunks as uint64 values mapped to the 5x5 Keccak grid
            chunk_lanes = chunk.view(self.xp.uint64).reshape(batch_size, 5, 5)
            state_lanes ^= chunk_lanes
            
            # Run the core matrix permutation 
            state_lanes = self._keccak_p_1600_batch(state_lanes)
            offset += RATE

        # Squeeze Phase
        output_chunks = []
        bytes_squeezed = 0
        while bytes_squeezed < output_len:
            # Flatten lane configurations back into sequential uint8 rows
            flat_state_bytes = state_lanes.view(self.xp.uint8).reshape(batch_size, 200)
            take_len = min(RATE, output_len - bytes_squeezed)
            output_chunks.append(flat_state_bytes[:, :take_len])
            bytes_squeezed += take_len
            if bytes_squeezed < output_len:
                state_lanes = self._keccak_p_1600_batch(state_lanes)

        # Concatenate along axis-1 to stitch raw chunks back into unique array rows per message
        result_array = self.xp.concatenate(output_chunks, axis=1)
        
        # If processing on the GPU, return contents seamlessly to host memory space
        if self.enable_gpu:
            return [bytes(row) for row in result_array.get()]
        else:
            return [bytes(row) for row in result_array]

    def _length_encode(self, val):
        if val == 0:
            return b'\x00'
        s = bytearray()
        while val > 0:
            s.insert(0, val % 256)
            val //= 256
        s.append(len(s))
        return bytes(s)

    def hash(self, message):
        """
        Hashes a single message input sequence.
        """
        return self.hash_batch([message])[0]

    def hash_batch(self, *messages):
        """
        Accepts variable argument lists of message structures or a single iterable of messages.
        """
        # If the user passed a single list/tuple of items, unpack it
        if len(messages) == 1 and isinstance(messages[0], (list, tuple)):
            # Ensure it's a list of messages, not just a single message wrapped in a list
            # We check if the first element is an integer (which means it's a single message)
            if len(messages[0]) > 0 and isinstance(messages[0][0], int):
                messages = [messages[0]]
            else:
                messages = messages[0]
        else:
            messages = list(messages)
            
        processed_msgs = [bytes(m) + self.custom_string + self._length_encode(len(self.custom_string)) for m in messages]
        
        # Kangaroo12 Tree hashing structural sorting
        results = {}
        for idx, s in enumerate(processed_msgs):
            if len(s) <= 8192:
                # Fast Path
                results[idx] = ('fast', s)
            else:
                # Tree Path: build blocks structure
                fin = bytearray(s[0:8192]) + bytearray([0x03] + [0x00] * 7)
                offset = 8192
                tree_blocks = []
                while offset < len(s):
                    block_size = min(len(s) - offset, 8192)
                    tree_blocks.append(s[offset:offset + block_size])
                    offset += block_size
                results[idx] = ('tree', fin, tree_blocks)

        # 1. Execute Fast Path Jobs in Batch
        fast_indices = [i for i, v in results.items() if v[0] == 'fast']
        if fast_indices:
            # Group by similar lengths to keep vectorized matrices completely uniform
            lens = [len(results[i][1]) for i in fast_indices]
            for unique_len in set(lens):
                sub_indices = [i for i in fast_indices if len(results[i][1]) == unique_len]
                sub_msgs = [results[i][1] for i in sub_indices]
                hashes = self._turboshake256_batch(sub_msgs, 0x07, self.output_byte_len)
                for idx, h in zip(sub_indices, hashes):
                    results[idx] = h

        # 2. Execute Tree Path Intermediates in Batch
        tree_indices = [i for i, v in results.items() if v[0] == 'tree']
        if tree_indices:
            cv_tasks = []
            task_mapping = [] 
            for idx in tree_indices:
                for block in results[idx][2]:
                    cv_tasks.append(block)
                    task_mapping.append(idx)
                    
            if cv_tasks:
                cv_lens = [len(b) for b in cv_tasks]
                cv_hashes = [None] * len(cv_tasks)
                for unique_len in set(cv_lens):
                    sub_task_ids = [i for i, b in enumerate(cv_tasks) if len(b) == unique_len]
                    sub_blocks = [cv_tasks[i] for i in sub_task_ids]
                    hashes = self._turboshake256_batch(sub_blocks, 0x0b, 32)
                    for chunk_id, h in zip(sub_task_ids, hashes):
                        cv_hashes[chunk_id] = h
                
                msg_cv_collections = {idx: bytearray() for idx in tree_indices}
                msg_block_counts = {idx: 0 for idx in tree_indices}
                for chunk_id, h in enumerate(cv_hashes):
                    target_msg_idx = task_mapping[chunk_id]
                    msg_cv_collections[target_msg_idx].extend(h)
                    msg_block_counts[target_msg_idx] += 1
                
                final_roots = []
                for idx in tree_indices:
                    fin = results[idx][1]
                    fin.extend(msg_cv_collections[idx])
                    fin.extend(self._length_encode(msg_block_counts[idx]))
                    fin.extend([0xff, 0xff])
                    final_roots.append(bytes(fin))
                
                root_lens = [len(r) for r in final_roots]
                final_hashes = [None] * len(tree_indices)
                for unique_len in set(root_lens):
                    sub_root_ids = [i for i, r in enumerate(final_roots) if len(r) == unique_len]
                    sub_roots = [final_roots[i] for i in sub_root_ids]
                    hashes = self._turboshake256_batch(sub_roots, 0x06, self.output_byte_len)
                    for r_id, h in zip(sub_root_ids, hashes):
                        final_hashes[r_id] = h
                        
                for idx, h in zip(tree_indices, final_hashes):
                    results[idx] = h

        return [results[i] for i in range(len(messages))]
    
if __name__ == "__main__":
    # Initialize your backend configurations
    k_cpu = Kangaroo12(custom_string=b"Test", enable_gpu=False)
    k_gpu = Kangaroo12(custom_string=b"Test", enable_gpu=True)  # Requires an environment with CuPy/CUDA

    # Test 1: Single Message Hashing
    msg_single = [0, 255, 127, 64]
    hash_cpu = k_cpu.hash(msg_single)
    print(f"CPU Single Hash: {hash_cpu.hex()}")

    # Test 2: Batched Message Processing
    msg_batch = [
        [0, 255, 127, 32],
        [12, 43, 99, 101, 202],
        [0] * 9000 # Forces Tree mode execution block branch split
    ]

    hashes_cpu = k_cpu.hash_batch(msg_batch)
    print(f"Batch Processing Items Count: {len(hashes_cpu)}")
    for idx, h in enumerate(hashes_cpu):
        print(f" -> Msg {idx} Output: {h.hex()[:20]}...")