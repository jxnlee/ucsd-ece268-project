import cupy as cp
import numpy as np

class MerkleTree:
    def __init__(self, data, hash, enable_gpu=False, is_batched=False):
        self.data = data
        self.hash = hash
        self.gpu = enable_gpu
        self.batched = is_batched

        self.xp = cp if self.gpu else np

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

        while len(current) > 1:
            # handle odd case by duplicating last node
            if len(current) % 2 != 0:
                current.append(current[-1])

            # gather left and right nodes
            left_nodes = current[0::2]
            right_nodes = current[1::2]

            # compute next level by hashing concatenated pairs
            next_level = self._hash_level([self._concatenate(left, right) for left, right in zip(left_nodes, right_nodes)])
            self.levels.append(next_level)

            # move up to the next level
            current = next_level

        if self.gpu:
            cp.cuda.Stream.null.synchronize()
        
        # return root hash
        return current[0]
    
    def get_proof(self, index):
        """
        Generates an audit path (Merkle Proof).
        Returns a list of tuples: (sibling_hash, is_left_sibling)
        """
        proof = []
        for level in self.levels[:-1]:  # Exclude root level
            is_right_child = (index % 2 == 1)
            
            if is_right_child:
                sibling_index = index - 1
                is_left_sibling = True
            else:
                # If an odd layer was padded, the sibling of the last element is itself
                sibling_index = index + 1 if (index + 1 < len(level)) else index
                is_left_sibling = False
                
            proof.append((level[sibling_index], is_left_sibling))
            index //= 2  # Move up to the parent's index position
            
        return proof
    
    def verify_proof(self, leaf, proof, expected_root):
        """
        Verifies that a leaf resolves to the expected root using the direction-aware proof.
        """
        computed_hash = self.hash(leaf)
        
        for sibling_hash, is_left_sibling in proof:
            if is_left_sibling:
                # Sibling belongs on the left
                computed_hash = self.hash(self._concatenate(sibling_hash, computed_hash))
            else:
                # Sibling belongs on the right
                computed_hash = self.hash(self._concatenate(computed_hash, sibling_hash))
                
        return computed_hash == expected_root