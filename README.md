# ucsd-ece268-project

Merkle Tree with Post-Quantum Secure Hash Project

Built and evaluated a [Merkle Tree](https://en.wikipedia.org/wiki/Merkle_tree) using hash functions that are post-quantum secure ([Kangaroo12](<(https://keccak.team/kangarootwelve.html)>) and [Rescue-Prime](https://www.esat.kuleuven.be/cosic/sites/rescue/)).

Hash functions were implemented end-to-end (with no pre-existing black-box hashing in the implementation of the specifications of the hash function)

Implemented Merkle Tree generation (from leaves to root) along with proof generation and verification.

Hash function and Merkle Tree code can be found in `hash_src/`, tests and experiments can be found in `tests/`, and benchmarking/analysis/demos can be found in `analysis/`.
