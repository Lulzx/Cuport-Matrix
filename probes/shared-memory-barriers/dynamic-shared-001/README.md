# shared-memory-barriers/dynamic-shared-001

One `extern __shared__` allocation of 2 x 192 words, carved into a `float`
array and a `uint32_t` array at offset `blockDim.x`. Each thread stores one
value in each, synchronises, and reads a neighbour's (offset 1 for floats,
7 for integers). Wrong carving shows up as integers in `fout` or floats in
`uout`. A block size of 192 rules out power-of-two shortcuts.
