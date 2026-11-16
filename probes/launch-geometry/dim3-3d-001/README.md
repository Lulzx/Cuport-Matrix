# launch-geometry/dim3-3d-001

A `dim3(3, 2, 4)` grid of `dim3(8, 4, 2)` blocks. Each thread writes its six
coordinates at the linear index CUDA defines (x fastest), and thread 0 writes
`blockDim` and `gridDim`. Swapped axes, a flattened y or z, or a wrong
linearisation all land values at the wrong index.
