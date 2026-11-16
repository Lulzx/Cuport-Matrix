# device-services/assert-001

One kernel with a device `assert(i != bad_index)`, launched twice. The first
launch passes an index no thread has, completes, and its output is checked
bit-exactly. The second launch passes an index drawn from the LCG so exactly
one of 64 threads trips; `cudaDeviceSynchronize` must return
`cudaErrorAssert` (710).

`tripped` separates "reported some error" from "reported no error at all":
a tool that turns `assert` into a plain trap without the 710 code mismatches
only `codes`, while a tool that compiles it away mismatches both. The assert
message goes to stderr and is not compared. Nothing after the second
synchronise is error-checked, because a device assert leaves the CUDA
context unusable by specification.
