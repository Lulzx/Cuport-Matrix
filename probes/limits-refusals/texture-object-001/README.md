# limits-refusals/texture-object-001

Creates a `cudaTextureObject_t` over a 256-word linear device buffer and
gathers from it with `tex1Dfetch<uint32_t>` using a stride-7 permutation.
The oracle is the same permutation applied on the host.

Texture *references* were removed in CUDA 12, so the probe uses the object
API, which is the only texture path a current tool has to support. A tool
that does not implement textures should fail at compile time (missing
types) or at `cudaCreateTextureObject` (`CUPORT-ERROR`), both
`refused-with-diagnostic`. A tool that maps textures to a Metal texture
with a wrong channel format, or applies filtering to an element-type read,
produces wrong words and lands in `silently-wrong`.
