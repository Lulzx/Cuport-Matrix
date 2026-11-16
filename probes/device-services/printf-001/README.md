# device-services/printf-001

Sixty-four threads (4 blocks x 16) each `printf` one line that exercises
`%u`, `%d`, `%08x`, `%.3f`, `%lld`, `%s` and a literal `%%`. The oracle
builds the same 64 lines in Python and the runner compares them as a
multiset, because CUDA does not order output across threads.

The float is `k/8` with `k < 16384`, which has at most three decimals, so
`%.3f` has no rounding ties and the comparison does not depend on the host
C library. The `lines` buffer only shows the probe reached the end; the
finding is in the text lines. A tool that drops lines, reorders fields,
formats `%lld` as 32-bit or truncates output on a small printf FIFO fails
here.
