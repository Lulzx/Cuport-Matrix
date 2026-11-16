# atomics/min-max-unsigned-001

The same 4096 bit patterns go through `atomicMin`/`atomicMax` twice, once
as `unsigned` and once as `int`. Half the inputs have the top bit set, so the
two orders disagree: the unsigned maximum is a "negative" number to the
signed overload. A translation that uses one comparison for both overloads
swaps minima and maxima in one of the two halves of the output.
