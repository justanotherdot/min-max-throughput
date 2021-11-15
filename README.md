# Min/Max Throughput

## Benchmarks

### Environment

Environment is an Intel Core(TM) i7-7500U, performance governor, Linux
5.10.0-7-amd64.

1. Set performance governor.

```
$ cpupower frequency-set --governor performance
```

2. Run the benchmarks

```
$ cargo criterion
```

### Results

```
min_max_multiple_passes time:   [60.331 ns 60.810 ns 61.387 ns]
min_max_conditional     time:   [29.273 ns 29.846 ns 30.468 ns]
min_max_bitwise_01      time:   [29.844 ns 31.097 ns 32.394 ns]
min_max_bitwise_02      time:   [129.16 ns 129.43 ns 129.80 ns]
min_max_simd_i16        time:   [86.872 ns 87.740 ns 88.851 ns]
```
