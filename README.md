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
min_max_multiple_passes    time:   [620.82 ns 628.99 ns 638.34 ns]
min_max_conditional        time:   [183.14 ns 184.40 ns 186.02 ns]
min_max_bitwise_01         time:   [182.30 ns 182.84 ns 183.59 ns]
min_max_bitwise_02         time:   [1.3158 us 1.3180 us 1.3208 us]
min_max_simd_i32_direct    time:   [161.92 ns 162.14 ns 162.37 ns]
min_max_simd_i32_indirect  time:   [418.78 ns 421.09 ns 424.25 ns]
min_max_portable_simd      time:   [2.0696 us 2.0820 us 2.0965 us]
```
