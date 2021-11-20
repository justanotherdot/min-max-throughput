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
min_max_multiple_passes   time:   [626.15 ns 627.87 ns 630.08 ns]
min_max_conditional       time:   [182.17 ns 182.56 ns 183.04 ns]
min_max_bitwise_01        time:   [182.14 ns 182.40 ns 182.70 ns]
min_max_bitwise_02        time:   [1.3124 us 1.3134 us 1.3150 us]
min_max_simd_i32_direct   time:   [1.3650 us 1.3706 us 1.3780 us]
min_max_simd_i32_indirect time:   [1.7274 us 1.7298 us 1.7325 us]
min_max_portable_simd     time:   [2.1274 us 2.1408 us 2.1595 us]
```
