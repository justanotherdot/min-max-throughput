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
min_max_multiple_passes time:   [794.29 ns 819.19 ns 846.59 ns]
min_max_conditional     time:   [237.54 ns 245.33 ns 253.06 ns]
min_max_bitwise_01      time:   [191.48 ns 193.62 ns 196.56 ns]
min_max_bitwise_02      time:   [1.3146 us 1.3155 us 1.3165 us]
min_max_simd_i32        time:   [698.01 ns 699.61 ns 701.57 ns]
```
