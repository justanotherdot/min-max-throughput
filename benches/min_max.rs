use criterion::{black_box, criterion_group, criterion_main, Criterion};
use min_max_throughput::*;
use std::mem::MaybeUninit;

pub fn criterion_benchmark(c: &mut Criterion) {
    let xs: Vec<_> = [MaybeUninit::<i64>::uninit(); 1000]
        .into_iter()
        .map(|mut uninit| {
            uninit.write(rand::random());
            unsafe { uninit.assume_init() }
        })
        .collect();
    let ys: Vec<_> = [MaybeUninit::<i32>::uninit(); 1000]
        .into_iter()
        .map(|mut uninit| {
            uninit.write(rand::random());
            unsafe { uninit.assume_init() }
        })
        .collect();

    c.bench_function("min_max_multiple_passes", |b| {
        b.iter(|| min_max_multiple_passes(black_box(&xs)))
    });
    c.bench_function("min_max_conditional", |b| {
        b.iter(|| min_max_conditional(black_box(&xs)))
    });
    c.bench_function("min_max_bitwise_01", |b| {
        b.iter(|| min_max_bitwise_01(black_box(&xs)))
    });
    c.bench_function("min_max_bitwise_02", |b| {
        b.iter(|| min_max_bitwise_02(black_box(&xs)))
    });
    c.bench_function("min_max_simd_i32_direct", |b| {
        b.iter(|| unsafe { min_max_simd_i32_direct(black_box(&ys)) })
    });
    c.bench_function("min_max_simd_i32_indirect", |b| {
        b.iter(|| unsafe { min_max_simd_i32_indirect(black_box(&ys)) })
    });
    c.bench_function("min_max_portable_simd", |b| {
        b.iter(|| unsafe { min_max_portable_simd(black_box(&ys)) })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
