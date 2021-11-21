#![feature(portable_simd)]
#![feature(stdarch)]

use core::arch::x86_64::*;

pub fn min_max_multiple_passes(integers: &[i64]) -> Option<(i64, i64)> {
    integers
        .iter()
        .min()
        .and_then(|min| integers.iter().max().map(|max| (*min, *max)))
}

pub fn min_max_conditional(integers: &[i64]) -> Option<(i64, i64)> {
    integers.into_iter().fold(None, |acc, x| match acc {
        None => Some((*x, *x)),
        Some((min, max)) => Some((
            if *x < min { *x } else { min },
            if *x > max { *x } else { max },
        )),
    })
}

#[inline(always)]
pub fn min_bitwise(x: i64, y: i64) -> i64 {
    y ^ ((x ^ y) & -((x < y) as i64))
}

#[inline(always)]
pub fn max_bitwise(x: i64, y: i64) -> i64 {
    x ^ ((x ^ y) & -((x < y) as i64))
}

pub fn min_max_bitwise_01(integers: &[i64]) -> Option<(i64, i64)> {
    integers.into_iter().fold(None, |acc, x| match acc {
        None => Some((*x, *x)),
        Some((min, max)) => Some((min_bitwise(*x, min), max_bitwise(*x, max))),
    })
}

pub fn min_max_bitwise_02(integers: &[i64]) -> Option<(i64, i64)> {
    integers.into_iter().fold(None, |acc, x| match acc {
        None => Some((*x, *x)),
        Some((min, max)) => Some((
            min + ((*x - min)
                & ((*x - min) >> (std::mem::size_of::<i64>() * std::mem::size_of::<u8>() - 1))),
            *x - ((*x - max)
                & ((*x - max) >> (std::mem::size_of::<i64>() * std::mem::size_of::<u8>() - 1))),
        )),
    })
}

// NOTE: This isn't really in a benchable state.
// For it to get there, it needs to:
//
// * pass tests
// * be an implementation that is not obviously problematic
#[cfg(target_arch = "x86_64")]
pub unsafe fn min_max_portable_simd(buff: &[i32]) -> Option<(i32, i32)> {
    if buff.is_empty() {
        return None;
    }
    use std::simd::*;
    let mut max = i32::MIN;
    let lanes: Vec<_> = buff
        .chunks(4)
        .filter_map(|slice| {
            if slice.len() == 4 {
                Some(i32x4::from_array([
                    *slice.get(0).unwrap_or(&0),
                    *slice.get(0 + 1).unwrap_or(&0),
                    *slice.get(0 + 2).unwrap_or(&0),
                    *slice.get(0 + 3).unwrap_or(&0),
                ]))
            } else {
                max = *slice.iter().max()?;
                None
            }
        })
        .collect();
    if buff.len() % 4 == 0 {
        let mut last_lane = lanes[0];
        for lane in lanes {
            let part = &(lane.lanes_lt(last_lane)).to_array();
            let part = i32x4::from_array([
                part[0] as i32,
                part[1] as i32,
                part[2] as i32,
                part[3] as i32,
            ]);
            last_lane = lane ^ ((lane ^ last_lane) & -(part));
            //last_lane = lane ^ ((lane ^ last_lane) & -(lane < last_lane));
        }
        // NOTE: Horizontal operaitons are slow in general.
        //
        // The proper way to do a reduction is a vertical, pairwise max on each invocation. This
        // can be done on the same line, reducing with trash to one side, like the `indirect`
        // approach in this module, or it can be direct, except in portable_simd I do not see a way
        // to do pairwise max, only a way to ask if one lane is larger than the other, which I am
        // not completely sure means what I expect, which is that the maximum value across all
        // values in each lane is present in the lane returned.
        max = last_lane.horizontal_max();
    }
    Some((0, max))
}

#[cfg(target_arch = "x86_64")]
pub fn pp_128(x: core::arch::x86_64::__m128i) -> core::arch::x86_64::__m128i {
    unsafe {
        dbg!(std::mem::transmute::<_, [i32; 4]>(x));
    }
    x
}

#[cfg(target_arch = "x86_64")]
pub fn pp_256(x: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
    unsafe {
        dbg!(std::mem::transmute::<_, [i32; 8]>(x));
    }
    x
}

#[cfg(target_arch = "x86_64")]
unsafe fn store_to_mm_256i(
    x0: i32,
    x1: i32,
    x2: i32,
    x3: i32,
    x4: i32,
    x5: i32,
    x6: i32,
    x7: i32,
) -> __m256i {
    _mm256_setr_epi32(x0, x1, x2, x3, x4, x5, x6, x7)
}

#[cfg(target_arch = "x86_64")]
unsafe fn splat(x0: i32) -> __m256i {
    _mm256_set_epi32(x0, x0, x0, x0, x0, x0, x0, x0)
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn min_max_simd_i32_direct(buff: &[i32]) -> Option<(i32, i32)> {
    if buff.is_empty() {
        return None;
    }
    let mut maxval = splat(i32::MIN);
    let mut minval = splat(i32::MAX);
    let mut max = i32::MIN;
    let mut min = i32::MAX;
    buff.chunks(8).into_iter().for_each(|slice| {
        if slice.len() == 8 {
            let chunk = store_to_mm_256i(
                *slice.get_unchecked(0),
                *slice.get_unchecked(1),
                *slice.get_unchecked(2),
                *slice.get_unchecked(3),
                *slice.get_unchecked(4),
                *slice.get_unchecked(5),
                *slice.get_unchecked(6),
                *slice.get_unchecked(7),
            );
            maxval = _mm256_max_epi32(maxval, chunk);
            minval = _mm256_min_epi32(minval, chunk);
        } else {
            max = *slice.iter().max().unwrap();
            min = *slice.iter().min().unwrap();
        }
    });
    let maxmax: [i32; 8] = std::mem::transmute(maxval);
    let minmin: [i32; 8] = std::mem::transmute(minval);
    let local_max = *maxmax.iter().max().unwrap();
    let local_min = *minmin.iter().min().unwrap();
    if max < local_max {
        max = local_max
    }
    if min > local_min {
        min = local_min
    }
    Some((min, max))
}

/// NOTE: This implements a more traditional approach.
///
/// 1. Take a single register
/// 2. Shuffle the elements into place
/// 3. Calculate the pairwise reduction (min/max/sum/product/etc.)
/// 4. Repeat at (1) until only a single element remains
///
/// Visualised:
///
/// ```text
/// (1.1) - initial
/// ┌───────────────────┐
/// │ 12 |  8 | -2 | 11 │ xmm0
/// └───────────────────┘
///
/// (2.1) - shuffle
/// ┌───────────────────┐
/// │ -2 | 11 | XX | XX │ xmm1
/// └───────────────────┘
///
/// (3.1) - max
/// ┌───────────────────┐
/// │ 12 | 11 | XX | XX │ xmm2
/// └───────────────────┘
///
/// (2.2) - shuffle
/// ┌───────────────────┐
/// │ 11 | XX | XX | XX │ xmm1
/// └───────────────────┘
///
/// (3.2) - max
/// ┌───────────────────┐
/// │ 12 | XX | XX | XX │ xmm1
/// └───────────────────┘
/// ```
///
/// The point is we can consider the buildup to the left as trash and take advantage of the
/// faster vertical instructions, and then extract out the first element of the vector when
/// we are done.
#[cfg(target_arch = "x86_64")]
pub unsafe fn min_max_simd_i32_indirect(buff: &[i32]) -> Option<(i32, i32)> {
    if buff.is_empty() {
        return None;
    }
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    buff.chunks(4).into_iter().for_each(|slice| {
        if slice.len() == 4 {
            let x = _mm_set_epi32(
                *slice.get_unchecked(0),
                *slice.get_unchecked(1),
                *slice.get_unchecked(2),
                *slice.get_unchecked(3),
            );

            let shuffled = _mm_shuffle_epi32::<{ _MM_SHUFFLE(1, 0, 3, 2) }>(x);
            // NOTE: with avx2 support, purportedly better perf.
            //let shuffled = _mm_unpackhi_epi64(x, x);
            let max1 = _mm_max_epi32(shuffled, x);
            let shuffled = _mm_shufflelo_epi16::<{ _MM_SHUFFLE(1, 0, 3, 2) }>(max1);
            let max2 = _mm_max_epi32(shuffled, max1);
            let local_max = _mm_cvtsi128_si32(max2);

            let shuffled = _mm_shuffle_epi32::<{ _MM_SHUFFLE(1, 0, 3, 2) }>(x);
            // NOTE: with avx2 support, purportedly better perf.
            //let shuffled = _mm_unpackhi_epi64(x, x);
            let min1 = _mm_min_epi32(shuffled, x);
            let shuffled = _mm_shufflelo_epi16::<{ _MM_SHUFFLE(1, 0, 3, 2) }>(min1);
            let min2 = _mm_min_epi32(shuffled, min1);
            let local_min = _mm_cvtsi128_si32(min2);

            if local_max > max {
                max = local_max;
            }
            if local_min < min {
                min = local_min;
            }
        } else {
            let local_max = *slice.iter().max().unwrap();
            let local_min = *slice.iter().min().unwrap();
            if local_max > max {
                max = local_max;
            }
            if local_min < min {
                min = local_min;
            }
        }
    });
    Some((min, max))
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn min_max_simd_i32(buff: &[i32]) -> Option<(i32, i32)> {
    min_max_simd_i32_direct(buff)
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::prelude::*;
    use rand::{thread_rng, Rng};

    fn vector() -> [i64; 10] {
        let mut rng = thread_rng();
        let mut result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        result.shuffle(&mut rng);
        result
    }

    fn array_i32(size: usize) -> Vec<i32> {
        let mut rng = thread_rng();
        let mut result: Vec<_> = std::iter::repeat(0)
            .take(size)
            .map(|_| rng.gen_range(i32::MIN..i32::MAX))
            .collect();
        result.shuffle(&mut rng);
        result
    }

    #[test]
    fn min_max_multiple_passes_works() {
        assert_eq!(min_max_multiple_passes(&[]), None);
        assert_eq!(min_max_multiple_passes(&vector()), Some((1, 10)));
    }

    #[test]
    fn min_max_conditional_works() {
        assert_eq!(min_max_conditional(&[]), None);
        assert_eq!(min_max_conditional(&vector()), Some((1, 10)));
    }

    #[test]
    fn min_max_bitwise_works_01() {
        assert_eq!(min_max_bitwise_01(&[]), None);
        assert_eq!(min_max_bitwise_01(&vector()), Some((1, 10)));
    }

    #[test]
    fn min_max_bitwise_02_works() {
        assert_eq!(min_max_bitwise_02(&[]), None);
        assert_eq!(min_max_bitwise_02(&vector()), Some((1, 10)));
    }

    #[test]
    fn min_max_simd_i32_direct_work_with_exact_sized_arrays() {
        unsafe {
            assert_eq!(min_max_simd_i32(&array_i32(0)), None);
            for x in 1..3 {
                let array = array_i32(x * 8);
                let min = *array.iter().min().unwrap();
                let max = *array.iter().max().unwrap();
                assert_eq!(min_max_simd_i32_direct(&array), Some((min, max)));
            }
        }
    }

    #[test]
    fn min_max_simd_i32_direct_work_with_inexact_sized_arrays() {
        unsafe {
            for x in 9..11 {
                let array = array_i32(x);
                let min = *array.iter().min().unwrap();
                let max = *array.iter().max().unwrap();
                assert_eq!(min_max_simd_i32_direct(&array), Some((min, max)));
            }
        }
    }

    #[test]
    fn min_max_simd_i32_direct_works() {
        for size in 1..100 + 1 {
            unsafe {
                let array = array_i32(size);
                let min = *array.iter().min().unwrap();
                let max = *array.iter().max().unwrap();
                assert_eq!(min_max_simd_i32_direct(&array), Some((min, max)));
            }
        }
    }

    #[test]
    fn min_max_simd_i32_indirect_work_with_exact_sized_arrays() {
        unsafe {
            assert_eq!(min_max_simd_i32(&array_i32(0)), None);
            for x in 1..3 {
                let array = array_i32(x * 8);
                let min = *array.iter().min().unwrap();
                let max = *array.iter().max().unwrap();
                assert_eq!(min_max_simd_i32_indirect(&array), Some((min, max)));
            }
        }
    }

    #[test]
    fn min_max_simd_i32_indirect_work_with_inexact_sized_arrays() {
        unsafe {
            for x in 9..11 {
                let array = array_i32(x);
                let min = *array.iter().min().unwrap();
                let max = *array.iter().max().unwrap();
                assert_eq!(min_max_simd_i32_indirect(&array), Some((min, max)));
            }
        }
    }

    #[test]
    fn min_max_simd_i32_indirect_works() {
        for size in 1..100 + 1 {
            unsafe {
                let array = array_i32(size);
                let min = *array.iter().min().unwrap();
                let max = *array.iter().max().unwrap();
                assert_eq!(min_max_simd_i32_indirect(&array), Some((min, max)));
            }
        }
    }

    //#[test]
    //fn min_max_portable_simd_work_with_exact_sized_arrays() {
    //    unsafe {
    //        for x in 1..3 {
    //            let array = array_i32(x * 8);
    //            //let min = *array.iter().min().unwrap();
    //            let max = *array.iter().max().unwrap();
    //            // NOTE: This function is currently only returning 0 for min.
    //            assert_eq!(min_max_portable_simd(&array), Some((0, max)));
    //        }
    //    }
    //}

    //#[test]
    //fn min_max_portable_simd_work_with_inexact_sized_arrays() {
    //    unsafe {
    //        for x in 9..11 {
    //            let array = array_i32(x);
    //            //let min = *array.iter().min().unwrap();
    //            let max = *array.iter().max().unwrap();
    //            // NOTE: This function is currently only returning 0 for min.
    //            assert_eq!(min_max_portable_simd(&array), Some((0, max)));
    //        }
    //    }
    //}

    //#[test]
    //fn min_max_portable_simd_works() {
    //    for size in 1..100 + 1 {
    //        unsafe {
    //            let array = array_i32(size);
    //            //let min = *array.iter().min().unwrap();
    //            let max = *array.iter().max().unwrap();
    //            // NOTE: This function is currently only returning 0 for min.
    //            assert_eq!(min_max_portable_simd(&array), Some((0, max)));
    //        }
    //    }
    //}
}
