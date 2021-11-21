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
pub fn pp(x: core::arch::x86_64::__m256i) -> core::arch::x86_64::__m256i {
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
    // TODO: try min_bitwise and max_bitwise below.
    for i in 0..8 {
        if max < *maxmax.get_unchecked(i) {
            max = *maxmax.get_unchecked(i);
        }
        if min > *minmin.get_unchecked(i) {
            min = *minmin.get_unchecked(i);
        }
    }
    Some((min, max))
}

/// NOTE: This implements a more traditional aproach, although with 256-bit lanes.
///
/// 1. Take a single register
/// 2. Shufflehi/shufflelo the elements rightwise
/// 3. Calculate the pairwise reduction (min/max/sum/product/etc.)
/// 4. Repeat at (1) until only a single element remains
///
/// Visualised:
///
/// (1.1) - initial
/// ┌──────────────────┐
/// │ 12 | 8 | -2 | 11 │ xmm0
/// └──────────────────┘
///
/// (2.1) - shuffle
/// ┌──────────────────┐
/// │ XX | XX | 12 | 8 │ xmm1
/// └──────────────────┘
///
/// (3.1) - max
/// ┌──────────────────┐
/// │ XX | X | 12 | 11 │ xmm2
/// └──────────────────┘
///
/// (2.2) - shuffle
/// ┌──────────────────┐
/// │ XX | X | XX | 12 │ xmm1
/// └──────────────────┘
///
/// (3.2) - max
/// ┌──────────────────┐
/// │ XX | X | XX | 12 │ xmm1
/// └──────────────────┘
///
/// The point is we can consider the buildup to the left as trash and take advantage of the
/// faster vertical instructions.
#[cfg(target_arch = "x86_64")]
pub unsafe fn min_max_simd_i32_indirect(buff: &[i32]) -> Option<(i32, i32)> {
    if buff.is_empty() {
        return None;
    }
    let mut min = i32::MAX;
    let mut max = i32::MIN;
    buff.chunks(8).into_iter().for_each(|slice| {
        if slice.len() == 8 {
            let x = store_to_mm_256i(
                *slice.get_unchecked(0),
                *slice.get_unchecked(1),
                *slice.get_unchecked(2),
                *slice.get_unchecked(3),
                *slice.get_unchecked(4),
                *slice.get_unchecked(5),
                *slice.get_unchecked(6),
                *slice.get_unchecked(7),
            );

            // TODO: try shift right instead of permutevar with control vector.
            //let shuffled = _mm256_shuffle_epi32::<0b00_00_00_11>(x);
            let indices = store_to_mm_256i(0, 0, 0, 0, 3, 2, 1, 0);
            let shuffled = _mm256_permutevar8x32_epi32(x, indices);
            let max1 = _mm256_max_epi32(shuffled, x);
            let indices = store_to_mm_256i(0, 0, 0, 0, 0, 0, 5, 4);
            let shuffled = _mm256_permutevar8x32_epi32(max1, indices);
            let max2 = _mm256_max_epi32(shuffled, max1);
            let indices = store_to_mm_256i(0, 0, 0, 0, 0, 0, 0, 6);
            let shuffled = _mm256_permutevar8x32_epi32(max2, indices);
            let max3 = _mm256_max_epi32(shuffled, max2);

            let indices = store_to_mm_256i(0, 0, 0, 0, 3, 2, 1, 0);
            let shuffled = _mm256_permutevar8x32_epi32(x, indices);
            let min1 = _mm256_min_epi32(shuffled, x);
            let indices = store_to_mm_256i(0, 0, 0, 0, 0, 0, 5, 4);
            let shuffled = _mm256_permutevar8x32_epi32(min1, indices);
            let min2 = _mm256_min_epi32(shuffled, min1);
            let indices = store_to_mm_256i(0, 0, 0, 0, 0, 0, 0, 6);
            let shuffled = _mm256_permutevar8x32_epi32(min2, indices);
            let min3 = _mm256_min_epi32(shuffled, min2);

            let local_max = _mm256_extract_epi32(max3, 7);
            let local_min = _mm256_extract_epi32(min3, 7);
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
