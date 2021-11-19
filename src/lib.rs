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

pub fn min_max_bitwise_01(integers: &[i64]) -> Option<(i64, i64)> {
    integers.into_iter().fold(None, |acc, x| match acc {
        None => Some((*x, *x)),
        Some((min, max)) => Some((
            min ^ ((*x ^ min) & -((*x < min) as i64)),
            *x ^ ((*x ^ max) & -((*x < max) as i64)),
        )),
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

#[cfg(target_arch = "x86_64")]
pub fn pp_sse2_register(x: core::arch::x86_64::__m128i) -> core::arch::x86_64::__m128i {
    unsafe {
        dbg!(std::mem::transmute::<_, [i16; 8]>(x));
    }
    x
}

#[cfg(target_arch = "x86_64")]
pub unsafe fn min_max_simd_i32(buff: &[i32]) -> Option<(i32, i32)> {
    // need to consider
    // * excess padding on non multiples of 8, e.g. 9 elements
    // * negative value
    // * upgrading to i32 buffer, rather than i32
    use core::arch::x86_64::*;
    unsafe fn store_to_mm_256i(
        x7: i32,
        x6: i32,
        x5: i32,
        x4: i32,
        x3: i32,
        x2: i32,
        x1: i32,
        x0: i32,
    ) -> __m256i {
        _mm256_set_epi32(x0, x1, x2, x3, x4, x5, x6, x7)
    }
    let mut maxmax = _mm256_setzero_si256();
    let mut minmin = _mm256_setzero_si256();
    let mut max = *buff.get(0)?;
    let mut min = *buff.get(0)?;
    let f8: &[__m256i] = &(0..(buff.len() / 8) + 1)
        .into_iter()
        .map(|ix| {
            store_to_mm_256i(
                *buff.get(ix * 8).unwrap_or(&0),
                *buff.get(ix * 8 + 1).unwrap_or(&0),
                *buff.get(ix * 8 + 2).unwrap_or(&0),
                *buff.get(ix * 8 + 3).unwrap_or(&0),
                *buff.get(ix * 8 + 4).unwrap_or(&0),
                *buff.get(ix * 8 + 5).unwrap_or(&0),
                *buff.get(ix * 8 + 6).unwrap_or(&0),
                *buff.get(ix * 8 + 7).unwrap_or(&0),
            )
        })
        .collect::<Vec<_>>();
    let mut maxval = _mm256_setzero_si256();
    let mut minval = _mm256_setzero_si256();
    for chunk in f8 {
        maxval = _mm256_max_epi32(maxval, *chunk);
        minval = _mm256_min_epi32(minval, *chunk);
    }
    // NOTE: if we go from i32 -> i64, we'll need to keep using __mm256_max_epi32.
    // in which the case the below might matter.
    // FIXME: I don't think the below actually does anything yet.
    //let mut maxval2 = maxval;
    //let mut minval2 = minval;
    //for _ in 0..3 {
    //    //pp(maxval);
    //    //pp(_mm_shufflehi_epi32::<3>(maxval));
    //    //pp(_mm_shufflehi_epi32::<8>(maxval));

    //    maxval = _mm_max_epi32(maxval, _mm_shufflehi_epi32::<3>(maxval));
    //    _mm_store_si256(&mut maxmax as *mut __m256i, maxval);
    //    maxval2 = _mm_max_epi32(maxval2, _mm_shufflelo_epi32::<3>(maxval2));
    //    _mm_store_si256(&mut maxmax as *mut __m256i, maxval2);

    //    minval = _mm_min_epi32(minval, _mm_shufflehi_epi32::<3>(minval));
    //    _mm_store_si256(&mut minmin as *mut __m256i, minval);
    //    minval2 = _mm_min_epi32(minval2, _mm_shufflelo_epi32::<3>(minval2));
    //    _mm_store_si256(&mut minmin as *mut __m256i, minval2);
    //}
    _mm256_store_si256(&mut maxmax as *mut __m256i, maxval);
    _mm256_store_si256(&mut minmin as *mut __m256i, minval);
    let maxmax: [i32; 8] = std::mem::transmute(maxmax);
    let minmin: [i32; 8] = std::mem::transmute(minmin);
    // get_unchecked
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
            .enumerate()
            .map(|(i, _)| i as i32 + 1)
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
    fn min_max_simd_i32_work_with_exact_sized_arrays() {
        unsafe {
            assert_eq!(min_max_simd_i32(&array_i32(0)), None);
            assert_eq!(min_max_simd_i32(&array_i32(8)), Some((0, 8)));
            assert_eq!(min_max_simd_i32(&array_i32(16)), Some((0, 16)));
        }
    }

    #[test]
    fn min_max_simd_i32_work_with_inexact_sized_arrays() {
        unsafe {
            assert_eq!(min_max_simd_i32(&array_i32(9)), Some((0, 9)));
            assert_eq!(min_max_simd_i32(&array_i32(10)), Some((0, 10)));
        }
    }

    #[test]
    fn min_max_simd_i32_works() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let size = rng.gen_range(0..i32::MAX);
            unsafe {
                assert_eq!(min_max_simd_i32(&array_i32(size as usize)), Some((0, size)));
            }
        }
    }
}
