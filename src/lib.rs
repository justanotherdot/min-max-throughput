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
pub unsafe fn min_max_simd_i16(buff: &[i16]) -> Option<(i16, i16)> {
    // need to consider
    // * excess padding on non multiples of 8, e.g. 9 elements
    // * negative value
    // * upgrading to i64 buffer, rather than i16
    use core::arch::x86_64::*;
    unsafe fn store_to_mm_128i(
        x7: i16,
        x6: i16,
        x5: i16,
        x4: i16,
        x3: i16,
        x2: i16,
        x1: i16,
        x0: i16,
    ) -> __m128i {
        _mm_set_epi16(x0, x1, x2, x3, x4, x5, x6, x7)
    }
    let mut maxmax = _mm_setzero_si128();
    let mut minmin = _mm_setzero_si128();
    let mut max = *buff.get(0)?;
    let mut min = *buff.get(0)?;
    let f8: &[__m128i] = &(0..(buff.len() / 8) + 1)
        .into_iter()
        .map(|ix| {
            store_to_mm_128i(
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
    let mut maxval = _mm_setzero_si128();
    let mut minval = _mm_setzero_si128();
    for chunk in f8 {
        maxval = _mm_max_epi16(maxval, *chunk);
        minval = _mm_min_epi16(minval, *chunk);
    }
    //let mut maxval2 = maxval;
    //let mut minval2 = minval;
    // FIXME: I don't think the below actually does anything yet.
    //for _ in 0..3 {
    //    //pp(maxval);
    //    //pp(_mm_shufflehi_epi16::<3>(maxval));
    //    //pp(_mm_shufflehi_epi16::<8>(maxval));

    //    maxval = _mm_max_epi16(maxval, _mm_shufflehi_epi16::<3>(maxval));
    //    _mm_store_si128(&mut maxmax as *mut __m128i, maxval);
    //    maxval2 = _mm_max_epi16(maxval2, _mm_shufflelo_epi16::<3>(maxval2));
    //    _mm_store_si128(&mut maxmax as *mut __m128i, maxval2);

    //    minval = _mm_min_epi16(minval, _mm_shufflehi_epi16::<3>(minval));
    //    _mm_store_si128(&mut minmin as *mut __m128i, minval);
    //    minval2 = _mm_min_epi16(minval2, _mm_shufflelo_epi16::<3>(minval2));
    //    _mm_store_si128(&mut minmin as *mut __m128i, minval2);
    //}
    _mm_store_si128(&mut maxmax as *mut __m128i, maxval);
    _mm_store_si128(&mut minmin as *mut __m128i, minval);
    let maxmax: [i16; 8] = std::mem::transmute(maxmax);
    let minmin: [i16; 8] = std::mem::transmute(minmin);
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

    fn array_i16(size: usize) -> Vec<i16> {
        let mut rng = thread_rng();
        let mut result: Vec<_> = std::iter::repeat(0)
            .take(size)
            .enumerate()
            .map(|(i, _)| i as i16 + 1)
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
    fn min_max_simd_i16_work_with_exact_sized_arrays() {
        unsafe {
            assert_eq!(min_max_simd_i16(&array_i16(0)), None);
            assert_eq!(min_max_simd_i16(&array_i16(8)), Some((0, 8)));
            assert_eq!(min_max_simd_i16(&array_i16(16)), Some((0, 16)));
        }
    }

    #[test]
    fn min_max_simd_i16_work_with_inexact_sized_arrays() {
        unsafe {
            assert_eq!(min_max_simd_i16(&array_i16(9)), Some((0, 9)));
            assert_eq!(min_max_simd_i16(&array_i16(10)), Some((0, 10)));
        }
    }

    #[test]
    fn min_max_simd_i16_works() {
        let mut rng = thread_rng();
        for _ in 0..100 {
            let size = rng.gen_range(0..i16::MAX);
            unsafe {
                assert_eq!(min_max_simd_i16(&array_i16(size as usize)), Some((0, size)));
            }
        }
    }
}
