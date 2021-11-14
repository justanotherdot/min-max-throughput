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

#[cfg(test)]
mod test {
    use super::*;
    use rand::prelude::*;

    fn vector() -> [i64; 10] {
        let mut rng = thread_rng();
        let mut result = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
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
}
