// src/training/shuffle.rs

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub fn shuffled_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(seed);

    for i in (1..n).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }

    indices
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_length_preserved() {
        let order = shuffled_indices(100, 42);
        assert_eq!(order.len(), 100);
    }

    #[test]
    fn test_contains_all_indices() {

        let order = shuffled_indices(50, 42);
        let mut seen = vec![false; 50];
        for &i in &order {
            assert!(i < 50);
            assert!(!seen[i], "index {} appeared twice", i);
            seen[i] = true;
        }
        for s in seen {
            assert!(s);
        }
    }

    #[test]
    fn test_same_seed_same_order() {

        let a = shuffled_indices(100, 42);
        let b = shuffled_indices(100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn test_different_seeds_differ() {

        let a = shuffled_indices(100, 42);
        let b = shuffled_indices(100, 7);
        assert_ne!(a, b);
    }

    #[test]
    fn test_empty_and_single() {

        let empty = shuffled_indices(0, 42);
        assert_eq!(empty.len(), 0);

        let single = shuffled_indices(1, 42);
        assert_eq!(single, vec![0]);
    }
}
