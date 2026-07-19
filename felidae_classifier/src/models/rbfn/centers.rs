// src/models/rbfn/centers.rs

use crate::tensor::Matrix;
use crate::models::rbfn::basis::squared_distance;


pub fn kmeans(x: &Matrix, k: usize, max_iters: usize, seed: u64) -> Matrix {
    let n_features = x.cols;


    let mut centers = Matrix::zeros(k, n_features);
    let stride = if k > 0 { x.rows / k } else { 1 };
    for c in 0..k {
        let example_index = (c * stride + seed as usize) % x.rows;
        for j in 0..n_features {
            centers.set(c, j, x.get(example_index, j));
        }
    }

    for _iter in 0..max_iters {
        let mut assignements = Vec::new();
        for i in 0..x.rows {
            let mut best_center = 0;
            let mut best_dist = squared_distance(x, i, &centers, 0);
            for c in 1..k {
                let dist = squared_distance(x, i, &centers, c);
                if dist < best_dist {
                    best_dist = dist;
                    best_center = c;
                }
            }
            assignements.push(best_center);
        }


        let mut new_centers = Matrix::zeros(k, n_features);
        let mut counts = vec![0.0_f32; k];

        for i in 0..x.rows {
            let c = assignements[i];
            counts[c] += 1.0;
            for j in 0..n_features {
                let val = new_centers.get(c, j) + x.get(i, j);
                new_centers.set(c, j, val);
            }
        }

        for c in 0..k {
            if counts[c] > 0.0 {
                for j in 0..n_features {
                    let val = new_centers.get(c, j) / counts[c];
                    new_centers.set(c, j, val);
                }
            } else {

                for j in 0..n_features {
                    new_centers.set(c, j, centers.get(c, j));
                }
            }
        }
        centers = new_centers;
    }
    centers
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_output_shape() {

        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            0.1, 0.1,
            0.2, 0.0,
            5.0, 5.0,
            5.1, 5.1,
            4.9, 5.0,
        ], 6, 2);
        let centers = kmeans(&x, 2, 10, 0);
        assert_eq!(centers.rows, 2);
        assert_eq!(centers.cols, 2);
    }

    #[test]
    fn test_kmeans_finds_two_clear_clusters() {

        let x = Matrix::from_vec(vec![
            0.0, 0.0,
            0.2, 0.1,
            0.1, 0.2,
            10.0, 10.0,
            10.2, 9.9,
            9.8, 10.1,
        ], 6, 2);
        let centers = kmeans(&x, 2, 20, 0);


        let dist_between_centers = squared_distance(&centers, 0, &centers, 1);
        assert!(dist_between_centers > 50.0,
            "centers should be far apart, got squared distance {}", dist_between_centers);
    }

    #[test]
    fn test_kmeans_center_is_near_its_points() {

        let x = Matrix::from_vec(vec![
            1.9, 2.1,
            2.0, 2.0,
            2.1, 1.9,
        ], 3, 2);
        let centers = kmeans(&x, 1, 10, 0);
        assert!((centers.get(0, 0) - 2.0).abs() < 0.2);
        assert!((centers.get(0, 1) - 2.0).abs() < 0.2);
    }
}
