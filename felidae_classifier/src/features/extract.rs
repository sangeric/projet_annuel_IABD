// src/features/extract.rs

use crate::data::image::LoadedImage;

// Number of features extract() produces.
// 6 color stats + 8 histogram + 6 edge = 20 features
pub const FEATURE_COUNT: usize = 20;

// Number of brightness histogram buckets.
const N_HIST_BUCKETS: usize = 8;

// Two edge-detection thresholds for richer texture information.
const EDGE_THRESHOLDS: [f32; 2] = [0.1, 0.25];

/// Extracts a fixed-size feature vector from a loaded image.
///
/// Features (in order):
///   [0..6]   color stats: mean_r, mean_g, mean_b, std_r, std_g, std_b
///   [6..14]  brightness histogram: 8 buckets covering 0.0 to 1.0
///   [14..20] edge density: 3 channels × 2 thresholds
///
/// These features capture:
///   - average color of the subject
///   - color variation (uniform vs patterned)
///   - brightness distribution
///   - texture density (smooth vs textured surfaces)
pub fn extract(image: &LoadedImage) -> Vec<f32> {
    let mut features = Vec::new();

    // Group 1: per-channel mean and standard deviation
    let (mean_r, mean_g, mean_b) = channel_means(image);
    features.push(mean_r);
    features.push(mean_g);
    features.push(mean_b);

    let (std_r, std_g, std_b) = channel_stds(image, mean_r, mean_g, mean_b);
    features.push(std_r);
    features.push(std_g);
    features.push(std_b);

    // Group 2: brightness histogram
    let histogram = brightness_histogram(image);
    for bucket in histogram {
        features.push(bucket);
    }

    // Group 3: edge density per channel, at each threshold
    for threshold in EDGE_THRESHOLDS {
        let (e_r, e_g, e_b) = edge_density(image, threshold);
        features.push(e_r);
        features.push(e_g);
        features.push(e_b);
    }

    features
}

// Returns the number of features extract() produces.
pub fn feature_count() -> usize {
    FEATURE_COUNT
}

// Mean value of each color channel across all pixels.
fn channel_means(image: &LoadedImage) -> (f32, f32, f32) {
    let mut sum_r = 0.0_f32;
    let mut sum_g = 0.0_f32;
    let mut sum_b = 0.0_f32;
    let n_pixels = (image.width * image.height) as f32;

    for y in 0..image.height {
        for x in 0..image.width {
            let (r, g, b) = image.get_pixel(x, y);
            sum_r += r;
            sum_g += g;
            sum_b += b;
        }
    }

    (sum_r / n_pixels, sum_g / n_pixels, sum_b / n_pixels)
}

// Standard deviation per color channel.
// High std = lots of color variation; low std = uniform color.
fn channel_stds(image: &LoadedImage, mean_r: f32, mean_g: f32, mean_b: f32) -> (f32, f32, f32) {
    let mut sum_sq_r = 0.0_f32;
    let mut sum_sq_g = 0.0_f32;
    let mut sum_sq_b = 0.0_f32;
    let n_pixels = (image.width * image.height) as f32;

    for y in 0..image.height {
        for x in 0..image.width {
            let (r, g, b) = image.get_pixel(x, y);
            sum_sq_r += (r - mean_r) * (r - mean_r);
            sum_sq_g += (g - mean_g) * (g - mean_g);
            sum_sq_b += (b - mean_b) * (b - mean_b);
        }
    }

    (
        (sum_sq_r / n_pixels).sqrt(),
        (sum_sq_g / n_pixels).sqrt(),
        (sum_sq_b / n_pixels).sqrt(),
    )
}

// 8-bucket histogram of pixel brightnesses (average of R, G, B per pixel).
// Each bucket value is the fraction of pixels falling in that bucket.
// Returns a Vec<f32> of length 8, summing to 1.0.
fn brightness_histogram(image: &LoadedImage) -> Vec<f32> {
    let mut counts = vec![0.0_f32; N_HIST_BUCKETS];
    let n_pixels = (image.width * image.height) as f32;

    for y in 0..image.height {
        for x in 0..image.width {
            let (r, g, b) = image.get_pixel(x, y);
            let brightness = (r + g + b) / 3.0;

            let mut bucket = (brightness * N_HIST_BUCKETS as f32) as usize;
            if bucket >= N_HIST_BUCKETS {
                bucket = N_HIST_BUCKETS - 1;
            }
            counts[bucket] += 1.0;
        }
    }

    for c in counts.iter_mut() {
        *c /= n_pixels;
    }
    counts
}

// Edge density per channel — fraction of pixels whose value differs from a
// neighbour by more than a threshold.
// High edge density = textured surface. Low = smooth surface.
fn edge_density(image: &LoadedImage, threshold: f32) -> (f32, f32, f32) {
    let mut edges_r = 0.0_f32;
    let mut edges_g = 0.0_f32;
    let mut edges_b = 0.0_f32;
    let mut total = 0.0_f32;

    for y in 0..image.height - 1 {
        for x in 0..image.width - 1 {
            let (r, g, b) = image.get_pixel(x, y);
            let (r_right, g_right, b_right) = image.get_pixel(x + 1, y);
            let (r_down, g_down, b_down) = image.get_pixel(x, y + 1);

            let diff_r = (r - r_right).abs().max((r - r_down).abs());
            let diff_g = (g - g_right).abs().max((g - g_down).abs());
            let diff_b = (b - b_right).abs().max((b - b_down).abs());

            if diff_r > threshold {
                edges_r += 1.0;
            }
            if diff_g > threshold {
                edges_g += 1.0;
            }
            if diff_b > threshold {
                edges_b += 1.0;
            }
            total += 1.0;
        }
    }

    (edges_r / total, edges_g / total, edges_b / total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::image::IMAGE_SIZE;

    fn make_uniform_image(r: f32, g: f32, b: f32) -> LoadedImage {
        let mut pixels = Vec::new();
        for _ in 0..(IMAGE_SIZE * IMAGE_SIZE) {
            pixels.push(r);
            pixels.push(g);
            pixels.push(b);
        }
        LoadedImage {
            pixels,
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
        }
    }

    fn make_checkerboard() -> LoadedImage {
        let mut pixels = Vec::new();
        for y in 0..IMAGE_SIZE {
            for x in 0..IMAGE_SIZE {
                let value = if (x + y) % 2 == 0 { 0.0 } else { 1.0 };
                pixels.push(value);
                pixels.push(value);
                pixels.push(value);
            }
        }
        LoadedImage {
            pixels,
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
        }
    }

    #[test]
    fn test_feature_count() {
        let img = make_uniform_image(0.5, 0.5, 0.5);
        let features = extract(&img);
        assert_eq!(features.len(), FEATURE_COUNT);
        assert_eq!(features.len(), 20);
    }

    #[test]
    fn test_uniform_image_means() {
        let img = make_uniform_image(1.0, 0.0, 0.0);
        let features = extract(&img);
        assert!((features[0] - 1.0).abs() < 1e-5);
        assert!((features[1] - 0.0).abs() < 1e-5);
        assert!((features[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_uniform_image_zero_std() {
        let img = make_uniform_image(0.7, 0.7, 0.7);
        let features = extract(&img);
        assert!(features[3].abs() < 1e-5);
        assert!(features[4].abs() < 1e-5);
        assert!(features[5].abs() < 1e-5);
    }

    #[test]
    fn test_uniform_image_no_edges() {
        let img = make_uniform_image(0.5, 0.5, 0.5);
        let features = extract(&img);
        for i in 14..20 {
            assert!(features[i].abs() < 1e-5, "expected no edges, got {}", features[i]);
        }
    }

    #[test]
    fn test_checkerboard_has_many_edges() {
        let img = make_checkerboard();
        let features = extract(&img);
        for i in 14..17 {
            assert!(features[i] > 0.9, "expected many edges, got {}", features[i]);
        }
    }

    #[test]
    fn test_histogram_sums_to_one() {
        let img = make_uniform_image(0.5, 0.5, 0.5);
        let features = extract(&img);
        let mut sum = 0.0_f32;
        for i in 6..14 {
            sum += features[i];
        }
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
