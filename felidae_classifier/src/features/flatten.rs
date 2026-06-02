// src/features/flatten.rs

use crate::data::image::LoadedImage;

pub fn flatten(image: &LoadedImage) -> Vec<f32> {
    image.pixels.clone()
}

pub fn feature_count() -> usize {
    use crate::data::image::IMAGE_SIZE;
    (IMAGE_SIZE * IMAGE_SIZE * 3) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::image::{LoadedImage, IMAGE_SIZE};

    fn make_test_image() -> LoadedImage {
        let n_values = (IMAGE_SIZE * IMAGE_SIZE * 3) as usize;
        let pixels = vec![0.5_f32; n_values];

        LoadedImage {
            pixels,
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
        }
    }

    #[test]
    fn test_flatten_length() {
        let img = make_test_image();
        let features = flatten(&img);

        assert_eq!(features.len(), 3072);
    }

    #[test]
    fn test_flatten_values() {
        let img = make_test_image();
        let features = flatten(&img);

        for v in &features {
            assert_eq!(*v, 0.5);
        }
    }

    #[test]
    fn test_features_count() {
        let img = make_test_image();
        let features = flatten(&img);
        assert_eq!(features.len(), feature_count());
    }
}
