// src/data/image/rs

use image::imageops::FilterType;

pub const IMAGE_SIZE: u32 = 32;

pub struct LoadedImage {
    pub pixels: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

impl LoadedImage {
    pub fn load(path: &str) -> Result<LoadedImage, String> {
        let img = match image::open(path) {
            Ok(img) => img,
            Err(e) => return Err(format!("Failed to open {}: {}", path, e)),
        };

        let resized = img.resize_exact(IMAGE_SIZE, IMAGE_SIZE, FilterType::Triangle);

        let rgb = resized.to_rgb8();

        let mut pixels = Vec::new();
        for pixel in rgb.pixels() {
            pixels.push(pixel[0] as f32 / 255.0);
            pixels.push(pixel[1] as f32 / 255.0);
            pixels.push(pixel[2] as f32 / 255.0);
        }

        Ok(LoadedImage {
            pixels,
            width: IMAGE_SIZE,
            height: IMAGE_SIZE,
        })
    }

    pub fn get_pixel(&self, x: u32, y: u32) -> (f32, f32, f32) {
        let index = ((y * self.width + x) * 3) as usize;
        (
            self.pixels[index],
            self.pixels[index + 1],
            self.pixels[index + 2],
        )
    }
}
