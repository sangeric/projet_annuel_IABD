use crate::load_data::DataSet;
use std::path::PathBuf;
use image::{DynamicImage, GenericImageView};
use image::imageops::FilterType;
use rand::prelude::*;
use crate::models::linear::function::train_linear;

pub fn pipeline(){
    let mut train = DataSet{
        features: Vec::<Vec<f32>>::new(),
        specie: Vec::<u8>::new(),
    };
    let size_width:u32 = 32;
    let size_height:u32 = 32;
    build_datasets(r"src\data\test\cats", r"src\data\test\lion", r"src\data\test\cheetah",&mut train, size_width, size_height);
    let nb_image_train = train.features.len();
    println!("\n----------------------------------------------------next----------------------------------------------------\n");
    println!("Train images: {}", nb_image_train);
    train_linear(&train, size_width, size_height, nb_image_train);
    //fetch_struct(&train);
}


pub fn build_datasets(image_path_cat: &str, image_path_lion: &str, image_path_cheetah: &str,dataset: &mut DataSet, size_width:u32, size_height:u32 ) {
    let mut samples: Vec<(Vec<f32>, u8)> = Vec::new();

    let cat: Vec<PathBuf> = std::fs::read_dir(image_path_cat)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    for image_path in cat {
        let transformed_data = {
            let resize = load_resize_image(&image_path, size_width, size_height);
            let rgb = convert_rgb(resize);
            normalize_rgb(rgb)
        };
        samples.push((transformed_data, 0));
    }

    let lion: Vec<PathBuf> = std::fs::read_dir(image_path_lion)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();

    for image_path in lion {
        let transformed_data = {
            let resize = load_resize_image(&image_path, size_width, size_height);
            let rgb = convert_rgb(resize);
            normalize_rgb(rgb)
        };
        samples.push((transformed_data, 1));
    }

    let cheetah: Vec<PathBuf> = std::fs::read_dir(image_path_cheetah)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();

    for image_path in cheetah{
        let transformed_data = {
            let resize = load_resize_image(&image_path, size_width, size_height);
            let rgb = convert_rgb(resize);
            normalize_rgb(rgb)
        };
        samples.push((transformed_data, 2));
    }

    let mut rng = rand::rng();
    samples.as_mut_slice().shuffle(&mut rng);
    dataset.features.clear();
    dataset.specie.clear();
    for (features, label) in samples {
        dataset.features.push(features);
        dataset.specie.push(label);
    }
}

pub fn load_resize_image(image_path: &PathBuf, size_width: u32, size_height: u32)->DynamicImage{
    let image = image::open(image_path).unwrap();
    let resized = image.resize_exact(size_width, size_height, FilterType::Nearest);
    resized
}

pub fn convert_rgb(dynamic_image: DynamicImage)->Vec<u8> {

    let (width, height) = dynamic_image.dimensions();
    let mut pixels = Vec::<u8>::new();

    for y in 0..height{
        for x in 0..width{
            let pixel = dynamic_image.get_pixel(x,y);
            pixels.push(pixel[0]);
            pixels.push(pixel[1]);
            pixels.push(pixel[2]);
        }
    }
    pixels
}

pub fn normalize_rgb(pixel: Vec<u8>)->Vec<f32>{
    let mut normalized =Vec::<f32>::new();
    for i in 0..pixel.len(){
        let val = pixel[i] as f32/255.0;
        normalized.push(val);
    }
    normalized
}

pub fn fetch_struct(data_set: &DataSet) {
    for i in 0..data_set.features.len() {
        let image = &data_set.features[i];
        let specie = data_set.specie[i];
        for (j, val) in image.iter().enumerate() {
            print!("{:.3} ", val);
            if (j + 1) % 25 == 0 {
                println!();
            }
        }
        println!("\n species: {}\n", specie);
        println!("\n------------------------------------ next image ------------------------------------\n");
    }
}