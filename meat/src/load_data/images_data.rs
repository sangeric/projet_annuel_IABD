use crate::load_data::DataSet;
use std::env;
use std::path::PathBuf;
use image::{DynamicImage, GenericImageView};
use image::imageops::FilterType;

pub fn pipeline(){
    let mut train = DataSet{
        features: Vec::<Vec<f32>>::new(),
        specie: Vec::<u8>::new(),
    };

    build_datasets(r"src\data\test\cats", r"src\data\test\lion",&mut train);

    println!("\n----------------------------------------------------next----------------------------------------------------\n");

    //build_datasets(r"src\data\test\lion", &mut train_lion);

    println!("Train cat images: {}", train.features.len());

    fetch_struct(&mut train);
}


pub fn build_datasets(image_path_cat: &str, image_path_lion: &str, dataset: &mut DataSet){
    let batch_size = 5;
    let size_width :u32 = 32;
    let size_heigh :u32 = 32;
    let cat: Vec<_> = std::fs::read_dir(image_path_cat)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();

    for batch in cat.chunks(batch_size){
        for image_features in batch{
            let transformed_data = {
                let resize = load_resize_image(image_features, size_width, size_heigh);
                resize.save("nom_du_fichier.jpg").unwrap();
                let rgb = convert_rgb(resize);
                normalize_rgb(rgb)
            };
            dataset.features.push(transformed_data);
            dataset.specie.push(0);
            println!("\n-------------------------------------------------------------------------------------------\n");
        }

    }

    let lion: Vec<_> = std::fs::read_dir(image_path_lion)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();

    for batch in lion.chunks(batch_size){
        for image_features in batch{
            let transformed_data = {
                let resize = load_resize_image(image_features, size_width, size_heigh);
                resize.save("nom_du_fichier.jpg").unwrap();
                let rgb = convert_rgb(resize);
                normalize_rgb(rgb)
            };
            dataset.features.push(transformed_data);
            dataset.specie.push(1);
            println!("\n-------------------------------------------------------------------------------------------\n");
        }

    }
}

pub fn load_resize_image(image_path: &PathBuf, size_width: u32, size_height: u32)->DynamicImage{
    let image = image::open(image_path).unwrap();
    let resized = image.resize_exact(size_width, size_height, FilterType::Nearest);
    resized
}

pub fn convert_rgb(dynamic_image: DynamicImage)->Vec::<u8> {

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
    println!("width {} height {}",width,height);
    for (i, val) in pixels.iter().enumerate() {
        print!("{} ", val);

        if (i + 1) % 20 == 0 {
            println!();
        }
    }

    pixels
}

pub fn normalize_rgb(pixel: Vec::<u8>)->Vec::<f32>{
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

            if (j + 1) % 20 == 0 {
                println!();
            }
        }

        println!("\nEspèce: {}\n", specie);
        println!("\n------------------------------------ next image ------------------------------------\n");
    }
}