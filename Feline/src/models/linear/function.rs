use image::RgbImage;
use crate::load_data::DataSet;
use crate::models::linear::Model;

pub fn train_linear(data_set: &DataSet, size_width: u32, size_height: u32, nb_image_train:usize){
    let nb_classes = 3;
    let number_features:usize = (size_width*size_height*3).try_into().unwrap();
    let mut cat = Model{
        bias: 0.2,
        weight: vec![0.0; number_features],
    };
    let mut lion = Model{
        bias: 0.3,
        weight: vec![0.0; number_features],
    };
    let mut cheetah = Model{
        bias: 0.2,
        weight: vec![0.0; number_features],
    };
    println!("after \n\n\n");
    let mut scores = vec![vec![0.0; nb_classes]; nb_image_train];
    scores = score_linear(data_set, &cat, &lion, &cheetah, nb_classes, nb_image_train, number_features);
    let mut softmax = Vec::<Vec<f32>>::new();
    softmax = softmax_calcul(scores, nb_classes, nb_image_train);
    for i in 0..nb_image_train{
        for j in 0..nb_classes{
            print!("{} ", softmax[i][j]);
        }
        println!();
    }
}

//z(x,w)= dot(x, w)+b
pub fn score_linear(data_set: &DataSet, cat: &Model, lion: &Model, guepard :&Model, nb_classes:usize, nb_image_train :usize, number_features: usize) -> Vec<Vec<f32>>{
    println!("{}", nb_image_train);
    let mut scores = vec![vec![0.0; nb_classes]; nb_image_train];
    for i in 0..nb_image_train{
        let image = &data_set.features[i];
        for j in 0..nb_classes{
            if(j == 0){
                scores[i][j] = score_image(image, &cat.weight, number_features) + cat.bias;
            }
            else if(j == 1){
                scores[i][j] = score_image(image, &lion.weight, number_features) + lion.bias;
            }
            else if(j == 2){
                scores[i][j] = score_image(image, &guepard.weight, number_features) + guepard.bias;
            }
        }
    }
    scores
}

pub fn score_image(rgb_image: &Vec<f32>, weight: &Vec<f32>, number_features: usize) -> f32 {
    let mut score: f32 = 0.0;
    for i in 0..number_features {
        score += weight[i] * rgb_image[i];
    }
    score
}

pub fn softmax_calcul(scores :Vec<Vec<f32>>, nb_classes :usize, nb_image_train: usize)->Vec<Vec<f32>>{
    let mut softmax = vec![vec![0.0;nb_classes];nb_image_train ];

    for i in 0..nb_image_train{
        let mut sum_denominator = 0.0;
        for j in 0..nb_classes{
            sum_denominator += scores[i][j].exp();
        }
        for k in 0..nb_classes{
            softmax[i][k] = scores[i][k].exp() / sum_denominator;
        }
    }
    softmax
}