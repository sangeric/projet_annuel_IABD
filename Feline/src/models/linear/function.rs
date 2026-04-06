use crate::load_data::DataSet;
use crate::models::linear::Model;
use rand::rng;
use rand::RngExt;
pub fn train_linear(data_set: &DataSet, size_width: u32, size_height: u32, nb_image_train:usize) ->(Vec<Vec<f32>>, Vec<f32>) {
    let nb_classes = 3;
    let lr: f32 = 0.1;
    let nb_epochs = 500;
    let number_features:usize = (size_width*size_height*3).try_into().unwrap();

    let random_weights = || {
        (0..number_features)
            .map(|_| rng().random_range(-0.01..0.01))
            .collect::<Vec<f32>>()
    };

    let random_bias = || rng().random_range(-0.01..0.01);

    let mut cat = Model {
        bias: random_bias(),
        weight: random_weights(),
    };

    let mut lion = Model {
        bias: random_bias(),
        weight: random_weights(),
    };

    let mut cheetah = Model {
        bias: random_bias(),
        weight: random_weights(),
    };

    println!("after \n\n\n");
    let mut scores = vec![vec![0.0; nb_classes]; nb_image_train];
    scores = score_linear(data_set, &cat, &lion, &cheetah, nb_classes, nb_image_train, number_features);

    println!("ici scores");
    for i in 0..nb_image_train{
        for j in 0..nb_classes{
            print!("{} ", scores[i][j]);
        }
        println!();
    }




    training_handler(&data_set, &mut cat, &mut lion, &mut cheetah, nb_classes, nb_image_train, number_features, lr, nb_epochs);
    scores = score_linear(data_set, &cat, &lion, &cheetah, nb_classes, nb_image_train, number_features);

    let mut softmax = Vec::<Vec<f32>>::new();
    softmax = softmax_calcul(scores, nb_classes, nb_image_train);
    let mut losses = Vec::<f32>::new();

    losses = loss(softmax,  nb_image_train, data_set.specie.clone());

    for i in 0..nb_image_train{
        if(losses[i] > 1.5){
            println!("cas KO {} ", losses[i]);
        }
        else {
            println!("cas non KO {} ", losses[i]);
        }

    }

    let weights = vec![cat.weight, lion.weight, cheetah.weight];
    let biases = vec![cat.bias, lion.bias, cheetah.bias];
    (weights, biases)

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

pub fn score_image_test(feature: &Vec<f32>, weight: &Vec<f32>, bias: f32) -> f32 {
    let mut score: f32 = 0.0;
    let number_features = feature.len();

    for i in 0..number_features {
        score += weight[i] * feature[i];
    }

    score + bias
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

pub fn loss(softmaxs: Vec::<Vec<f32>>, nb_image_train: usize, species: Vec<u8>)->Vec::<f32>{
    let mut losses = vec![0.0;nb_image_train];

    for i in 0..nb_image_train{
        let mut index = species[i] as usize;
        let p = softmaxs[i][index].max(1e-9);
        losses[i] = -p.ln();
    }
    losses
}


pub fn training_handler(data_set: &DataSet, cat: &mut Model, lion: &mut Model, cheetah: &mut Model,
                        nb_classes: usize, nb_image_train: usize, number_features: usize, lr: f32, nb_epochs: usize) {

    for epoch in 0..nb_epochs {
        let scores = score_linear(data_set, &cat, &lion, &cheetah, nb_classes, nb_image_train, number_features);
        let softmax = softmax_calcul(scores, nb_classes, nb_image_train);

        let mut d_cat = vec![0.0; number_features];
        let mut d_lion = vec![0.0; number_features];
        let mut d_cheetah = vec![0.0; number_features];

        let mut db_cat = 0.0;
        let mut db_lion = 0.0;
        let mut db_cheetah = 0.0;

        for i in 0..nb_image_train {
            for j in 0..nb_classes {
                let mut indicator = 0.0;
                if data_set.specie[i] as usize == j {
                    indicator = 1.0;
                }
                let delta = softmax[i][j] - indicator;

                match j {
                    0 => {
                        for k in 0..number_features {
                            d_cat[k] += delta * data_set.features[i][k];
                        }
                        db_cat += delta;
                    }
                    1 => {
                        for k in 0..number_features {
                            d_lion[k] += delta * data_set.features[i][k];
                        }
                        db_lion += delta;
                    }
                    2 => {
                        for k in 0..number_features {
                            d_cheetah[k] += delta * data_set.features[i][k];
                        }
                        db_cheetah += delta;
                    }
                    _ => {}
                }
            }
        }

        for k in 0..number_features {
            cat.weight[k] -= lr * d_cat[k] / nb_image_train as f32;
            lion.weight[k] -= lr * d_lion[k] / nb_image_train as f32;
            cheetah.weight[k] -= lr * d_cheetah[k] / nb_image_train as f32;
        }
        cat.bias -= lr * db_cat / nb_image_train as f32;
        lion.bias -= lr * db_lion / nb_image_train as f32;
        cheetah.bias -= lr * db_cheetah / nb_image_train as f32;

        let losses = loss(softmax, nb_image_train, data_set.specie.clone());
        let mut sum: f32 = 0.0;
        for &l in &losses {
            sum += l;
        }
        let avg_loss: f32 = sum / (nb_image_train as f32);

        println!("Epoch {}: avg_loss = {:.6}", epoch + 1, avg_loss);
    }
}

pub fn test_handler(test: &DataSet, weights: &Vec<Vec<f32>>, biases: &Vec<f32>) {
    let nb_classes = weights.len();
    let nb_images = test.features.len();
    let number_features = test.features[0].len();

    let mut scores: Vec<Vec<f32>> = vec![vec![0.0; nb_classes]; nb_images];

    for i in 0..nb_images {
        for j in 0..nb_classes {
            scores[i][j] = score_image(&test.features[i], &weights[j], number_features);
        }
    }

    let probs = softmax_calcul(scores, nb_classes, nb_images);

    let mut correct = 0;
    for i in 0..nb_images {
        let (max_index, _) = probs[i]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        if max_index as u8 == test.specie[i] {
            correct += 1;
        }
    }

    let accuracy = correct as f32 / nb_images as f32;
    println!("Accuracy: {}", accuracy);
}