use rand::RngExt;
use crate::tensor::Matrix;

use super::Rosenblatt;
use rand::SeedableRng;

impl Rosenblatt {
    pub fn new(n_features: usize, learning_rate: f32, bias: f32, target_class: usize, seed_offset: u64) -> Self {
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64 + seed_offset;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut weight_data = Vec::new();
        //biais
        for _ in 0..n_features+1{
            weight_data.push(rng.random_range(-0.01..0.01));
        }

        Self{
            learning_rate,
            target_class,
            weights : Matrix::from_vec(weight_data,1, n_features+1),
            bias,
        }
    }


    pub fn get_target(&mut self)->usize{
        return self.target_class;
    }

    pub fn get_weight(&mut self)->&Matrix{
        return &self.weights;
    }


    //  Le cours Rosenblatt: 𝑊 ← 𝑊 + 𝛼 ( 𝑌𝑘 − 𝑔(𝑋𝑘 ) 𝑋𝑘
    // 𝛼 le pas d’apprentissage
    // 𝑋𝑘 les paramètres de l’exemples k et le biais 𝑥0  𝑘 = 1.
    // 𝑌𝑘 la sortie attendue pour l’exemple k.
    // 𝑔(𝑋𝑘 ) la sortie obtenue par le perceptron pour l’exemple k.
    pub fn train(&mut self, x : &Matrix, y: &[usize], epochs : usize){
        //add the bias
        let x_with_bias = Self::add_bias_column(self,x);
        let nb_sample :f32 = x.rows as f32;

        let mut tab_accuracy:Vec<f32> = vec![0.0; epochs/100];

        for epoch in 0..epochs{
            for position in 0..x.rows{
                let xk = Self::get_xk(&x_with_bias,position);
                let g_xk = Self::somme(&xk, &self.weights);
                let prediction:f32 = if g_xk > 0.0 {
                    1.0
                }else{
                    -1.0
                };

                let yk:f32 = if y[position] == self.target_class{
                    1.0
                }else{
                    -1.0
                };

                let error = yk - prediction as f32;

                if prediction == yk{
                    tab_accuracy[epoch/100]+=1.0;
                }

                for i in 0..self.weights.data.len() {
                    self.weights.data[i] +=
                        self.learning_rate * error * xk.data[i];
                }
            }

        }
        println!("iciiiiiiii tab accuracy ------------------ {:?}", tab_accuracy);
        Self::fetch_accuracy(epochs, tab_accuracy, nb_sample);
    }




    //we add 1 at the begining of every rows of the dataset because of the bias
    pub fn add_bias_column(&mut self,x : &Matrix) -> Matrix{
        let mut flat_x = Vec::new();

        for rows in 0..x.rows {
            flat_x.push(self.bias);
            for cols in 0..x.cols{
                flat_x.push(x.get(rows, cols));
            }
        }
        Matrix::from_vec(flat_x, x.rows, x.cols +1)
    }

    pub fn somme(x: &Matrix, weight: &Matrix)->f32{
        let mut somme:f32 = 0.0;
        for i in 0..x.cols{
            somme += x.get(0,i) * weight.get(0,i);
        }
        somme
    }

    pub fn get_xk(x_with_bias: &Matrix, position: usize) -> Matrix{
        let mut xk = Vec::new();
        for rows in 0..x_with_bias.rows {
            for cols in 0..x_with_bias.cols{
                if rows == position{
                    xk.push(x_with_bias.get(rows, cols));
                }
            }
        }
        Matrix::from_vec(xk, 1, x_with_bias.cols)
    }

    pub fn fetch_accuracy(epoch:usize, errors: Vec<f32>, nb_sample: f32){
        let epoch_f32 = epoch as f32;
        for i in 0..errors.len() {
            println!("Epoch {:>4}-{} | Success: {:>4} | Accuracy: {:>6.2}%", i*100,(i+1)*100, errors[i] , (errors[i] / (nb_sample * 100.0)) * 100.0 );
        }
    }

    pub fn predict(&mut self, x: &Matrix) -> Vec<f32>{
        let x_with_bias = Self::add_bias_column(self,x);
        let mut prediction = Vec::new();

        for row in 0..x.rows{
            let xk = Self::get_xk(&x_with_bias,row);
            let g_xk = Self::somme(&xk, &self.get_weight());
            prediction.push(g_xk);
        }
        prediction
    }

    pub fn print_prediction_result(prediction : Vec<usize>, label : &Vec<usize>){
        let class_names = ["cat", "lion", "cheetah"];
        for i in 0..prediction.len(){
            let status = if prediction[i] == label[i] {"OK"} else {"KO"};
            println!(
                "  Sample {:>2} | predicted: {:>7} | actual: {:>7} | {}",
                i,
                class_names[prediction[i]],
                class_names[label[i]],
                status
            )
        }
    }

    pub fn transform(&mut self, x : &Matrix, y_size : usize) -> Matrix{
        let mut x_transform =Vec::new();

        for i in 0..y_size{
            for j in 0..x.cols{
                x_transform.push(x.get(i, j));
            }

            let x1 = x.get(i, 0);
            let x2 = x.get(i, 1);

            x_transform.push(x1.powi(2));
            x_transform.push(x2.powi(2));
            x_transform.push(x1 * x2);
        }
        //on met +3 car x_with biais contient déjà biais, x1 et x2 il manque x1², x2² et x1*x2
        Matrix::from_vec(x_transform, y_size , x.cols + 3)
    }
}