use rand::RngExt;
use crate::tensor::Matrix;

use super::{Rosenblatt, RosenblattClassifier};
use rand::SeedableRng;

impl Rosenblatt {
    pub fn new(n_features: usize, learning_rate: f32, bias: f32, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut weight_data = Vec::new();

        for _ in 0..n_features+1{
            weight_data.push(rng.random_range(-0.01..0.01));
        }

        Self{
            learning_rate,
            weights : Matrix::from_vec(weight_data,1, n_features+1),
            bias,
        }
    }



    pub fn get_weight(&mut self)->&Matrix{
        return &self.weights;
    }

    pub fn get_bias(&self) -> f32 {
        self.bias
    }

    pub fn get_learning_rate(&self) -> f32 {
        self.learning_rate
    }

    pub fn to_binary_labels(y : &Vec<usize>, value :usize)->Vec<i32>{
        let mut new_y = Vec::new();
        for i in 0..y.len(){
            if y[i] == value{
                new_y.push(1);
            }
            else {
                new_y.push(-1);
            }
        }
        new_y
    }

    pub fn train(&mut self, x : &Matrix, y: &Vec<i32>, epochs : usize){
        //add the bias
        let x_with_bias = Self::add_bias_column(self,x);
        let nb_sample :f32 = x.rows as f32;

        let mut tab_success:Vec<f32> = vec![0.0; epochs];

        for epoch in 0..epochs{
            for position in 0..x.rows{
                let xk = Self::get_xk(&x_with_bias,position);
                let g_xk = Self::somme(&xk, &self.weights);
                let prediction:f32 = if g_xk > 0.0 {
                    1.0
                }else{
                    -1.0
                };

                let yk = y[position] as f32;

                let error = yk - prediction as f32;

                if prediction == yk{
                    tab_success[epoch] += 1.0;
                }

                for i in 0..self.weights.data.len() {
                    self.weights.data[i] +=
                        self.learning_rate * error * xk.data[i];
                }
            }

        }
        println!();
        Self::fetch_accuracy(epochs, tab_success, nb_sample);
    }

    pub fn train_with_eval(
        &mut self,
        x: &Matrix,
        y: &Vec<i32>,
        x_test: &Matrix,
        y_test: &Vec<i32>,
        epochs: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let x_with_bias = Self::add_bias_column(self, x);
        let nb_sample: f32 = x.rows as f32;

        let mut train_history = Vec::with_capacity(epochs);
        let mut test_history = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut success = 0.0;

            for position in 0..x.rows {
                let xk = Self::get_xk(&x_with_bias, position);
                let g_xk = Self::somme(&xk, &self.weights);
                let prediction: f32 = if g_xk > 0.0 { 1.0 } else { -1.0 };
                let yk = y[position] as f32;
                let error = yk - prediction;

                if prediction == yk {
                    success += 1.0;
                }

                for i in 0..self.weights.data.len() {
                    self.weights.data[i] += self.learning_rate * error * xk.data[i];
                }
            }

            let train_acc = success / nb_sample;
            let test_acc = Self::compute_accuracy_with_eval(self, x_test, y_test);

            println!(
                "Epoch {:>4} | Train: {:>5.1}% | Test: {:>5.1}%",
                epoch, train_acc * 100.0, test_acc * 100.0
            );

            train_history.push(train_acc);
            test_history.push(test_acc);
        }

        (train_history, test_history)
    }

    pub fn compute_accuracy_with_eval(&mut self, x: &Matrix, y: &Vec<i32>) -> f32 {
        let raw_scores = Self::predict(self, x);
        let mut success = 0.0;

        for i in 0..raw_scores.len() {
            let prediction = if raw_scores[i] > 0.0 { 1.0 } else { -1.0 };
            if prediction == y[i] as f32 {
                success += 1.0;
            }
        }

        success / x.rows as f32
    }

    pub fn compute_accuracy(&mut self, x: &Matrix, y: &Vec<i32>) -> f32 {
        let x_with_bias = Self::add_bias_column(self, x);
        let mut success = 0.0;

        for position in 0..x.rows {
            let xk = Self::get_xk(&x_with_bias, position);
            let g_xk = Self::somme(&xk, &self.weights);
            let prediction: f32 = if g_xk > 0.0 { 1.0 } else { -1.0 };
            if prediction == y[position] as f32 {
                success += 1.0;
            }
        }

        success / x.rows as f32
    }




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

    pub fn get_xk(x_with_bias: &Matrix, position: usize) -> Matrix {
        let mut xk = Vec::with_capacity(x_with_bias.cols);
        for col in 0..x_with_bias.cols {
            xk.push(x_with_bias.get(position, col));
        }
        Matrix::from_vec(xk, 1, x_with_bias.cols)
    }

    pub fn fetch_accuracy(epochs: usize, successes: Vec<f32>, nb_sample: f32){
        for i in 0..successes.len() {
            println!(
                "Epoch {:>4} | Success: {:>4} | Accuracy: {:>6.2}%",
                i,
                successes[i],
                (successes[i] / nb_sample) * 100.0
            );
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

    pub fn print_prediction_result(prediction : &Vec<usize>, label : &Vec<usize>){
        let class_names = ["cat", "cheetah", "lion"];
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

    pub fn transform(x : &Matrix) -> Matrix{
        let mut x_transformed =Vec::new();

        for i in 0..x.rows{
            for j in 0..x.cols{
                x_transformed.push(x.get(i, j));
            }

            for a in 0..x.cols {
                for b in a..x.cols {
                    x_transformed.push(x.get(i, a) * x.get(i, b));
                }
            }
        }
        let new_col = x.cols + (x.cols * (x.cols+1))/2;
        Matrix::from_vec(x_transformed, x.rows , new_col)
    }



    pub fn argmax(cat : &Vec<f32>, lion : &Vec<f32>, cheetah : &Vec<f32>) -> Vec<usize>{
        let mut argmax = Vec::new();

        for i in 0..cat.len() {
            let scores = [cat[i], cheetah[i], lion[i]];
            let max = Self::max_argmax(scores);
            argmax.push(max);
        }

        argmax
    }
    pub fn max_argmax(scores: [f32; 3]) -> usize{
        let mut index_max : usize = 0;
        let mut max : f32 = scores[0];
        for i in 1..scores.len(){
            if max < scores[i]{
                max = scores[i];
                index_max = i;
            }
        }
        index_max
    }

}

#[repr(C)]
pub struct MatrixFFIRos{
    data: *mut f32,
    rows: usize,
    cols: usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn create_rosenblatt(n_features : usize, learning_rate : f32, bias : f32, seed : u64) -> *mut Rosenblatt{
    let rosenblatt = Box::new(Rosenblatt::new(n_features, learning_rate, bias, seed));
    Box::into_raw(rosenblatt)
}

#[unsafe(no_mangle)]
pub extern "C" fn release_rosenblatt(rosenblatt: *mut Rosenblatt){
    unsafe {
        let _ = Box::from_raw(rosenblatt);
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn train_rosenblatt(
    rosenblatt: *mut Rosenblatt,
    x: *const f32,
    rows: usize,
    cols: usize,
    y: *const i32,
    y_len: usize,
    epochs : usize,
) {
    unsafe {
        let rosenblatt = unsafe {
            match rosenblatt.as_mut() {
                Some(c) => c,
                None => return,
            }
        };
        let x_slice = std::slice::from_raw_parts(x, rows * cols);
        let x_matrix = Matrix::from_vec(x_slice.to_vec(), rows, cols);

        let y_i32: Vec<i32> = std::slice::from_raw_parts(y, y_len).to_vec();

        rosenblatt.train(&x_matrix, &y_i32, epochs);
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn predict_rosenblatt(
    rosenblatt: *mut Rosenblatt,
    X: *const f32,
    rows: usize,
    cols: usize
) -> *mut f32{
    unsafe {
        let rosenblatt = unsafe {
            match rosenblatt.as_mut() {
                Some(c) => c,
                None => return std::ptr::null_mut(),
            }
        };

        let x_slice = std::slice::from_raw_parts(X, rows * cols);

        let x_matrix = Matrix::from_vec(x_slice.to_vec(),rows, cols);

        let mut final_predict = rosenblatt.predict(&x_matrix);
        final_predict.shrink_to_fit();
        final_predict.leak().as_mut_ptr()
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn transforming_ros(X: *const f32,
                                    rows : usize,
                                    cols : usize,

)-> *mut MatrixFFIRos{
    unsafe{
        let x_vec = std::slice::from_raw_parts(X, rows*cols).to_vec();
        let x_matrix = Matrix::from_vec(x_vec, rows, cols);
        let x_transformed = Rosenblatt::transform(&x_matrix);

        let data = x_transformed.data.leak().as_mut_ptr();
        let rows = x_transformed.rows;
        let cols = x_transformed.cols;

        Box::into_raw(Box::new(MatrixFFIRos{data, rows,cols}))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn get_weights_rosenblatt(
    rosenblatt: *mut Rosenblatt,
) -> *mut f32 {
    unsafe {
        let rosenblatt = match rosenblatt.as_mut() {
            Some(c) => c,
            None => return std::ptr::null_mut(),
        };

        let weights = rosenblatt.get_weight();
        let mut weights_vec = weights.data.clone();
        weights_vec.shrink_to_fit();
        weights_vec.leak().as_mut_ptr()
    }
}