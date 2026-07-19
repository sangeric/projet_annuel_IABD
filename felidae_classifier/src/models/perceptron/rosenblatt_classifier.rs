use crate::models::perceptron::{Rosenblatt, RosenblattClassifier};
use crate::tensor::Matrix;
use std::fs::File;
use std::io::{Read, Write};

use std::os::raw::c_char;
use std::ffi::CStr;

impl RosenblattClassifier{
    pub fn new(n_features: usize, learning_rate : f32, bias_cat: f32, bias_lion: f32, bias_cheetah: f32, seed: u64) -> Self{

        let cat=Rosenblatt::new(n_features,learning_rate, bias_cat,  seed);
        let lion=Rosenblatt::new(n_features,learning_rate, bias_lion,  seed+200);
        let cheetah=Rosenblatt::new(n_features,learning_rate, bias_cheetah,  seed+400);

        Self{
            cat,
            lion,
            cheetah
        }
    }

    pub fn train(&mut self, x : &Matrix, y : &Vec<usize> , epochs : usize){
        let y_cat = Rosenblatt::to_binary_labels(y, 0);
        let y_cheetah = Rosenblatt::to_binary_labels(y, 2);
        let y_lion = Rosenblatt::to_binary_labels(y, 1);


        self.cat.train(x, &y_cat, epochs);
        self.lion.train(x,&y_lion, epochs);
        self.cheetah.train(x,&y_cheetah, epochs);
    }

    pub fn train_with_eval(
        &mut self,
        x: &Matrix,
        y: &Vec<usize>,
        x_test: &Matrix,
        y_test: &Vec<usize>,
        epochs: usize,
    ) -> ((Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>), (Vec<f32>, Vec<f32>)) {
        let y_cat = Rosenblatt::to_binary_labels(y, 0);
        let y_lion = Rosenblatt::to_binary_labels(y, 1);
        let y_cheetah = Rosenblatt::to_binary_labels(y, 2);

        let y_cat_test = Rosenblatt::to_binary_labels(y_test, 0);
        let y_lion_test = Rosenblatt::to_binary_labels(y_test, 1);
        let y_cheetah_test = Rosenblatt::to_binary_labels(y_test, 2);

        let cat_hist = self.cat.train_with_eval(x, &y_cat, x_test, &y_cat_test, epochs);
        let lion_hist = self.lion.train_with_eval(x, &y_lion, x_test, &y_lion_test, epochs);
        let cheetah_hist = self.cheetah.train_with_eval(x, &y_cheetah, x_test, &y_cheetah_test, epochs);

        (cat_hist, lion_hist, cheetah_hist)
    }

    pub fn predict(&mut self, x : &Matrix, y : &Vec<usize>) -> Vec<usize>{
        let cat_predict = self.cat.predict(&x);
        let lion_predict = self.lion.predict(&x);
        let cheetah_predict = self.cheetah.predict(&x);

        let tab_argmax:Vec<usize> = Rosenblatt::argmax(&cat_predict, &lion_predict, &cheetah_predict);
        if y.len() == tab_argmax.len() {
            Rosenblatt::print_prediction_result(&tab_argmax, y);
        }
        tab_argmax
    }

    pub fn transform(&mut self, x : &Matrix) -> Matrix{
        Rosenblatt::transform(&x)
    }

    pub fn get_all_weight(&mut self){
        println!("Cat    : {:?}", self.cat.get_weight());
        println!("Lion   : {:?}", self.lion.get_weight());
        println!("Cheetah: {:?}", self.cheetah.get_weight());
    }

    pub fn get_all_weight_vec(&mut self) -> Vec<f32> {
        let mut all_weights = Vec::new();
        all_weights.extend_from_slice(&self.cat.get_weight().data);
        all_weights.extend_from_slice(&self.lion.get_weight().data);
        all_weights.extend_from_slice(&self.cheetah.get_weight().data);
        all_weights
    }

    pub fn get_all_params(&self){
        println!("Cat    -> learning_rate: {}, bias: {}", self.cat.get_learning_rate(), self.cat.get_bias());
        println!("Lion   -> learning_rate: {}, bias: {}", self.lion.get_learning_rate(), self.lion.get_bias());
        println!("Cheetah-> learning_rate: {}, bias: {}", self.cheetah.get_learning_rate(), self.cheetah.get_bias());
    }


    pub fn save(&self, path: &str, seeds: [u64; 3]) -> Result<(), String> {
        let mut file = match File::create(path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Failed to create {}: {}", path, e)),
        };

        if let Err(e) = file.write_all(b"RBLC") {
            return Err(format!("write failed: {}", e));
        }

        let version: u32 = 1;
        if let Err(e) = file.write_all(&version.to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }

        let rosenblatts = [&self.cat, &self.cheetah, &self.lion];
        for (i, r) in rosenblatts.iter().enumerate() {
            if let Err(e) = file.write_all(&seeds[i].to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
            if let Err(e) = file.write_all(&r.learning_rate.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
            if let Err(e) = file.write_all(&r.bias.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }

            let w_rows = r.weights.rows as u32;
            let w_cols = r.weights.cols as u32;
            if let Err(e) = file.write_all(&w_rows.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
            if let Err(e) = file.write_all(&w_cols.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
            for &value in &r.weights.data {
                if let Err(e) = file.write_all(&value.to_le_bytes()) {
                    return Err(format!("write failed: {}", e));
                }
            }
        }

        Ok(())
    }


    pub fn load(path: &str) -> Result<(RosenblattClassifier, [u64; 3]), String> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Failed to open {}: {}", path, e)),
        };

        let read_u32 = |file: &mut File| -> Result<u32, String> {
            let mut buf = [0u8; 4];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(format!("read failed: {}", e));
            }
            Ok(u32::from_le_bytes(buf))
        };
        let read_u64 = |file: &mut File| -> Result<u64, String> {
            let mut buf = [0u8; 8];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(format!("read failed: {}", e));
            }
            Ok(u64::from_le_bytes(buf))
        };
        let read_f32 = |file: &mut File| -> Result<f32, String> {
            let mut buf = [0u8; 4];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(format!("read failed: {}", e));
            }
            Ok(f32::from_le_bytes(buf))
        };

        let mut magic = [0u8; 4];
        if let Err(e) = file.read_exact(&mut magic) {
            return Err(format!("read failed: {}", e));
        }
        if &magic != b"RBLC" {
            return Err(format!("Not a RosenblattClassifier file: bad magic number {:?}", magic));
        }

        let version = read_u32(&mut file)?;
        if version != 1 {
            return Err(format!(
                "Unsupported RosenblattClassifier file version: {} (this build supports version 1)",
                version
            ));
        }

        let mut rosenblatts: Vec<Rosenblatt> = Vec::new();
        let mut seeds: Vec<u64> = Vec::new();

        for _ in 0..3 {
            let seed = read_u64(&mut file)?;
            let learning_rate = read_f32(&mut file)?;
            let bias = read_f32(&mut file)?;

            let w_rows = read_u32(&mut file)? as usize;
            let w_cols = read_u32(&mut file)? as usize;
            let mut w_data = Vec::with_capacity(w_rows * w_cols);
            for _ in 0..(w_rows * w_cols) {
                w_data.push(read_f32(&mut file)?);
            }
            let weights = Matrix::from_vec(w_data, w_rows, w_cols);

            seeds.push(seed);
            rosenblatts.push(Rosenblatt {
                learning_rate,
                weights,
                bias,
            });
        }

        let lion = rosenblatts.pop().unwrap();
        let cheetah = rosenblatts.pop().unwrap();
        let cat = rosenblatts.pop().unwrap();

        let seed_cheetah = seeds.pop().unwrap();
        let seed_lion = seeds.pop().unwrap();
        let seed_cat = seeds.pop().unwrap();

        Ok((
            RosenblattClassifier { cat, lion, cheetah },
            [seed_cat, seed_cheetah, seed_lion],
        ))
    }
}

#[repr(C)]
pub struct MatrixFFI{
    data: *mut f32,
    rows: usize,
    cols : usize,
}

#[unsafe(no_mangle)]
pub extern "C" fn create_rosenblatt_classifier(n_features : usize, learning_rate : f32, bias_cat : f32, bias_lion : f32, bias_cheetah : f32, seed : u64) -> * mut RosenblattClassifier {
    let classifier = Box::new(RosenblattClassifier::new(n_features, learning_rate, bias_cat, bias_lion, bias_cheetah, seed));
    eprintln!("train called");
    Box::into_raw(classifier)
}

#[unsafe(no_mangle)]
pub extern "C" fn release_rosenblatt_classifier(classifier: *mut RosenblattClassifier){
    unsafe {
        let _ = Box::from_raw(classifier);
    };
}

#[unsafe(no_mangle)]
pub extern "C" fn train_classifier_rosen(
    classifier: *mut RosenblattClassifier,
    x: *const f32,
    rows: usize,
    cols: usize,
    y: *const usize,
    y_len: usize,
    epochs: usize,
) {
    unsafe {
        let classifier: &mut RosenblattClassifier = match classifier.as_mut() {
            Some(c) => c,
            None => return,
        };
        let x_slice = std::slice::from_raw_parts(x, rows * cols);
        let y_slice = std::slice::from_raw_parts(y, y_len);

        let x_matrix = Matrix::from_vec(x_slice.to_vec(), rows, cols);
        let y_vec = y_slice.to_vec();

        classifier.train(&x_matrix, &y_vec, epochs);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn predict_classifier(classifier: *mut RosenblattClassifier,
                                 x: *const f32,
                                 rows: usize,
                                 cols: usize,
                                 y: *const usize,
                                 y_len: usize
)-> *mut usize{
    unsafe{
        let classifier: &mut RosenblattClassifier = match classifier.as_mut() {
            Some(c) => c,
            None => return std::ptr::null_mut(),
        };

        let x_slice = std::slice::from_raw_parts(x, rows*cols);
        let y_slice = std::slice::from_raw_parts(y, y_len);

        let x_matrix = Matrix::from_vec(x_slice.to_vec(), rows, cols);
        let y_vec = y_slice.to_vec();

        let mut final_predict = classifier.predict(&x_matrix, &y_vec);
        final_predict.shrink_to_fit();
        final_predict.leak().as_mut_ptr()

    }
}

#[unsafe(no_mangle)]
pub extern "C" fn transforming(classifier: *mut RosenblattClassifier,
                        x: *const f32,
                        rows: usize,
                        cols: usize,
) -> *mut MatrixFFI{
    unsafe{
        let classifier: &mut RosenblattClassifier = match classifier.as_mut() {
            Some(c) => c,
            None => return std::ptr::null_mut(),
        };
        let x_vec:Vec<f32> = std::slice::from_raw_parts(x, rows*cols).to_vec();
        let x_matrix = Matrix::from_vec(x_vec, rows, cols);
        let x_transformed = classifier.transform(&x_matrix);

        let data = x_transformed.data.leak().as_mut_ptr();
        let rows = x_transformed.rows;
        let cols = x_transformed.cols;

        Box::into_raw(Box::new(MatrixFFI{data, rows,cols}))
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn release_matrix(matrix: *mut Matrix){
    unsafe {
        let _ = Box::from_raw(matrix);
    }
}


#[unsafe(no_mangle)]
pub extern "C" fn get_weights_classifier(
    classifier: *mut RosenblattClassifier,
) -> *mut f32 {
    unsafe {
        let classifier = match classifier.as_mut() {
            Some(c) => c,
            None => return std::ptr::null_mut(),
        };

        let mut all_weights = classifier.get_all_weight_vec();
        all_weights.shrink_to_fit();
        all_weights.leak().as_mut_ptr()
    }
}

#[repr(C)]
pub struct RosenblattParams {
    pub bias: f32,
    pub learning_rate: f32,
}

#[no_mangle]
pub extern "C" fn rosenblatt_load(
    path: *const c_char,
    out_seeds: *mut u64,
    out_params: *mut RosenblattParams,
    out_rows: *mut usize,
    out_cols: *mut usize,
    out_cat_weights: *mut f32,
    out_lion_weights: *mut f32,
    out_cheetah_weights: *mut f32,
) -> *mut RosenblattClassifier {
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };

    match RosenblattClassifier::load(path) {
        Ok((classifier, seeds)) => {
            if !out_seeds.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(seeds.as_ptr(), out_seeds, 3);
                }
            }

            if !out_params.is_null() {
                let params = RosenblattParams {
                    bias: classifier.cat.bias,
                    learning_rate: classifier.cat.learning_rate,
                };
                unsafe {
                    std::ptr::write(out_params, params);
                }
            }

            if !out_rows.is_null() {
                unsafe { *out_rows = classifier.cat.weights.rows; }
            }
            if !out_cols.is_null() {
                unsafe { *out_cols = classifier.cat.weights.cols; }
            }

            if !out_cat_weights.is_null() {
                let data = &classifier.cat.weights.data;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), out_cat_weights, data.len());
                }
            }
            if !out_lion_weights.is_null() {
                let data = &classifier.lion.weights.data;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), out_lion_weights, data.len());
                }
            }
            if !out_cheetah_weights.is_null() {
                let data = &classifier.cheetah.weights.data;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), out_cheetah_weights, data.len());
                }
            }

            Box::into_raw(Box::new(classifier))
        }
        Err(_) => std::ptr::null_mut(),
    }
}


#[no_mangle]
pub extern "C" fn rosenblatt_save(
    classifier: *const RosenblattClassifier,
    path: *const c_char,
    seeds: *const u64,
) -> i32 {
    if classifier.is_null() || path.is_null() || seeds.is_null() {
        return -1;
    }

    let classifier = unsafe { &*classifier };
    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(p) => p,
        Err(_) => return -1,
    };

    let seeds_arr: [u64; 3] = unsafe { [*seeds, *seeds.add(1), *seeds.add(2)] };

    match classifier.save(path, seeds_arr) {
        Ok(()) => 0,
        Err(_) => -1,
    }
}