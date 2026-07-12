// src/ffi.rs

use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

use crate::models::mlp::MLP;
use crate::models::rbfn::RBFN;
use crate::tensor::Matrix;

// MLP
#[no_mangle]
pub extern "C" fn mlp_create(
    layer_sizes_ptr: *const u32,
    n_layers: usize,
    activation: *const c_char,
    output_activation: *const c_char,
) -> *mut MLP {
    let sizes: Vec<usize> = unsafe {
        let slice = std::slice::from_raw_parts(layer_sizes_ptr, n_layers);
        let mut v = Vec::new();
        for i in 0..slice.len() {
            v.push(slice[i] as usize);
        }
        v
    };

    let act = unsafe { CStr::from_ptr(activation).to_str().unwrap_or("tanh") };
    let out_act = unsafe { CStr::from_ptr(output_activation).to_str().unwrap_or("tanh") };

    match MLP::new(&sizes, act, out_act) {
        Ok(mlp) => Box::into_raw(Box::new(mlp)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn mlp_train(
    mlp: *mut MLP,
    x_data: *const f32,
    x_rows: usize,
    x_cols: usize,
    y_data: *const f32,
    y_rows: usize,
    y_cols: usize,
    epochs: usize,
    learning_rate: f32,
    log_dir: *const c_char,
) {
    let mlp = unsafe { &mut *mlp };

    let log_dir = unsafe {
        if log_dir.is_null() {
            "./runs/logdir".to_string()
        } else {
            CStr::from_ptr(log_dir).to_str().unwrap_or("./runs/logdir").to_string()
        }
    };

    let x_vec = unsafe { std::slice::from_raw_parts(x_data, x_rows * x_cols).to_vec() };
    let y_vec = unsafe { std::slice::from_raw_parts(y_data, y_rows * y_cols).to_vec() };

    let x = Matrix::from_vec(x_vec, x_rows, x_cols);
    let y = Matrix::from_vec(y_vec, y_rows, y_cols);

    // No test split from FFI — pass same data for train and test
    mlp.train(&x, &y, &x, &y, epochs, learning_rate, &log_dir);
}

#[no_mangle]
pub extern "C" fn mlp_predict_classes(
    mlp: *const MLP,
    x_data: *const f32,
    x_rows: usize,
    x_cols: usize,
    out_predictions: *mut u32,
) {
    let mlp = unsafe { &*mlp };

    let x_vec = unsafe { std::slice::from_raw_parts(x_data, x_rows * x_cols).to_vec() };
    let x = Matrix::from_vec(x_vec, x_rows, x_cols);

    let preds = mlp.predict(&x);

    unsafe {
        for i in 0..preds.len() {
            *out_predictions.add(i) = preds[i] as u32;
        }
    }
}

#[no_mangle]
pub extern "C" fn mlp_predict_raw(
    mlp: *const MLP,
    x_data: *const f32,
    x_rows: usize,
    x_cols: usize,
    out_predictions: *mut f32,
) {
    let mlp = unsafe { &*mlp };

    let x_vec = unsafe { std::slice::from_raw_parts(x_data, x_rows * x_cols).to_vec() };
    let x = Matrix::from_vec(x_vec, x_rows, x_cols);

    let result = mlp.forward(&x);

    unsafe {
        for i in 0..result.data.len() {
            *out_predictions.add(i) = result.data[i];
        }
    }
}

#[no_mangle]
pub extern "C" fn mlp_save(mlp: *const MLP, path: *const c_char) -> c_int {
    let mlp = unsafe { &*mlp };
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };

    match mlp.save(path) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn mlp_load(path: *const c_char) -> *mut MLP {
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };

    match MLP::load(path) {
        Ok(mlp) => Box::into_raw(Box::new(mlp)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn mlp_destroy(mlp: *mut MLP) {
    if !mlp.is_null() {
        unsafe { drop(Box::from_raw(mlp)) };
    }
}

#[no_mangle]
pub extern "C" fn extract_features(
    image_data: *const f32,
    out: *mut f32,
    out_len: usize,
) {
    use crate::features::extract::extract;
    use crate::data::image::{LoadedImage, IMAGE_SIZE};

    let n = (IMAGE_SIZE * IMAGE_SIZE * 3) as usize;
    let pixels = unsafe { std::slice::from_raw_parts(image_data, n).to_vec() };

    let image = LoadedImage {
        pixels,
        width: IMAGE_SIZE,
        height: IMAGE_SIZE,
    };

    let features = extract(&image);

    unsafe {
        for i in 0..features.len().min(out_len) {
            *out.add(i) = features[i];
        }
    }
}


// RBFN
#[no_mangle]
pub extern "C" fn rbfn_train(
    x_data: *const f32,
    x_rows: usize,
    x_cols: usize,
    y_data: *const f32,
    y_rows: usize,
    y_cols: usize,
    k: usize,
    gamma: f32,
    kmeans_iters: usize,
    seed: u64,
) -> *mut RBFN {
    let x_vec = unsafe { std::slice::from_raw_parts(x_data, x_rows * x_cols).to_vec() };
    let y_vec = unsafe { std::slice::from_raw_parts(y_data, y_rows * y_cols).to_vec() };

    let x = Matrix::from_vec(x_vec, x_rows, x_cols);
    let y = Matrix::from_vec(y_vec, y_rows, y_cols);

    let model = RBFN::train(&x, &y, k, gamma, kmeans_iters, seed);
    Box::into_raw(Box::new(model))
}

#[no_mangle]
pub extern "C" fn rbfn_train_naive(
    x_data: *const f32,
    x_rows: usize,
    x_cols: usize,
    y_data: *const f32,
    y_rows: usize,
    y_cols: usize,
    gamma: f32,
) -> *mut RBFN {
    let x_vec = unsafe { std::slice::from_raw_parts(x_data, x_rows * x_cols).to_vec() };
    let y_vec = unsafe { std::slice::from_raw_parts(y_data, y_rows * y_cols).to_vec() };
    let x = Matrix::from_vec(x_vec, x_rows, x_cols);
    let y = Matrix::from_vec(y_vec, y_rows, y_cols);
    let model = RBFN::train_naive(&x, &y, gamma);
    Box::into_raw(Box::new(model))
}

#[no_mangle]
pub extern "C" fn rbfn_predict_classes(
    model: *const RBFN,
    x_data: *const f32,
    x_rows: usize,
    x_cols: usize,
    out_predictions: *mut u32,
) {
    let model = unsafe { &*model };

    let x_vec = unsafe { std::slice::from_raw_parts(x_data, x_rows * x_cols).to_vec() };
    let x = Matrix::from_vec(x_vec, x_rows, x_cols);

    let preds = model.predict(&x);

    unsafe {
        for i in 0..preds.len() {
            *out_predictions.add(i) = preds[i] as u32;
        }
    }
}

#[no_mangle]
pub extern "C" fn rbfn_save(model: *const RBFN, path: *const c_char) -> c_int {
    let model = unsafe { &*model };
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    match model.save(path) {
        Ok(_) => 0,
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn rbfn_load(path: *const c_char) -> *mut RBFN {
    let path = unsafe { CStr::from_ptr(path).to_str().unwrap_or("") };
    match RBFN::load(path) {
        Ok(model) => Box::into_raw(Box::new(model)),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn rbfn_destroy(model: *mut RBFN) {
    if !model.is_null() {
        unsafe { drop(Box::from_raw(model)) };
    }
}
