use image::buffer::Rows;
use serde::forward_to_deserialize_any;
use crate::tensor::Matrix;
use super::Regression;

impl Regression{
    pub fn new(nb_features : usize) -> Self{
        Self {
            weights: Matrix::zeros(1, nb_features + 1),
        }
    }

    pub fn get_weight(&mut self)->&Matrix{
        &self.weights
    }

    pub fn fit(&mut self,x: &Matrix, y : &Matrix, bias : f32){
        let x_with_bias = self.add_bias_column(&x, bias);
        let transpose = x_with_bias.transpose();
        let w = transpose.dot(&x_with_bias).inverse().dot(&transpose).dot(y);
        self.weights = w;
    }

    pub fn add_bias_column(&mut self,x : &Matrix, bias : f32) -> Matrix{
        let mut flat_x = Vec::new();

        for rows in 0..x.rows {
            flat_x.push(bias);
            for cols in 0..x.cols{
                flat_x.push(x.get(rows, cols));
            }
        }
        Matrix::from_vec(flat_x, x.rows, x.cols +1)
    }

    pub fn predict(&mut self, x : &Matrix, bias :f32) -> Matrix{
        let weight = self.weights.clone();
        let x_with_bias = self.add_bias_column(&x,bias);
        let mut result = Vec::new();

        for position in 0..x.rows{
            let get_xk = Self::get_xk_regression(&x_with_bias, position);
            let somme = Self::somme_regression(&get_xk, &weight);
            result.push(somme);
        }
        Matrix::from_vec(result,x_with_bias.rows,1)
    }

    pub fn somme_regression(x: &Matrix, weight: &Matrix)->f32{
        let mut somme:f32 = 0.0;
        for i in 0..x.cols{
            somme += x.get(0,i) * weight.get(0,i);
        }
        somme
    }

    pub fn get_xk_regression(x_with_bias: &Matrix, position: usize) -> Matrix{
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
}
#[repr(C)]
pub struct MatrixFFIRegression{
    data : *mut f32,
    rows: usize,
    cols : usize
}

#[unsafe(no_mangle)]
pub extern "C" fn create_regression(
    nb_feature : usize
)->*mut Regression{
    unsafe {
        let regression = Box::new(Regression::new(nb_feature));
        Box::into_raw(regression)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn fit_regression(regression: *mut Regression,
                                 x : *const f32,
                                 x_rows: usize,
                                 x_cols: usize,
                                 y: *const f32,
                                 y_len: usize,
                                 bias : f32,
){
    unsafe{
        let regression = unsafe {
            match regression.as_mut() {
                Some(c) =>c,
                None => return,
            }
        };
        let x_slice = std::slice::from_raw_parts(x, x_rows * x_cols);
        let x_matrix = Matrix::from_vec(x_slice.to_vec(), x_rows, x_cols);
        let y_slice = std::slice::from_raw_parts(y, y_len);
        let y_matrix = Matrix::from_vec(y_slice.to_vec(), y_len, 1);

        regression.fit(&x_matrix, &y_matrix, bias);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn predict_regression(regression: *mut Regression,
                                     x : *const f32,
                                     x_rows : usize,
                                     x_cols : usize,
                                     bias: f32,
) -> *mut MatrixFFIRegression{
    unsafe {
        let regression = unsafe{
            match regression.as_mut(){
                Some(c) => c,
                None => return std::ptr::null_mut(),
            }
        };
        let x_slice = std::slice::from_raw_parts(x, x_cols * x_rows);
        let x_matrix = Matrix::from_vec(x_slice.to_vec(), x_rows, x_cols);

        let predict = regression.predict(&x_matrix, bias);

        let data = predict.data.leak().as_mut_ptr();
        let rows = predict.rows;
        let cols = predict.cols;

        Box::into_raw(Box::new(MatrixFFIRegression{data, rows, cols}))
    }
}