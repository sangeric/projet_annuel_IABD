use rand::RngExt;
use crate::tensor::Matrix;

use rand::SeedableRng;
use rand::rngs::StdRng;
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
        let mut x_with_bias = self.add_bias_column(&x, bias);
        let mut transpose = x_with_bias.transpose();
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
        let mut weight = self.weights.clone();
        let mut x_with_bias = self.add_bias_column(&x,bias);
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