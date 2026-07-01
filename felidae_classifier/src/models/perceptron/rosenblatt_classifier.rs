use crate::models::perceptron::{Rosenblatt, RosenblattClassifier};
use crate::tensor::Matrix;

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
        let y_lion = Rosenblatt::to_binary_labels(y, 1);
        let y_cheetah = Rosenblatt::to_binary_labels(y, 2);

        self.cat.train(x, &y_cat, epochs);
        self.lion.train(x,&y_lion, epochs);
        self.cheetah.train(x,&y_cheetah, epochs);
    }

    pub fn predict(&mut self, x : &Matrix, y : &Vec<usize>) -> Vec<usize>{
        let cat_predict = self.cat.predict(&x);
        let lion_predict = self.lion.predict(&x);
        let cheetah_predict = self.cheetah.predict(&x);

        let tab_argmax:Vec<usize> = Rosenblatt::argmax(&cat_predict, &lion_predict, &cheetah_predict);
        Rosenblatt::print_prediction_result(&tab_argmax, y);
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
pub extern "C" fn train_classifier(
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
