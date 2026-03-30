pub mod train_linear;
pub use train_linear::extract_csv;
use std::{error::Error, io , process};
use csv::ReaderBuilder;


pub struct DataSet{
    header : Vec<String>,
    data : Option<Vec<Vec<i32>>>,
}