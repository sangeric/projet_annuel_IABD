use std::io;
use crate::models::linear::test_linear;
use csv::{Reader, ReaderBuilder};
use crate::training::DataSet;
use std::{
    env,
    error::Error,
    ffi::OsStr,
    fs::File,
    process,
};


pub fn extract_csv() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = "src/data/data.csv";

    let mut reader = ReaderBuilder::new()
        .delimiter(b';')
        .from_path(file_path)?;
    
    let headers = reader.headers()?.clone();
    println!("Headers: {:?}", headers);
    
    for result in reader.records(){
        let record = result?;
        println!("{:?}", record);
    }

    Ok(())
}

