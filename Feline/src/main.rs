mod models;
mod training;
mod common;
mod load_data;

use std::alloc::System;
use std::fs;
use std::process;
use crate::common::transform_data;
use crate::load_data::images_data::{pipeline};

fn main() {
    pipeline();
}
