extern crate rand;

pub mod data_ops;
pub mod neural_network;
mod neuron;

#[derive(Debug)]
pub struct MinMax {
    pub min: f64,
    pub max: f64,
    pub f_data: Vec<Vec<f64>>,
}

impl MinMax {
    pub fn new(data: Vec<Vec<f64>>, min_max: Option<(f64, f64)>) -> Self {
        let (min, max) = min_max.unwrap_or((0_f64, 1_f64));
        MinMax {
            min,
            max,
            f_data: data,
        }
    }
}
