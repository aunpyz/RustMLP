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
