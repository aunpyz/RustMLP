use self::NeuronType::Input;
use super::rand::{self, thread_rng, Rng};

pub enum NeuronType {
    Input,
    Output,
    HiddenLayer,
}

#[derive(Debug)]
pub struct Neuron {
    bias: f64,
    next: usize,
    weight: Vec<f64>,
}

impl Neuron {
    pub fn new(next: usize, neuron_type: NeuronType) -> Self {
        let mut weight_init = Vec::with_capacity(next);
        // weight initialization range
        let weight_range: f64 = 1.0_f64 / (next as f64).sqrt();
        let mut rng = thread_rng();
        for _i in 0..next {
            // assigning weight
            weight_init.push(rng.gen_range(-weight_range, weight_range));
        }
        Neuron {
            // not sure about bias initialization
            bias: match neuron_type {
                Input => 0 as f64,
                _ => rand::random::<f64>(),
            },
            next: next,
            weight: weight_init,
        }
    }
}
