use super::rand::{self, thread_rng, Rng};

#[derive(Debug, Clone)]
pub struct Neuron {
    next: usize,
    weight: Vec<f64>,
}

impl Neuron {
    pub fn new(next: usize) -> Self {
        let cap = next + 1;
        let mut weight_init = Vec::with_capacity(cap);
        // weight initialization range
        let weight_range: f64 = 1.0_f64 / (if next > 0 { next as f64 } else { 1 as f64 }).sqrt();
        let mut rng = thread_rng();
        for i in 0..cap {
            // assigning weight
            if i == 0 {
                weight_init.push(1.0_f64);
            } else {
                weight_init.push(rng.gen_range(-weight_range, weight_range));
            }
        }
        Neuron {
            // bias as a part of weights
            next: next,
            weight: weight_init,
        }
    }
}
