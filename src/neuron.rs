use super::rand::{thread_rng, Rng};

use std::fmt;

#[derive(Debug, Clone)]
pub struct Neuron {
    pub next: usize,
    pub weight: Vec<f64>,
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut display = String::new();
        display.push_str(&format!(
            "-----*Neuron*-----\n\
             Next: {}\n\
             Bias: {}\n",
            self.next, self.weight[0]
        ));
        if self.weight.len() > 1 {
            display.push_str("Weight to next nodes:\n");
            for i in 1..self.weight.len() {
                display.push_str(&format!("{}, ", self.weight[i]));
            }
            // remove ", "
            display = display[0..display.len() - 2].to_string();
        }
        display.push_str("\n------------------\n");
        write!(f, "{}", display)
    }
}

impl Neuron {
    pub fn new(next: usize) -> Self {
        let cap = next + 1;
        let mut weight_init = Vec::with_capacity(cap);
        // weight initialization range
        let weight_range: f64 = 1.0_f64 / (if next > 0 { next as f64 } else { f64::from(1) }).sqrt();
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
            next,
            weight: weight_init,
        }
    }

    pub fn empty(next: usize) -> Self {
        let cap = next + 1;
        let mut weight_init = Vec::with_capacity(cap);

        for _i in 0..cap {
            weight_init.push(0_f64);
        }

        Neuron {
            next,
            weight: weight_init,
        }
    }
}
