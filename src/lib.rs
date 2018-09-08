extern crate rand;

use NodeType::HiddenLayer;

pub enum NodeType {
    Input,
    Output,
    HiddenLayer,
}

pub struct Node {
    bias: f32,
    next: usize,
    weight: Vec<f32>,
}

impl Node {
    pub fn new(next: usize, node_type: NodeType) -> Self {
        Node {
            bias: match node_type {
                HiddenLayer => rand::random::<f32>(),
                _ => 0 as f32,
            },
            next: next,
            weight: Vec::with_capacity(next),
        }
    }
}
