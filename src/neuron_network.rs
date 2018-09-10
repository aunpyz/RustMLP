// fully connected neuron network
use super::neuron::{Neuron, NeuronType::*};
use super::rand::{thread_rng, Rng};

#[derive(Debug)]
pub struct NeuronNetwork {
    input: Vec<Neuron>,
    hidden_layer: Vec<Vec<Neuron>>,
    output: Vec<Neuron>,
}

impl NeuronNetwork {
    // input is number of input(s)
    // hidden_layer is number of neurons in eachhidden layer
    // output is number of output(s)
    pub fn new(input: usize, mut hidden_layer: Vec<usize>, output: usize) -> NeuronNetwork {
        let layers = hidden_layer.len();

        // last one for output neuron connections
        hidden_layer.push(output);

        let mut input_neuron: Vec<Neuron> = Vec::with_capacity(input);
        let mut hidden_layer_neuron: Vec<Vec<Neuron>> = Vec::with_capacity(layers);
        let mut output_neuron: Vec<Neuron> = Vec::with_capacity(output);

        for _i in 0..input {
            input_neuron.push(Neuron::new(hidden_layer[0], Input));
        }
        for i in 0..layers {
            let mut hidden_neuron: Vec<Neuron> = Vec::with_capacity(hidden_layer[i]);
            for _j in 0..hidden_layer[i] {
                hidden_neuron.push(Neuron::new(hidden_layer[i + 1], HiddenLayer));
            }
            hidden_layer_neuron.push(hidden_neuron);
        }
        for _i in 0..output {
            output_neuron.push(Neuron::new(0, Output));
        }

        NeuronNetwork {
            input: input_neuron,
            hidden_layer: hidden_layer_neuron,
            output: output_neuron,
        }
    }
}
