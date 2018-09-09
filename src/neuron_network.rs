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
    // hidden_layer is number of layer(s) in hidden layer
    // output is number of output(s)
    pub fn new(input: usize, hidden_layer: usize, output: usize) -> NeuronNetwork {
        // number of hidden neuron per layer
        let mut hidden_neuron_size: Vec<usize> = Vec::with_capacity(hidden_layer + 1);
        let mut rng = thread_rng();
        for _i in 0..hidden_layer {
            // fixed size for now
            // max inclusive randomness
            hidden_neuron_size.push(rng.gen_range(2, 6));
        }
        // last one for output neuron connections
        hidden_neuron_size.push(output);

        let mut input_neuron: Vec<Neuron> = Vec::with_capacity(input);
        let mut hidden_layer_neuron: Vec<Vec<Neuron>> = Vec::with_capacity(hidden_layer);
        let mut output_neuron: Vec<Neuron> = Vec::with_capacity(output);

        for _i in 0..input {
            input_neuron.push(Neuron::new(hidden_neuron_size[0], Input));
        }
        for i in 0..hidden_layer {
            let mut hidden_neuron: Vec<Neuron> = Vec::with_capacity(hidden_neuron_size[i]);
            for _j in 0..hidden_neuron_size[i] {
                hidden_neuron.push(Neuron::new(hidden_neuron_size[i + 1], HiddenLayer));
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
