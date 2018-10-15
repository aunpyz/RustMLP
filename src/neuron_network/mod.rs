// fully connected neuron network
use super::neuron::Neuron;
use super::rand::{thread_rng, Rng};

use std::fs::File;
use std::io::{BufRead, BufReader};

mod function;

#[derive(Debug, Clone)]
pub struct NeuronNetwork {
    input: Vec<Neuron>,
    hidden_layer: Vec<Vec<Neuron>>,
    output: Vec<Neuron>,
}

impl NeuronNetwork {
    // input is number of input(s)
    // hidden_layer is number of neurons in each hidden layer
    // output is number of output(s)
    pub fn new(input: usize, mut hidden_layer: Vec<usize>, output: usize) -> NeuronNetwork {
        let layers = hidden_layer.len();

        // last one for output neuron connections
        hidden_layer.push(output);

        let mut input_neuron: Vec<Neuron> = Vec::with_capacity(input);
        let mut hidden_layer_neuron: Vec<Vec<Neuron>> = Vec::with_capacity(layers);
        let mut output_neuron: Vec<Neuron> = Vec::with_capacity(output);

        for _i in 0..input {
            input_neuron.push(Neuron::new(hidden_layer[0]));
        }
        for i in 0..layers {
            let mut hidden_neuron: Vec<Neuron> = Vec::with_capacity(hidden_layer[i]);
            for _j in 0..hidden_layer[i] {
                hidden_neuron.push(Neuron::new(hidden_layer[i + 1]));
            }
            hidden_layer_neuron.push(hidden_neuron);
        }
        for _i in 0..output {
            output_neuron.push(Neuron::new(0));
        }

        NeuronNetwork {
            input: input_neuron,
            hidden_layer: hidden_layer_neuron,
            output: output_neuron,
        }
    }

    fn forward_pass(
        &self,
        data_section: &Vec<Vec<Vec<f64>>>,
        (input, hidden_layer, output): (usize, usize, usize),
    ) {
        let n = data_section.len();
        for i in 0..n {
            for j in 0..n {
                // ignore index i
                if j == i {
                    continue;
                }
                let data = &data_section[j];
                // iterate through line of normalized raw data
                for (_index, item) in data.iter().enumerate() {
                    let mut layer = 0;

                    let mut errors: Vec<f64> = Vec::with_capacity(output);
                    // output for hidden layer nodes and output nodes
                    let mut output_nodes: Vec<Vec<f64>> = Vec::with_capacity(hidden_layer + 1);

                    {
                        // input layer feed to hidden layer
                        let next_layer_node = self.input[0].next;
                        let mut output: Vec<f64> = Vec::with_capacity(next_layer_node);

                        for k in 0..next_layer_node {
                            output.push(self.hidden_layer[0][k].weight[0]);
                        }
                        for k in 0..next_layer_node {
                            for n in 0..input {
                                output[k] += self.input[n].weight[k + 1] * item[n];
                            }
                        }
                        for k in 0..next_layer_node {
                            output[k] = function::sigmoid(output[k]);
                        }

                        output_nodes.push(output);
                    }

                    {
                        // hidden layers feed to output layer
                        for k in 0..hidden_layer {
                            let input_node = output_nodes[layer].len();
                            let next_layer_node = self.hidden_layer[k][0].next;
                            let mut output: Vec<f64> = Vec::with_capacity(next_layer_node);

                            if k + 1 >= hidden_layer {
                                // last hidden layer's layer connected to output layer
                                for l in 0..next_layer_node {
                                    output.push(self.output[l].weight[0]);
                                }
                            } else {
                                for l in 0..next_layer_node {
                                    output.push(self.hidden_layer[k + 1][l].weight[0]);
                                }
                            }
                            for l in 0..next_layer_node {
                                for n in 0..input_node {
                                    output[l] += self.hidden_layer[k][n].weight[l + 1]
                                        * output_nodes[layer][n];
                                }
                            }
                            for l in 0..next_layer_node {
                                output[l] = function::sigmoid(output[l]);
                            }

                            output_nodes.push(output);
                            // go to next layer
                            layer += 1;
                        }
                    }

                    {
                        // output layer
                        let output_layer_node = output_nodes.pop().unwrap();
                        for k in 0..output {
                            let error = item[input + k] - output_layer_node[k];
                            errors.push(error);
                        }
                    }
                    println!("{:?}", errors);
                }
            }
        }
    }
}

pub fn cross_validation(
    (input, hidden_layer, output): (usize, Vec<usize>, usize),
    file: BufReader<File>,
    validate_section: usize,
) {
    let n_hidden_layer = hidden_layer.len();
    let mut nn = NeuronNetwork::new(input, hidden_layer, output);

    let mut input_data = Vec::<Vec<f64>>::new();
    let mut all_data = Vec::<f64>::new();
    for line in file.lines() {
        let line = line.unwrap();
        let split = line.split_whitespace().collect::<Vec<&str>>();
        let mut vec = function::to_f64_vec(split);
        // vec clone will be consumed after push
        input_data.push(vec.clone());
        all_data.append(&mut vec);
    }
    let normalized_data = function::normalize(all_data, input_data);
    let (min, max) = (normalized_data.min, normalized_data.max);

    let section = function::split_section(normalized_data, validate_section);
    // let n = section.len();
    // normalized_data will no longer available

    nn.forward_pass(&section, (input, n_hidden_layer, output));

    // for i in 0..n {
    //     for j in 0..n {
    //         // ignore index i
    //         if j == i {
    //             continue;
    //         }
    //         let data = &section[j];
    //         // iterate through line of normalized raw data
    //         for (_index, item) in data.iter().enumerate() {
    //             let mut layer = 0;

    //             let mut errors: Vec<f64> = Vec::with_capacity(output);
    //             // output for hidden layer nodes and output nodes
    //             let mut output_nodes: Vec<Vec<f64>> = Vec::with_capacity(n_hidden_layer + 1);

    //             {
    //                 // input layer feed to hidden layer
    //                 let next_layer_node = nn.input[0].next;
    //                 let mut output: Vec<f64> = Vec::with_capacity(next_layer_node);

    //                 for k in 0..next_layer_node {
    //                     output.push(nn.hidden_layer[0][k].weight[0]);
    //                 }
    //                 for k in 0..next_layer_node {
    //                     for n in 0..input {
    //                         output[k] += nn.input[n].weight[k + 1] * item[n];
    //                     }
    //                 }
    //                 for k in 0..next_layer_node {
    //                     output[k] = function::sigmoid(output[k]);
    //                 }

    //                 output_nodes.push(output);
    //             }

    //             {
    //                 // hidden layers feed to output layer
    //                 for k in 0..n_hidden_layer {
    //                     let input_node = output_nodes[layer].len();
    //                     let next_layer_node = nn.hidden_layer[k][0].next;
    //                     let mut output: Vec<f64> = Vec::with_capacity(next_layer_node);

    //                     if k + 1 >= n_hidden_layer {
    //                         // last hidden layer's layer connected to output layer
    //                         for l in 0..next_layer_node {
    //                             output.push(nn.output[l].weight[0]);
    //                         }
    //                     } else {
    //                         for l in 0..next_layer_node {
    //                             output.push(nn.hidden_layer[k + 1][l].weight[0]);
    //                         }
    //                     }
    //                     for l in 0..next_layer_node {
    //                         for n in 0..input_node {
    //                             output[l] +=
    //                                 nn.hidden_layer[k][n].weight[l + 1] * output_nodes[layer][n];
    //                         }
    //                     }
    //                     for l in 0..next_layer_node {
    //                         output[l] = function::sigmoid(output[l]);
    //                     }

    //                     output_nodes.push(output);
    //                     // go to next layer
    //                     layer += 1;
    //                 }
    //             }

    //             {
    //                 // output layer
    //                 let output_layer_node = output_nodes.pop().unwrap();
    //                 for k in 0..output {
    //                     let error = item[input + k] - output_layer_node[k];
    //                     errors.push(error);
    //                 }
    //             }
    //         }
    //     }
    // }
}
