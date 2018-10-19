// fully connected neuron network
use super::neuron::Neuron;
use super::rand::{thread_rng, Rng};

use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};

mod function;

#[derive(Debug, Clone)]
pub struct NeuronNetwork {
    input: Vec<Neuron>,
    hidden_layer: Vec<Vec<Neuron>>,
    output: Vec<Neuron>,
    learning_rate: f64,
    momentum_rate: f64,
}

impl fmt::Display for NeuronNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut display = String::new();
        display.push_str(&format!(
            "Neuron Network\n\
             Learning rate: {}, Momentum rate: {}\n\
             Input: {}\n\
             ",
            self.learning_rate,
            self.momentum_rate,
            self.input.len()
        ));
        for i in 0..self.input.len() {
            display.push_str(&format!(
                "{}\n\
                 ",
                self.input[i].to_string()
            ));
        }
        display.push_str(&format!(
            "Hidden layer: {}\n\
             ",
            self.hidden_layer.len()
        ));
        for i in 0..self.hidden_layer.len() {
            display.push_str(&format!(
                "\t{}\n\
                 ",
                i + 1
            ));
            for j in 0..self.hidden_layer[i].len() {
                display.push_str(&format!(
                    "{}\n\
                     ",
                    self.hidden_layer[i][j].to_string()
                ));
            }
        }
        display.push_str(&format!(
            "Output: {}\n\
             ",
            self.output.len()
        ));
        for i in 0..self.output.len() {
            display.push_str(&format!(
                "{}
                ",
                self.output[i].to_string()
            ));
        }
        // remove newline
        display = display[0..display.len() - 1].to_string();
        write!(f, "{}", display)
    }
}

impl NeuronNetwork {
    // input is number of input(s)
    // hidden_layer is number of neurons in each hidden layer
    // output is number of output(s)
    fn new(input: usize, mut hidden_layer: Vec<usize>, output: usize) -> Self {
        let mut rng = thread_rng();
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
            learning_rate: rng.gen_range(0.1, 1_f64),
            momentum_rate: rng.gen_range(0.1, 1_f64),
        }
    }

    fn empty(input: usize, mut hidden_layer: Vec<usize>, output: usize) -> Self {
        let layers = hidden_layer.len();
        hidden_layer.push(output);

        let mut input_neuron: Vec<Neuron> = Vec::with_capacity(input);
        let mut hidden_layer_neuron: Vec<Vec<Neuron>> = Vec::with_capacity(layers);
        let mut output_neuron: Vec<Neuron> = Vec::with_capacity(output);

        for _i in 0..input {
            input_neuron.push(Neuron::empty(hidden_layer[0]));
        }
        for i in 0..layers {
            let mut hidden_neuron: Vec<Neuron> = Vec::with_capacity(hidden_layer[i]);
            for _j in 0..hidden_layer[i] {
                hidden_neuron.push(Neuron::empty(hidden_layer[i + 1]));
            }
            hidden_layer_neuron.push(hidden_neuron);
        }
        for _i in 0..output {
            output_neuron.push(Neuron::empty(0));
        }

        NeuronNetwork {
            input: input_neuron,
            hidden_layer: hidden_layer_neuron,
            output: output_neuron,
            learning_rate: 0_f64,
            momentum_rate: 0_f64,
        }
    }

    fn forward_pass(
        &self,
        item: &Vec<f64>,
        (input, hidden_layer, output): (usize, usize, usize),
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        // iterate through line of normalized raw data
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
                            * function::sigmoid(output_nodes[layer][n]);
                    }
                }

                output_nodes.push(output);
                // go to next layer
                layer += 1;
            }
        }

        {
            // output layer
            let output_layer_node = output_nodes.last().unwrap();
            for k in 0..output {
                let error = item[input + k] - function::sigmoid(output_layer_node[k]);
                errors.push(error);
            }
        }
        (output_nodes, errors)
    }

    fn backward_pass(
        &mut self,
        output_nodes: Vec<Vec<f64>>,
        errors: Vec<f64>,
        prev_neuron_netowrk: NeuronNetwork,
        input: Vec<f64>,
    ) -> Self {
        let prev_nn = self.clone();
        let mut gradients: Vec<Vec<f64>> = Vec::new();
        for (i, output) in output_nodes.iter().rev().enumerate() {
            let mut gradient: Vec<f64> = Vec::new();
            if i == 0 {
                // output
                for j in 0..output.len() {
                    gradient.push(errors[j] * function::d_sigmoid(output[j]));
                }
            } else {
                // hidden layer
                for j in 0..output.len() {
                    let mut sum_gradient: f64 = 0_f64;
                    for k in 0..gradients[i - 1].len() {
                        sum_gradient += gradients[i - 1][k];
                    }
                    let g = function::d_sigmoid(output[j]) * sum_gradient;
                    assert!(
                        g.is_finite(),
                        "output {} : {},
                        sum_gradient : {}",
                        j,
                        output[j],
                        sum_gradient
                    );
                    gradient.push(g);
                }
            }
            gradients.push(gradient);
        }

        // adjust weight
        {
            let mut layer = 0;
            gradients.reverse();
            {
                // input
                let input_next_layer = self.input[0].next;
                for i in 0..input_next_layer {
                    let bias = self.hidden_layer[0][i].weight[0];
                    let d = self.momentum_rate
                        * (bias - prev_neuron_netowrk.hidden_layer[0][i].weight[0])
                        + self.learning_rate * gradients[layer][i] * 1_f64;
                    assert!(
                        d.is_finite(),
                        "output weight : {}, 
                    t-1 output weight : {}, 
                    learning rate : {}, 
                    gradients : {}, 
                    momentum rate : {}, 
                    input : {}",
                        bias,
                        prev_neuron_netowrk.hidden_layer[0][i].weight[0],
                        self.learning_rate,
                        gradients[layer][i],
                        self.momentum_rate,
                        1_f64
                    );
                    self.hidden_layer[0][i].weight[0] = bias + d;
                }
                for i in 0..input_next_layer {
                    for j in 0..input.len() {
                        let weight = self.input[j].weight[i + 1];
                        let d = self.momentum_rate
                            * (weight - prev_neuron_netowrk.input[j].weight[i + 1])
                            + self.learning_rate * gradients[layer][i] * input[j];
                        assert!(
                            d.is_finite(),
                            "output weight : {},
                         t-1 output weight : {}, 
                         learning rate : {}, 
                         gradients : {}, 
                         momentum rate : {}, 
                         input : {}",
                            weight,
                            prev_neuron_netowrk.input[j].weight[i + 1],
                            self.learning_rate,
                            gradients[layer][i],
                            self.momentum_rate,
                            input[j]
                        );
                        self.input[j].weight[i + 1] = weight + d;
                    }
                }
                layer += 1;
            }
            {
                // hidden layer
                let hidden_layer = self.hidden_layer.len();
                for i in 0..hidden_layer {
                    let input_next_layer = self.hidden_layer[i][0].next;

                    if i + 1 >= hidden_layer {
                        // node(s) connected to output node(s)
                        for j in 0..input_next_layer {
                            let bias = self.output[j].weight[0];
                            let d = self.momentum_rate
                                * (bias - prev_neuron_netowrk.output[j].weight[0])
                                + self.learning_rate * gradients[layer][j] * 1_f64;
                            assert!(d.is_finite());
                            self.output[j].weight[0] = bias + d;
                        }
                        for j in 0..input_next_layer {
                            for k in 0..self.hidden_layer[i].len() {
                                let weight = self.hidden_layer[i][k].weight[j + 1];
                                let d = self.momentum_rate
                                    * (weight
                                        - prev_neuron_netowrk.hidden_layer[i][k].weight[j + 1])
                                    + self.learning_rate
                                        * gradients[layer][j]
                                        * function::sigmoid(output_nodes[layer - 1][k]);
                                assert!(d.is_finite());
                                self.hidden_layer[i][k].weight[j + 1] = weight + d;
                            }
                        }
                    } else {
                        for j in 0..input_next_layer {
                            let bias = self.hidden_layer[i + 1][j].weight[0];
                            let d = self.momentum_rate
                                * (bias - prev_neuron_netowrk.hidden_layer[i + 1][j].weight[0])
                                + self.learning_rate * gradients[layer][j] * 1_f64;
                            assert!(d.is_finite());
                            self.hidden_layer[i + 1][j].weight[0] = bias + d;
                        }
                        for j in 0..input_next_layer {
                            for k in 0..self.hidden_layer[i].len() {
                                let weight = self.hidden_layer[i][k].weight[j + 1];
                                let d = self.momentum_rate
                                    * (weight
                                        - prev_neuron_netowrk.hidden_layer[i][k].weight[j + 1])
                                    + self.learning_rate
                                        * gradients[layer][j]
                                        * function::sigmoid(output_nodes[layer - 1][k]);
                                assert!(d.is_finite());
                                self.hidden_layer[i][k].weight[j + 1] = weight + d;
                            }
                        }
                    }
                    layer += 1;
                }
            }
        }

        // send previous NeuronNetwork back
        prev_nn
    }
}

pub fn cross_validation(
    (input, hidden_layer, output): (usize, Vec<usize>, usize),
    file: BufReader<File>,
    validate_section: usize,
    epoch: usize,
    stop_treshhold: f64,
) {
    let mut nn = NeuronNetwork::new(input, hidden_layer.clone(), output);

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
    // normalized_data will no longer available
    let n = section.len();
    for i in 0..n {
        println!("{}", nn);
        let mut prev_nn = NeuronNetwork::empty(input, hidden_layer.clone(), output);
        for iter in 0..epoch {
            let mut error: f64 = 1_f64;
            for j in 0..n {
                // ignore index i
                if j == i {
                    continue;
                }
                let data = &section[j];
                for (_index, item) in data.iter().enumerate() {
                    let (output_nodes, errors) =
                        nn.forward_pass(item, (input, hidden_layer.len(), output));
                    prev_nn = nn.backward_pass(
                        output_nodes,
                        errors.clone(),
                        prev_nn,
                        item[0..input].to_vec(),
                    );
                    let sum_sqrt_err = function::sum_sqrt_err(errors);
                    // println!("{}", nn);
                    // println!("Sum square error : {}", sum_sqrt_err);
                    error = sum_sqrt_err;
                }
            }
            // println!("{}", nn);
            println!("Sum square error: {}", error);
            if error <= stop_treshhold {
                println!("stop at : {}", iter);
                break;
            }
        }
    }
}
