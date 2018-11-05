// fully connected neuron network
use super::neuron::Neuron;
use super::rand::{thread_rng, Rng};
use data_ops::denormalize;
use MinMax;

use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

mod function;

#[derive(Debug, Clone)]
pub struct NeuralNetwork {
    input: Vec<Neuron>,
    hidden_layer: Vec<Vec<Neuron>>,
    output: Vec<Neuron>,
    learning_rate: f64,
    momentum_rate: f64,
}

impl fmt::Display for NeuralNetwork {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut display = String::new();
        display.push_str(&format!(
            "Neuron Network\n\
             Learning rate: {}, Momentum rate: {}\n\
             Input: {}\n",
            self.learning_rate,
            self.momentum_rate,
            self.input.len()
        ));
        for i in 0..self.input.len() {
            display.push_str(&format!("{}\n", self.input[i].to_string()));
        }
        display.push_str(&format!("Hidden layer: {}\n", self.hidden_layer.len()));
        for i in 0..self.hidden_layer.len() {
            display.push_str(&format!("{}\n", i + 1));
            for j in 0..self.hidden_layer[i].len() {
                display.push_str(&format!("{}\n", self.hidden_layer[i][j].to_string()));
            }
        }
        display.push_str(&format!("Output: {}\n", self.output.len()));
        for i in 0..self.output.len() {
            display.push_str(&format!("{}\n", self.output[i].to_string()));
        }
        // remove newline
        display = display[0..display.len() - 2].to_string();
        write!(f, "{}", display)
    }
}

impl NeuralNetwork {
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

        NeuralNetwork {
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

        NeuralNetwork {
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
        prev_neuron_netowrk: NeuralNetwork,
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
                    let hidden_level = self.hidden_layer.len() - i;
                    for k in 0..gradients[i - 1].len() {
                        sum_gradient +=
                            gradients[i - 1][k] * self.hidden_layer[hidden_level][j].weight[k + 1];
                    }
                    let g = function::d_sigmoid(output[j]) * sum_gradient;
                    assert!(
                        g.is_finite(),
                        "output {} : {},\n\
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
                        "output weight : {},\n\
                         t-1 output weight : {},\n\
                         learning rate : {},\n\
                         gradients : {},\n\
                         momentum rate : {},\n\
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
                            "output weight : {},\n\
                             t-1 output weight : {},\n\
                             learning rate : {},\n\
                             gradients : {},\n\
                             momentum rate : {},\n\
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
        if prev_nn.input[0].weight[0] <= 0_f64 {
            assert!(false, "{}\n{}", prev_neuron_netowrk, prev_nn);
        }
        // send previous NeuralNetwork back
        prev_nn
    }
}

pub fn cross_validation(
    (input, hidden_layer, output): (usize, Vec<usize>, usize),
    normalized_data: MinMax,
    validate_section: usize,
    epoch: usize,
    stop_treshhold: f64,
    out: String,
    need_denormalized: bool,
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>) {
    let path = format!("./out/{}", out);
    let path = Path::new(&path);
    let display = path.display();

    let mut out: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut d_out: Vec<Vec<Vec<f64>>> = Vec::new();

    let mut f = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why.description()),
        Ok(f) => f,
    };

    let (min, max) = (normalized_data.min, normalized_data.max);

    let total_data: f64 = normalized_data.f_data.len() as f64;
    let section = function::split_section(normalized_data, validate_section);
    // normalized_data will no longer available
    let n = section.len();
    let master_nn = NeuralNetwork::new(input, hidden_layer.clone(), output);
    let mut n_error: f64 = 0_f64;

    for i in 0..n {
        let mut nn = master_nn.clone();
        let mut prev_nn = NeuralNetwork::empty(input, hidden_layer.clone(), output);
        if let Err(why) = f.write_all(format!("\
            ============================================================================================\n\
            BEFORE\n{}\n", nn).as_bytes()){
            panic!("couldn't write to {}: {}", display, why.description());
        }
        let mut t = 0_f64;
        let mut error: f64 = 0_f64;
        for iter in 0..epoch {
            for j in 0..n {
                // ignore index i
                if j == i {
                    continue;
                }
                let data = &section[j];
                for item in data.iter() {
                    let (output_nodes, errors) =
                        nn.forward_pass(item, (input, hidden_layer.len(), output));
                    prev_nn = nn.backward_pass(
                        output_nodes,
                        errors.clone(),
                        prev_nn,
                        item[0..input].to_vec(),
                    );
                    let mean_sqrt_err = function::mean_sqrt_err(errors);
                    error += mean_sqrt_err;
                    t += 1_f64;
                }
            }
            if let Err(why) =
                f.write_all(format!("Average mean squared error: {}\n", error / t).as_bytes())
            {
                panic!("couldn't write to {}: {}", display, why.description());
            }
            if error / t <= stop_treshhold {
                if let Err(why) = f.write_all(format!("stop at : {}\n", iter).as_bytes()) {
                    panic!("couldn't write to {}: {}", display, why.description());
                }
                break;
            }
        }
        let data = &section[i];
        let mut tmp_out: Vec<Vec<f64>> = Vec::new();
        let mut tmp_d_out: Vec<Vec<f64>> = Vec::new();

        if let Err(why) = f.write_all(format!("\
            ============================================================================================\n\
            RESULT").as_bytes()){
            panic!("couldn't write to {}: {}", display, why.description());
        }

        let mut test_error: f64 = 0_f64; 
        for item in data.iter() {
            let (mut output, errors) = nn.forward_pass(item, (input, hidden_layer.len(), output));
            let output = {
                let output_v = output.pop().unwrap();
                let mut output_y: Vec<f64> = Vec::with_capacity(output_v.len());
                for item in output_v.iter() {
                    output_y.push(function::sigmoid(*item));
                }

                if need_denormalized {
                    denormalize(output_y, (min, max))
                } else {
                    output_y
                }
            };
            assert_eq!(output.len(), errors.len());
            let mean_sqrt_err = function::mean_sqrt_err(errors);
            test_error += mean_sqrt_err;
            if let Err(why) =
                f.write_all(format!("\nMean squared error : {}\n", mean_sqrt_err).as_bytes())
            {
                panic!("couldn't write to {}: {}", display, why.description());
            }

            let mut d: Vec<f64> = Vec::new();
            for i in 0..output.len() {
                let desire_output = item[input + i] * (max - min) + min;
                d.push(desire_output);
                if let Err(why) = f.write_all(
                    format!(
                        "desired output : {}\n\
                         output : {}\n\
                         error : {}\n",
                        desire_output,
                        output[i],
                        (desire_output - output[i]).abs()
                    ).as_bytes(),
                ) {
                    panic!("couldn't write to {}: {}", display, why.description());
                }
            }

            tmp_d_out.push(d);
            tmp_out.push(output);
        }

        n_error += test_error;
        d_out.push(tmp_d_out);
        out.push(tmp_out);

        if let Err(why) = f.write_all(format!("\n\
            Average mean squared error: {}\n\
            ============================================================================================\n\
            AFTER\n{}\n\n\n", test_error/data.len() as f64, nn).as_bytes()) {
            panic!("couldn't write to {}: {}", display, why.description());
        }
    }
    if let Err(why) =
        f.write_all(format!("\nAVERAGE MEAN SQUARED ERROR: {}\n", n_error / total_data).as_bytes())
    {
        panic!("couldn't write to {}: {}", display, why.description());
    }

    // for classification
    (out, d_out)
}
