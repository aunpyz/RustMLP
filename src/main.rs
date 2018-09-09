extern crate mlp;

use mlp::neuron_network::NeuronNetwork;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};

fn remove_line(mut f: BufReader<File>, n: u8) -> BufReader<File> {
    let mut str = String::new();
    for _i in 0..n {
        f.read_line(&mut str).expect("can't remove line");
    }
    f
}

fn main() {
    // text file is not available in version control
    let filename = "./data/Flood_dataset.txt";
    println!("In file {}", filename);

    let f = File::open(filename).expect("file not found");
    let mut f = BufReader::new(f);

    let mut contents = String::new();
    // f.read_to_string(&mut contents)
    //     .expect("something went wrong reading the file");

    // println!("With text:\n{}", contents);
    // for line in f.lines(){
    //     println!("{}", line.unwrap());
    // }

    f = remove_line(f, 2);

    f.read_line(&mut contents).expect("file to read line");
    println!("{}", contents);
    let split = contents.split_whitespace();
    let nn = NeuronNetwork::new(2, 3, 1);
    println!("{:?}", nn);
}
