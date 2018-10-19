extern crate mlp;

use mlp::neuron_network::*;
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
    const e: f64 = 1e-7;

    // text file is not available in version control
    let filename = "./data/Flood_dataset.txt";
    println!("In file {}", filename);

    let f = File::open(filename).expect("file not found");
    let mut f = BufReader::new(f);

    // f.read_to_string(&mut contents)
    //     .expect("something went wrong reading the file");

    // println!("With text:\n{}", contents);
    // for line in f.lines(){
    //     println!("{}", line.unwrap());
    // }

    f = remove_line(f, 2);

    let hidden_layers = vec![3, 2];
    cross_validation((8, hidden_layers, 1), f, 10, 100, e);
}

fn print_split(split: std::str::SplitWhitespace) {
    use std::str::SplitWhitespace;
    let clone_split = SplitWhitespace::clone(&split);
    println!("{} elements", SplitWhitespace::count(clone_split));
    for s in split {
        println!("{}", s);
    }
}
