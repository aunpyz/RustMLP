extern crate mlp;

use mlp::data_ops;
use mlp::neural_network::*;
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
    const E: f64 = 1e-7;

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

    let mut input_data = Vec::<Vec<f64>>::new();
    let mut all_data = Vec::<f64>::new();
    for line in f.lines() {
        let line = line.unwrap();
        let split = line.split_whitespace().collect::<Vec<&str>>();
        let mut vec = data_ops::to_f64_vec(split);
        // vec clone will be consumed after push
        input_data.push(vec.clone());
        all_data.append(&mut vec);
    }
    let normalized_data = data_ops::normalize(all_data, input_data);

    let hidden_layers = vec![4, 3];
    cross_validation(
        (8, hidden_layers, 1),
        normalized_data,
        10,
        500,
        E,
        String::from("flood_cross.txt"),
        true,
    );
}

fn print_split(split: std::str::SplitWhitespace) {
    use std::str::SplitWhitespace;
    let clone_split = SplitWhitespace::clone(&split);
    println!("{} elements", SplitWhitespace::count(clone_split));
    for s in split {
        println!("{}", s);
    }
}
