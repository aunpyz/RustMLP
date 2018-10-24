extern crate mlp;

use mlp::data_ops;
use mlp::neural_network::cross_validation;
use std::fs::File;
use std::io::BufReader;

fn main() {
    const E: f64 = 1e-7;

    // text files are not available in version control
    {
        let filename = "./data/Flood_dataset.txt";
        println!("In file {}", filename);

        let f = File::open(filename).expect("file not found");
        let f = BufReader::new(f);

        let (input_data, all_data) = data_ops::remove_line(f, 2, false, 0, 1);

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
        println!("{} Done", filename);
    }

    {
        let filename = "./data/cross.pat";
        println!("In file {}", filename);

        let f = File::open(filename).expect("file not found");
        let f = BufReader::new(f);

        let (input_data, _all_data) = data_ops::remove_line(f, 1, true, 2, 2);
        let input_data = data_ops::shuffle(input_data);

        let normalized_data = mlp::MinMax::new(input_data, None);
        let hidden_layers = vec![4, 3];
        cross_validation(
            (2, hidden_layers, 2),
            normalized_data,
            10,
            500,
            E,
            String::from("cross_cross.txt"),
            false,
        );
        println!("{} Done", filename);
    }
}
