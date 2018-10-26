use MinMax;

use rand::{self, Rng};
use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

// not flexible fn at all
pub fn confusion_matrix(
    (out, desire_output): (Vec<Vec<Vec<f64>>>, Vec<Vec<Vec<f64>>>),
    out_filename: String,
) {
    // only works for 2 outputs
    assert_eq!(out[0][0].len(), 2);

    let path = format!("./out/{}", out_filename);
    let path = Path::new(&path);
    let display = path.display();

    let mut result_matrix: Vec<Vec<usize>> = vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]];
    let mut matrix: Vec<Vec<usize>> = vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]];
    let mut f = match File::create(&path) {
        Err(why) => panic!("couldn't create {}: {}", display, why.description()),
        Ok(f) => f,
    };

    for i in 0..out.len() {
        for j in 0..out[i].len() {
            let row = if out[i][j][0] > out[i][j][1] {
                0
            } else if out[i][j][0] < out[i][j][1] {
                1
            } else {
                2
            };
            let col = if desire_output[i][j][0] > desire_output[i][j][1] {
                0
            } else if desire_output[i][j][0] < desire_output[i][j][1] {
                1
            } else {
                2
            };

            matrix[row][col] += 1;
        }
        if let Err(why) = f.write_all(format!(
            "\
            output\\expected output\t|\tclass 1\t|\tclass 2\t|\tundefined\n\
            class 1\t\t\t\t\t|\t{}\t\t|\t{}\t\t|\t{}\n\
            class 2\t\t\t\t\t|\t{}\t\t|\t{}\t\t|\t{}\n\
            undefined\t\t\t\t|\t{}\t\t|\t{}\t\t|\t{}\n\
            ============================================================================================\n",
            matrix[0][0], matrix[0][1], matrix[0][2],
            matrix[1][0], matrix[1][1], matrix[1][2],
            matrix[2][0], matrix[2][1], matrix[2][2]).as_bytes()){
            panic!("couldn't write to {}: {}", display, why.description());
        }
        for i in 0..result_matrix.len() {
            for j in 0..result_matrix[i].len() {
                result_matrix[i][j] += matrix[i][j];
            }
        }
        matrix = vec![vec![0, 0, 0], vec![0, 0, 0], vec![0, 0, 0]];
    }

    if let Err(why) = f.write_all(format!(
            "\n\
            output\\expected output\t|\tclass 1\t|\tclass 2\t|\tundefined\n\
            class 1\t\t\t\t\t|\t{}\t\t|\t{}\t\t|\t{}\n\
            class 2\t\t\t\t\t|\t{}\t\t|\t{}\t\t|\t{}\n\
            undefined\t\t\t\t|\t{}\t\t|\t{}\t\t|\t{}\n\
            ============================================================================================\n",
            result_matrix[0][0], result_matrix[0][1], result_matrix[0][2],
            result_matrix[1][0], result_matrix[1][1], result_matrix[1][2],
            result_matrix[2][0], result_matrix[2][1], result_matrix[2][2]).as_bytes()){
            panic!("couldn't write to {}: {}", display, why.description());
        }
}

pub fn remove_line(
    f: BufReader<File>,
    n: u8,
    recursion: bool,
    step: u8,
    line_per_data: u8,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut input_data = Vec::<Vec<f64>>::new();
    let mut all_data = Vec::<f64>::new();
    if recursion {
        let mut line_removed = 0;
        let mut line_read = 0;
        let mut line_data = 0;
        let mut input_vec = Vec::<f64>::new();
        for line in f.lines() {
            // if #step line(s) read, fall through line_removed, continue
            if line_read >= step {
                line_removed = 0;
                line_read = 0;
            }
            if line_removed < n {
                line_removed += 1;
                continue;
            }
            let line = line.unwrap();
            let split = line.split_whitespace().collect::<Vec<&str>>();
            let mut vec = to_f64_vec(split);
            // vec clone will be consumed after push
            input_vec.append(&mut vec.clone());
            all_data.append(&mut vec);
            line_read += 1;
            line_data += 1;

            if line_data >= line_per_data {
                input_data.push(input_vec.clone());
                input_vec.clear();
                line_data = 0;
            }
        }
    } else {
        let mut line_removed = 0;
        let mut line_data = 0;
        let mut input_vec = Vec::<f64>::new();
        for line in f.lines() {
            if line_removed < n {
                line_removed += 1;
                continue;
            }
            let line = line.unwrap();
            let split = line.split_whitespace().collect::<Vec<&str>>();
            let mut vec = to_f64_vec(split);
            input_vec.append(&mut vec.clone());
            all_data.append(&mut vec);
            line_data += 1;

            if line_data >= line_per_data {
                input_data.push(input_vec.clone());
                input_vec.clear();
                line_data = 0;
            }
        }
    }
    (input_data, all_data)
}

pub fn shuffle(mut data: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    rand::thread_rng().shuffle(&mut data);
    data
}

pub fn to_f64_vec(vec: Vec<&str>) -> Vec<f64> {
    let mut f64_vec: Vec<f64> = Vec::new();
    for item in vec {
        f64_vec.push(item.parse::<f64>().unwrap())
    }
    f64_vec
}

pub fn normalize(all_data: Vec<f64>, input_data: Vec<Vec<f64>>) -> MinMax {
    let min_max = min_max(all_data);
    let mut f_data_normalized = input_data;
    let divisor = min_max.1 - min_max.0;
    let min = min_max.0;
    for i in 0..f_data_normalized.len() {
        for j in 0..f_data_normalized[i].len() {
            f_data_normalized[i][j] = (f_data_normalized[i][j] - min) / divisor;
        }
    }
    // shuffle data
    rand::thread_rng().shuffle(&mut f_data_normalized);
    MinMax::new(f_data_normalized, Some(min_max))
}

pub fn denormalize(mut all_data: Vec<f64>, (min, max): (f64, f64)) -> Vec<f64> {
    let multiplier = max - min;
    for i in 0..all_data.len() {
        all_data[i] = all_data[i] * multiplier + min;
    }
    all_data
}

// (min, max) pair returned
fn min_max(data: Vec<f64>) -> (f64, f64) {
    let len = data.len();
    let mut min: f64;
    let mut max: f64;
    if len < 2 {
        // less than 2 element in vector
        (0_f64, 0_f64)
    } else {
        // init min & max
        {
            let d0 = data[0];
            let d1 = data[1];
            if d0 > d1 {
                max = d0;
                min = d1;
            } else {
                max = d1;
                min = d0;
            }
        }

        let mut index = 2;
        if len % 2 == 1 {
            while index + 1 < len {
                let d0 = data[index];
                let d1 = data[index + 1];
                if d0 > d1 {
                    if d0 > max {
                        max = d0;
                    }
                    if d1 < min {
                        min = d1;
                    }
                } else {
                    if d1 > max {
                        max = d1;
                    }
                    if d0 < min {
                        min = d0;
                    }
                }
                index += 2;
            }
            let dn = data[index];
            if dn > max {
                max = dn;
            } else if dn < min {
                min = dn;
            }
        } else {
            while index < len {
                let d0 = data[index];
                let d1 = data[index + 1];
                if d0 > d1 {
                    if d0 > max {
                        max = d0;
                    }
                    if d1 < min {
                        min = d1;
                    }
                } else {
                    if d1 > max {
                        max = d1;
                    }
                    if d0 < min {
                        min = d0;
                    }
                }
                index += 2;
            }
        }
        (min, max)
    }
}
