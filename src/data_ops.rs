use rand::{self, Rng};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use MinMax;

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
