use rand::{self, Rng};

#[derive(Debug)]
pub struct MinMax {
    pub min: f64,
    pub max: f64,
    pub f_data: Vec<Vec<f64>>,
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
    MinMax {
        f_data: f_data_normalized,
        min: min_max.0,
        max: min_max.1,
    }
}

pub fn sigmoid(t: f64) -> f64 {
    1_f64 / (1_f64 + (-t).exp())
}

pub fn d_sigmoid(t: f64) -> f64 {
    let exp = (-t).exp();
    let exp = if_infinity(exp);
    exp / (exp + 1_f64).powi(2)
}

pub fn sum_sqrt_err(error_vec: Vec<f64>) -> f64 {
    let mut sse: f64 = 0 as f64;
    for (_i, item) in error_vec.iter().enumerate() {
        sse += item.powi(2);
    }
    sse / 2.0
}

pub fn split_section(data: MinMax, s: usize) -> Vec<Vec<Vec<f64>>> {
    let len = data.f_data.len();
    let n = len / s;
    let mut split_data: Vec<Vec<Vec<f64>>> = Vec::new();
    for i in 0..s {
        // range slice start..end, from start to, not including, end
        if i == s - 1 {
            // last data chunk
            split_data.push(data.f_data[i * n..len].to_vec());
        } else {
            split_data.push(data.f_data[i * n..(i + 1) * n].to_vec());
        }
    }
    split_data
}

fn if_infinity(value: f64) -> f64 {
    use std::f64;

    if value.is_infinite() {
        f64::MAX
    } else {
        value
    }
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
