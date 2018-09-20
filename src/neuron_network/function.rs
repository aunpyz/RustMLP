use rand::{self, Rng};

#[derive(Debug)]
pub struct MinMax {
    min: f64,
    max: f64,
    f_data: Vec<Vec<f64>>,
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
