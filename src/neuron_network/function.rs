#[derive(Debug)]
pub struct MinMax {
    min: f64,
    max: f64,
    f_data: Vec<f64>,
}

pub fn normalize(data: Vec<&str>) -> MinMax {
    let min_max = min_max(data);
    let mut f_data_normalized = min_max.f_data;
    let divisor = min_max.max - min_max.min;
    let min = min_max.min;
    for i in 0..f_data_normalized.len() {
        f_data_normalized[i] = (f_data_normalized[i] - min) / divisor;
    }
    MinMax {
        f_data: f_data_normalized,
        ..min_max
    }
}

fn min_max(data: Vec<&str>) -> MinMax {
    let len = data.len();
    let mut min: f64;
    let mut max: f64;
    let mut f_data: Vec<f64> = Vec::with_capacity(len);
    if len < 2 {
        // less than 2 element in vector
        MinMax {
            min: 0_f64,
            max: 0_f64,
            f_data: vec![0_f64],
        }
    } else {
        // init min & max
        {
            let d0 = data[0].parse::<f64>().unwrap();
            let d1 = data[1].parse::<f64>().unwrap();
            f_data.push(d0);
            f_data.push(d1);
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
                let d0 = data[index].parse::<f64>().unwrap();
                let d1 = data[index + 1].parse::<f64>().unwrap();
                f_data.push(d0);
                f_data.push(d1);
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
            let dn = data[index].parse::<f64>().unwrap();
            f_data.push(dn);
            if dn > max {
                max = dn;
            } else if dn < min {
                min = dn;
            }
        } else {
            while index < len {
                let d0 = data[index].parse::<f64>().unwrap();
                let d1 = data[index + 1].parse::<f64>().unwrap();
                f_data.push(d0);
                f_data.push(d1);
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
        MinMax { min, max, f_data }
    }
}
