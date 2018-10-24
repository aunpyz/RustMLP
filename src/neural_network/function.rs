use MinMax;

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
