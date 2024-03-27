use std::fs::File;

use dwt::{transform, wavelet::Haar, Operation};

pub fn to_signed(x: u64, bit_width: u8) -> i64 {
    if x > (1u64 << (bit_width - 1)) {
        (x as i128 - (1i128 << bit_width)) as i64
    } else {
        x as i64
    }
}

pub fn from_signed(x: i64, bit_width: u8) -> u64 {
    (x as i128).rem_euclid(1i128 << bit_width) as u64
}

pub fn quantize(x: f64, precision: u8, bit_width: u8) -> u64 {
    from_signed((x * ((1u128 << precision) as f64)) as i64, bit_width)
}

pub fn unquantize(x: u64, precision: u8, bit_width: u8) -> f64 {
    to_signed(x, bit_width) as f64 / ((1u128 << precision) as f64)
}

pub fn add(a: u64, b: u64, bit_width: u8) -> u64 {
    (a as u128 + b as u128).rem_euclid(1u128 << bit_width) as u64
}

pub fn mul(a: u64, b: u64, bit_width: u8) -> u64 {
    (a as u128 * b as u128).rem_euclid(1u128 << bit_width) as u64
}

pub fn sigmoid(x: u64, input_precision: u8, output_precision: u8, bit_width: u8) -> u64 {
    let x = to_signed(x, bit_width) as f64;
    let shift = (1u128 << input_precision) as f64;
    let sig = 1f64 / (1f64 + (-x / shift).exp());
    (sig * ((1u128 << output_precision) as f64)) as u64
}

pub fn load_weights_and_biases() -> (Vec<f64>, f64) {
    let weights_csv = File::open("data/penguins_weight.csv").unwrap();
    let mut reader = csv::Reader::from_reader(weights_csv);
    let mut weights = vec![];
    let mut bias = 0f64;

    for result in reader.deserialize() {
        let res: Vec<f64> = result.expect("a CSV record");
        bias = res[0];
        weights = res[1..].to_vec();
    }

    (weights, bias)
}

pub fn quantize_weights_and_bias(
    weights: &[f64],
    bias: f64,
    precision: u8,
    bit_width: u8,
) -> (Vec<u64>, u64) {
    let weights_int = weights
        .iter()
        .map(|&w| quantize(w, precision, bit_width))
        .collect::<Vec<_>>();
    // Quantize and double precision as bias will be added to double precision terms
    let bias_int = mul(
        1 << precision,
        quantize(bias, precision, bit_width),
        bit_width,
    );

    (weights_int, bias_int)
}

pub fn prepare_penguins_dataset() -> (Vec<Vec<f64>>, Vec<usize>) {
    let data_csv = File::open("data/penguins_data.csv").unwrap();
    let mut reader = csv::Reader::from_reader(data_csv);
    let mut dataset = vec![];

    for result in reader.deserialize() {
        let res: Vec<f64> = result.expect("a CSV record");
        dataset.push(res);
    }

    let target_csv = File::open("data/penguins_target.csv").unwrap();
    let mut reader = csv::Reader::from_reader(target_csv);
    let mut targets = vec![];
    for result in reader.deserialize() {
        let res: Vec<f64> = result.expect("a CSV record");
        targets.push(res[0] as usize);
    }

    (dataset, targets)
}

pub fn means_and_stds(dataset: &[Vec<f64>], num_features: usize) -> (Vec<f64>, Vec<f64>) {
    let mut maxs = vec![0f64; num_features];
    let mut mins = vec![0f64; num_features];

    for sample in dataset.iter() {
        for (feature, s) in sample.iter().enumerate() {
            if maxs[feature] < *s {
                maxs[feature] = *s;
            }
            if mins[feature] > *s {
                mins[feature] = *s;
            }
        }
    }

    (mins, maxs)
}

pub fn haar(precision: u8, bit_width: u8) -> (Vec<u64>, Vec<u64>) {
    let max = 1 << bit_width;
    let mut data = Vec::new();
    for x in 0..max {
        data.push(unquantize(x, precision, bit_width).exp());
    }
    let level = 2u8;
    transform(&mut data, Operation::Forward, &Haar::new(), level as usize);
    let coef_len = 1 << (bit_width - level);
    let haar = data
        .get(0..coef_len)
        .unwrap()
        .iter()
        .map(|x| quantize(*x, precision, bit_width));
    let lsb = haar.clone().map(|x| x & 0xFF).collect();
    let msb = haar.map(|x| x >> (bit_width / 2) & 0xFF).collect();
    (lsb, msb)
}
