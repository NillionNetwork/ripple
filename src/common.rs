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

pub fn exponential(x: u64, input_precision: u8, output_precision: u8, bit_width: u8) -> u64 {
    let x = to_signed(x, bit_width) as f64;
    let shift = (1u128 << input_precision) as f64;
    let exp = (x / shift).exp();
    (exp * ((1u128 << output_precision) as f64)) as u64
}

pub fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

pub fn load_weights_and_biases() -> (Vec<Vec<f64>>, Vec<f64>) {
    let weights_csv = File::open("data/iris_weights.csv").unwrap();
    let mut reader = csv::Reader::from_reader(weights_csv);
    let mut weights = vec![];
    let mut biases = vec![];

    for result in reader.deserialize() {
        let res: Vec<f64> = result.expect("a CSV record");
        biases.push(res[0]);
        weights.push(res[1..].to_vec());
    }

    (weights, biases)
}

pub fn quantize_weights_and_biases(
    weights: &[Vec<f64>],
    biases: &[f64],
    precision: u8,
    bit_width: u8,
) -> (Vec<Vec<u64>>, Vec<u64>) {
    let weights_int = weights
        .iter()
        .map(|row| {
            row.iter()
                .map(|&w| quantize(w, precision, bit_width))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bias_int = biases
        .iter()
        // Quantize and double precision as bias will be added to double precision terms
        .map(|&b| mul(1 << precision, quantize(b, precision, bit_width), bit_width))
        .collect::<Vec<_>>();

    (weights_int, bias_int)
}

pub fn prepare_iris_dataset() -> (Vec<Vec<f64>>, Vec<usize>) {
    let iris = linfa_datasets::iris();
    let mut iris_dataset = vec![];
    let mut targets = vec![];

    for (sample, target) in iris.sample_iter() {
        iris_dataset.push(sample.to_vec());
        targets.push(*target.first().unwrap());
    }

    (iris_dataset, targets)
}

pub fn means_and_stds(dataset: &[Vec<f64>], num_features: usize) -> (Vec<f64>, Vec<f64>) {
    let mut means = vec![0f64; num_features];
    let mut stds = vec![0f64; num_features];

    for sample in dataset.iter() {
        for (feature, s) in sample.iter().enumerate() {
            means[feature] += s;
        }
    }
    for mean in means.iter_mut() {
        *mean /= dataset.len() as f64;
    }
    for sample in dataset.iter() {
        for (feature, s) in sample.iter().enumerate() {
            let dev = s - means[feature];
            stds[feature] += dev * dev;
        }
    }
    for std in stds.iter_mut() {
        *std = (*std / dataset.len() as f64).sqrt();
    }

    (means, stds)
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
