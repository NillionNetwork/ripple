use std::fs::File;

use debug_print::debug_println;
use rayon::prelude::*;

fn to_signed(x: u64) -> i64 {
    if x > (1u64 << 63) {
        (x as i128 - (1i128 << 64)) as i64
    } else {
        x as i64
    }
}

fn from_signed(x: i64) -> u64 {
    (x as i128).rem_euclid(1i128 << 64) as u64
}

fn quantize(x: f64, precision: u8) -> u64 {
    from_signed((x * ((1u128 << precision) as f64)) as i64)
}

fn unquantize(x: u64, precision: u8) -> f64 {
    to_signed(x) as f64 / ((1u128 << precision) as f64)
}

fn add(a: u64, b: u64) -> u64 {
    (a as u128 + b as u128).rem_euclid(1u128 << 64) as u64
}

fn mul(a: u64, b: u64) -> u64 {
    (a as u128 * b as u128).rem_euclid(1u128 << 64) as u64
}

fn truncate(x: u64, precision: u8) -> u64 {
    from_signed(to_signed(x) / (1i64 << precision))
}

fn sigmoid(x: u64, input_precision: u8, output_precision: u8) -> u64 {
    let x = to_signed(x) as f64;
    let shift = (1u128 << input_precision) as f64;
    let exp = (x / shift).exp();
    (exp * ((1u128 << output_precision) as f64)) as u64
    // ((1.0 / (1.0 + exp)) * (1 << output_precision) as f64) as u64
}

fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

fn load_weights_and_biases() -> (Vec<Vec<f64>>, Vec<f64>) {
    let weights_csv = File::open("iris_weights.csv").unwrap();
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

fn quantize_weights_and_biases(
    weights: &[Vec<f64>],
    biases: &[f64],
    precision: u8,
) -> (Vec<Vec<u64>>, Vec<u64>) {
    let weights_int = weights
        .iter()
        .map(|row| {
            row.iter()
                .map(|&w| quantize(w.into(), precision))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bias_int = biases
        .iter()
        .map(|&w| quantize(w.into(), precision))
        .collect::<Vec<_>>();

    (weights_int, bias_int)
}

fn prepare_iris_dataset() -> (Vec<Vec<f64>>, Vec<usize>) {
    let iris = linfa_datasets::iris();
    let mut iris_dataset = vec![];
    let mut targets = vec![];

    for (sample, target) in iris.sample_iter() {
        iris_dataset.push(sample.to_vec());
        targets.push(*target.first().unwrap());
    }

    (iris_dataset, targets)
}

fn means_and_stds(dataset: &[Vec<f64>], num_features: usize) -> (Vec<f64>, Vec<f64>) {
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

fn main() {
    let precision = 6;
    let (weights, biases) = load_weights_and_biases();
    let (weights_int, bias_int) = quantize_weights_and_biases(&weights, &biases, precision);

    let (iris_dataset, targets) = prepare_iris_dataset();
    let num_features = iris_dataset[0].len();
    let (means, stds) = means_and_stds(&iris_dataset, num_features);

    let quantized_dataset: Vec<Vec<_>> = iris_dataset
        .par_iter() // Use par_iter() for parallel iteration
        .map(|sample| {
            sample
                .par_iter()
                .zip(means.par_iter().zip(stds.par_iter()))
                .map(|(&s, (mean, std))| quantize((s - mean) / std, precision))
                .collect()
        })
        .collect();

    let mut total = 0;
    for (target, sample) in targets.iter().zip(quantized_dataset.iter()) {
        // Server computation
        let probabilities: Vec<_> = weights_int
            .par_iter()
            .zip(bias_int.par_iter())
            .map(|(model, &bias)| {
                let mut prediction = bias;
                for (&s, &w) in sample.iter().zip(model.iter()) {
                    let cur = truncate(mul(w, s), precision);
                    prediction = add(prediction, cur);
                }
                sigmoid(prediction, precision, precision)
            })
            .collect();

        // Client computation
        let class = argmax(&probabilities).unwrap();
        debug_println!("predicted {class:?}, target {target:?}");
        if class == *target {
            total += 1;
        }
    }

    let accuracy = (total as f64 / iris_dataset.len() as f64) * 100.0;
    println!("Accuracy {accuracy}%");
}
