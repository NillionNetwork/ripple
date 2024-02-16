use std::fs::File;

// use tfhe::{generate_keys, prelude::*, set_server_key, ConfigBuilder, FheUint16};

fn quantize(x: f32, precision: u8) -> i32 {
    (x * ((1 << precision) as f32)) as i32
}

fn sigmoid(x: i32, input_precision: u8, output_precision: u8) -> i32 {
    let exp = (-x as f32 / ((1 << input_precision) as f32)).exp();
    ((1.0 / (1.0 + exp)) * (1 << output_precision) as f32) as i32
}

fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

fn main() {
    let weights_csv = File::open("iris_weights.csv").unwrap();
    let mut reader = csv::Reader::from_reader(weights_csv);
    let mut weights = Vec::new();
    let mut biases = Vec::new();

    for result in reader.deserialize() {
        let res: Vec<f32> = result.expect("a CSV record");
        biases.push(res[0]);
        weights.push(res[1..].to_vec());
    }

    let precision = 8;
    // Quantization of weights to 16 bits
    let weights_int = weights
        .iter()
        .map(|row| {
            row.iter()
                .map(|&w| quantize(w, precision))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bias_int = biases
        .iter()
        .map(|&w| quantize(w, precision))
        .collect::<Vec<_>>();

    let iris_dataset = linfa_datasets::iris();

    let num_records = iris_dataset.records.shape()[0];
    let num_features = iris_dataset.records.shape()[1];
    let mut means = vec![0f64; num_features];
    let mut stds = vec![0f64; num_features];
    for (sample, _) in iris_dataset.sample_iter() {
        for (feature, s) in sample.iter().enumerate() {
            means[feature] += s;
        }
    }
    for mean in means.iter_mut() {
        *mean /= num_records as f64;
    }
    for (sample, _) in iris_dataset.sample_iter() {
        for (feature, s) in sample.iter().enumerate() {
            let dev = s - means[feature];
            stds[feature] += dev * dev;
        }
    }
    for std in stds.iter_mut() {
        *std = (*std / num_records as f64).sqrt();
    }

    let mut total = 0;
    for (num, (sample, target)) in iris_dataset.sample_iter().enumerate() {
        let mut probabilities = Vec::new();
        for (model, bias) in weights_int.iter().zip(bias_int.iter()) {
            let mut prediction = 0;
            for ((&s, w), (mean, std)) in sample
                .iter()
                .zip(model.iter())
                .zip(means.iter().zip(stds.iter()))
            {
                // println!("s {:?}", s);
                let n = (s - mean) / std;
                let quantized = quantize(n as f32, precision);
                prediction += w * quantized + bias;
            }
            let activation = sigmoid(prediction, precision * 2, precision * 2);
            probabilities.push(activation);
        }
        let class = argmax(&probabilities).unwrap();
        println!(
            "[{num}] predicted {:?}, target {:?}",
            class,
            target.first().unwrap()
        );
        if class == *target.first().unwrap() {
            total += 1;
        }
    }
    let accuracy = (total as f32 / num_records as f32) * 100.0;
    println!("Accuracy {accuracy}%");
}
