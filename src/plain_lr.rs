// use debug_print::debug_println;
use fhe_lut::common::*;
use rayon::prelude::*;

pub fn quantize_dataset(
    dataset: &Vec<Vec<f64>>,
    means: &Vec<f64>,
    stds: &Vec<f64>,
    precision: u8,
    bit_width: u8,
) -> Vec<Vec<u64>> {
    dataset
        .par_iter() // Use par_iter() for parallel iteration
        .map(|sample| {
            sample
                .par_iter()
                .zip(means.par_iter().zip(stds.par_iter()))
                .map(|(&s, (mean, std))| quantize((s - mean) / std, precision, bit_width))
                .collect()
        })
        .collect()
}

fn main() {
    let bit_width = 16u8;
    let precision = bit_width >> 2;

    let (weights, biases) = load_weights_and_biases();
    let (weights_int, bias_int) =
        quantize_weights_and_biases(&weights, &biases, precision, bit_width);

    let (iris_dataset, targets) = prepare_iris_dataset();
    let num_features = iris_dataset[0].len();
    let (means, stds) = means_and_stds(&iris_dataset, num_features);

    let quantized_dataset = quantize_dataset(&iris_dataset, &means, &stds, precision, bit_width);

    let mut total = 0;
    for (target, sample) in targets.iter().zip(quantized_dataset.iter()) {
        // Server computation
        let probabilities = weights_int
            .par_iter()
            .zip(bias_int.par_iter())
            .map(|(model, &bias)| {
                let mut prediction = bias;
                for (&s, &w) in sample.iter().zip(model.iter()) {
                    println!("s: {:?}", s);
                    println!("weight: {:?}", w);
                    prediction = add(prediction, mul(w, s, bit_width), bit_width);
                    println!("MAC result: {:?}", prediction);
                }
                println!();
                exponential(prediction, 2 * precision, precision, bit_width)
            })
            .collect::<Vec<_>>();

        // Client computation
        let class = argmax(&probabilities).unwrap();
        println!("predicted {class:?}, target {target:?}");
        if class == *target {
            total += 1;
        }
    }

    let accuracy = (total as f64 / iris_dataset.len() as f64) * 100.0;
    println!("Accuracy {accuracy}%");
    println!("precision: {precision}, bit_width: {bit_width}");
}
