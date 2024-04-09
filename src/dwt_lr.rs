// use debug_print::debug_println;
use fhe_lut::common::*;
use rayon::prelude::*;

pub fn quantize_dataset(dataset: &Vec<Vec<f64>>, precision: u8, bit_width: u8) -> Vec<Vec<u64>> {
    dataset
        .par_iter() // Use par_iter() for parallel iteration
        .map(|sample| {
            sample
                .par_iter()
                .map(|&s| quantize(s, precision, bit_width))
                .collect()
        })
        .collect()
}

fn main() {
    let bit_width = 24u8;
    let precision = 8;
    let table_size = 12;

    println!("Starting Haar");
    let (lut_lsb, _lut_msb) = haar(table_size, precision * 2, table_size, bit_width);
    println!("{:?}", lut_lsb);
    // println!("{:?}", lut_msb);
    println!("End Haar");

    let (weights, bias) = load_weights_and_biases();
    let (weights_int, bias_int) = quantize_weights_and_bias(&weights, bias, precision, bit_width);

    let (dataset, targets) = prepare_penguins_dataset();
    let quantized_dataset = quantize_dataset(&dataset, precision, bit_width);

    println!("Starting Run");

    let mut total = 0;
    for (target, sample) in targets.iter().zip(quantized_dataset.iter()) {
        // Server computation
        let mut prediction = bias_int;
        for (&s, &w) in sample.iter().zip(weights_int.iter()) {
            // println!("s: {:?}", s);
            // println!("weight: {:?}", w);
            prediction = add(prediction, mul(w, s, bit_width), bit_width);
            // println!("MAC result: {:?}", prediction);
        }
        // println!("prediction {prediction}");
        let probability = sigmoid(prediction, 2 * precision, table_size, bit_width);
        let prediction = prediction >> (bit_width - table_size);
        let lut_probability = lut_lsb[prediction as usize];

        // println!("{probability1} {probability}");
        println!("diff {:?}", probability as i64 - lut_probability as i64);

        let class = (lut_probability > quantize(0.5, table_size, bit_width)) as usize;

        // Client computation
        // println!("predicted {class:?}, target {target:?}");
        if class == *target {
            total += 1;
        }
        // println!();
    }
    let accuracy = (total as f64 / dataset.len() as f64) * 100.0;
    println!("Accuracy {accuracy}%");
    println!("table size: {table_size}, precision: {precision}, bit_width: {bit_width}");
}
