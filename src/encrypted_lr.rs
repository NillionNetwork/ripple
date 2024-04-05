use std::time::Instant;

use fhe_lut::common::*;
use rayon::prelude::*;
use tfhe::{
    integer::{gen_keys_radix, wopbs::*},
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

fn main() {
    // ------- Client side ------- //
    let bit_width = 24u8;
    let precision = 8;
    assert!(precision <= bit_width / 2);

    // Number of blocks per ciphertext
    let nb_blocks = bit_width / 2;

    let start = Instant::now();
    // Generate radix keys
    let (client_key, server_key) = gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, nb_blocks.into());

    // Generate key for PBS (without padding)
    let wopbs_key = WopbsKey::new_wopbs_key(
        &client_key,
        &server_key,
        &WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    );
    println!(
        "Key generation done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    let (weights, bias) = load_weights_and_biases();
    let (weights_int, bias_int) = quantize_weights_and_bias(&weights, bias, precision, bit_width);
    let (dataset, targets) = prepare_penguins_dataset();

    let start = Instant::now();
    let mut encrypted_dataset: Vec<Vec<_>> = dataset
        .par_iter() // Use par_iter() for parallel iteration
        .map(|sample| {
            sample
                .par_iter()
                .map(|&s| {
                    let quantized = quantize(s, precision, bit_width);
                    client_key.encrypt(quantized)
                })
                .collect()
        })
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //

    // Build LUT for Sigmoid -- Offline cost
    let start = Instant::now();
    println!("Generating LUT.");
    let sigmoid_lut = wopbs_key.generate_lut_radix(&encrypted_dataset[0][0], |x: u64| {
        sigmoid(x, 2 * precision, precision, bit_width)
    });
    println!("Generated LUT in {:?} sec.", start.elapsed().as_secs_f64());

    let encrypted_dataset_short = encrypted_dataset.get_mut(0..8).unwrap();

    // Inference
    let all_probabilities = encrypted_dataset_short
        .par_iter_mut()
        .enumerate()
        .map(|(cnt, sample)| {
            let start = Instant::now();
            println!("Started inference #{:?}.", cnt);

            let mut prediction = server_key.create_trivial_radix(bias_int, nb_blocks.into());
            for (s, &weight) in sample.iter_mut().zip(weights_int.iter()) {
                let ct_prod = server_key.smart_scalar_mul(s, weight);
                prediction = server_key.unchecked_add(&ct_prod, &prediction);
            }
            prediction = wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction);
            let activation = wopbs_key.wopbs(&prediction, &sigmoid_lut);
            let probability = wopbs_key.keyswitch_to_pbs_params(&activation);

            println!(
                "Finished inference #{:?} in {:?} sec.",
                cnt,
                start.elapsed().as_secs_f64()
            );
            probability
        })
        .collect::<Vec<_>>();

    // ------- Client side ------- //
    let mut total = 0;
    for (num, (target, probability)) in targets.iter().zip(all_probabilities.iter()).enumerate() {
        let ptxt_probability: u64 = client_key.decrypt(probability);

        let class = (ptxt_probability > quantize(0.5, precision, bit_width)) as usize;
        println!("[{}] predicted {:?}, target {:?}", num, class, target);
        if class == *target {
            total += 1;
        }
    }
    let accuracy = (total as f32 / encrypted_dataset_short.len() as f32) * 100.0;
    println!("Accuracy {accuracy}%");
}
