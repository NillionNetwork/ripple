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
    let bit_width = 16u8;
    let precision = bit_width >> 2;
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

    let (weights, biases) = load_weights_and_biases();
    let (weights_int, bias_int) =
        quantize_weights_and_biases(&weights, &biases, precision, bit_width);
    let (iris_dataset, targets) = prepare_iris_dataset();
    let num_features = iris_dataset[0].len();
    let (means, stds) = means_and_stds(&iris_dataset, num_features);

    let start = Instant::now();
    let mut encrypted_dataset: Vec<Vec<_>> = iris_dataset
        .par_iter() // Use par_iter() for parallel iteration
        .map(|sample| {
            sample
                .par_iter()
                .zip(means.par_iter().zip(stds.par_iter()))
                .map(|(&s, (mean, std))| {
                    let quantized = quantize((s - mean) / std, precision, bit_width);
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

    // Build LUT for Sigmoid
    let exponential_lut = wopbs_key.generate_lut_radix(&encrypted_dataset[0][0], |x: u64| {
        exponential(x, 2 * precision, precision, bit_width)
    });

    let encrypted_dataset_short = encrypted_dataset.get_mut(0..4).unwrap();

    let all_probabilities = encrypted_dataset_short
        .par_iter_mut()
        .enumerate()
        .map(|(cnt, sample)| {
            let start = Instant::now();
            let probabilities = weights_int
                .iter()
                .zip(bias_int.iter())
                // .par_iter()
                // .zip(bias_int.par_iter())
                .map(|(model, &bias)| {
                    let mut prediction = server_key.create_trivial_radix(bias, nb_blocks.into());
                    for (s, &weight) in sample.iter_mut().zip(model.iter()) {
                        let mut d: u64 = client_key.decrypt(s);
                        println!("s: {:?}", d);
                        println!("weight: {:?}", weight);
                        let ct_prod = server_key.smart_scalar_mul(s, weight);
                        d = client_key.decrypt(&ct_prod);
                        println!("mul result: {:?}", d);
                        prediction = server_key.unchecked_add(&ct_prod, &prediction);
                        // FIXME: DEBUG
                        d = client_key.decrypt(&prediction);
                        println!("MAC result: {:?}", d);
                        println!();
                    }
                    println!();
                    prediction = wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction);
                    let activation = wopbs_key.wopbs(&prediction, &exponential_lut);

                    let probability = wopbs_key.keyswitch_to_pbs_params(&activation);
                    let d: u64 = client_key.decrypt(&probability);
                    println!("Exponential result: {:?}", d);

                    probability
                })
                .collect::<Vec<_>>();
            println!(
                "Finished inference #{:?} in {:?} sec.",
                cnt,
                start.elapsed().as_secs_f64()
            );
            probabilities
        })
        .collect::<Vec<_>>();
    // }

    // ------- Client side ------- //
    let mut total = 0;
    for (num, (target, probabilities)) in targets.iter().zip(all_probabilities.iter()).enumerate() {
        let ptxt_probabilities = probabilities
            .iter()
            .map(|p| client_key.decrypt(p))
            .collect::<Vec<u64>>();
        println!("{:?}", ptxt_probabilities);
        let class = argmax(&ptxt_probabilities).unwrap();
        println!("[{}] predicted {:?}, target {:?}", num, class, target);
        if class == *target {
            total += 1;
        }
    }
    let accuracy = (total as f32 / encrypted_dataset_short.len() as f32) * 100.0;
    println!("Accuracy {accuracy}%");
}
