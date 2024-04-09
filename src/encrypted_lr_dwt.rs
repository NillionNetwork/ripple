use std::time::Instant;

use clap::{App, Arg};
use fhe_lut::common::*;
use rayon::prelude::*;
// use serde::{Deserialize, Serialize};
use tfhe::{
    integer::{
        // ciphertext::BaseRadixCiphertext,
        gen_keys_radix,
        wopbs::*,
        IntegerCiphertext,
        IntegerRadixCiphertext,
        RadixCiphertext,
    },
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

fn eval_exp(x: u64, exp_map: &Vec<u64>) -> u64 {
    exp_map[x as usize]
}

fn main() {
    println!("Encrypted Logistic Regression using Discrete Wavelet Transform");

    let matches = App::new("Ripple")
        .about("Vanilla Encrypted Logistic Regression")
        .arg(
            Arg::new("num-samples")
                .long("num-samples")
                .short('n')
                .takes_value(true)
                .value_name("INT")
                .help("Number of samples")
                .default_value("1")
                .required(false),
        )
        .get_matches();

    let num_samples = matches
        .value_of("num-samples")
        .unwrap_or("1")
        .parse::<usize>()
        .expect("Number of samples must be an integer");

    // ------- Client side ------- //
    let bit_width = 24;
    let precision = 8;
    let table_size = bit_width / 2;
    assert!(precision <= bit_width / 2);

    let (lut_lsb, _lut_msb) = haar(table_size, 2 * precision, precision, bit_width);

    // Number of blocks per ciphertext
    let nb_blocks = bit_width >> 2;
    println!("Number of blocks for the radix decomposition: {:?}", nb_blocks);

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
                    let mut lsb = client_key
                        .encrypt(quantized & (1 << ((nb_blocks << 1) - 1)))
                        .into_blocks(); // Get LSBs
                    let msb = client_key
                        .encrypt(quantized >> (nb_blocks << 1))
                        .into_blocks(); // Get MSBs
                    lsb.extend(msb);
                    RadixCiphertext::from_blocks(lsb)
                })
                .collect()
        })
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //

    let lut_gen_start = Instant::now();
    println!("Generating LUT.");
    let mut dummy: RadixCiphertext =
        server_key.create_trivial_radix(2_u64, (nb_blocks << 1).into());
    for _ in 0..weights_int.len() {
        let dummy_2 = server_key.scalar_mul_parallelized(&dummy, 2_u64);
        dummy = server_key.add_parallelized(&dummy_2, &dummy);
    }
    let dummy_blocks = &dummy.into_blocks()[(nb_blocks as usize)..((nb_blocks << 1) as usize)];
    let dummy_msb = RadixCiphertext::from_blocks(dummy_blocks.to_vec());
    let dummy_msb = server_key.scalar_add_parallelized(&dummy_msb, 1);
    let exp_lut_lsb = wopbs_key.generate_lut_radix(&dummy_msb, |x: u64| eval_exp(x, &lut_lsb));
    println!(
        "LUT generation done in {:?} sec.",
        lut_gen_start.elapsed().as_secs_f64()
    );

    // Inference
    assert!(num_samples <= encrypted_dataset.len());
    let all_probabilities = if num_samples > 1 {
        encrypted_dataset
            .par_iter_mut()
            .enumerate()
            .take(num_samples)
            .map(|(cnt, sample)| {
                let start = Instant::now();
                println!("Starting inference #{:?}.", cnt);

                let mut prediction = server_key.create_trivial_radix(bias_int, nb_blocks.into());
                for (s, &weight) in sample.iter_mut().zip(weights_int.iter()) {
                    let ct_prod = server_key.scalar_mul_parallelized(s, weight);
                    prediction = server_key.add_parallelized(&ct_prod, &prediction);
                }
                // Truncate
                let prediction_blocks =
                    &prediction.into_blocks()[(nb_blocks as usize)..((nb_blocks << 1) as usize)];
                let prediction_msb = RadixCiphertext::from_blocks(prediction_blocks.to_vec());
                let prediction_msb = server_key.scalar_add_parallelized(&prediction_msb, 1);
                // Keyswitch and Bootstrap
                prediction = wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction_msb);
                let activation_lsb = wopbs_key.wopbs(&prediction, &exp_lut_lsb);
                let probability = wopbs_key.keyswitch_to_pbs_params(&activation_lsb);
                println!(
                    "Finished inference #{:?} in {:?} sec.",
                    cnt,
                    start.elapsed().as_secs_f64()
                );
                probability
            })
            .collect::<Vec<_>>()
    } else {
        let start = Instant::now();
        println!("Starting inference.");

        let mut prediction = server_key.create_trivial_radix(bias_int, nb_blocks.into());
        for (s, &weight) in encrypted_dataset[0].iter_mut().zip(weights_int.iter()) {
            let ct_prod = server_key.scalar_mul_parallelized(s, weight);
            prediction = server_key.add_parallelized(&ct_prod, &prediction);
        }
        // Truncate
        let prediction_blocks =
            &prediction.into_blocks()[(nb_blocks as usize)..((nb_blocks << 1) as usize)];
        let prediction_msb = RadixCiphertext::from_blocks(prediction_blocks.to_vec());
        let prediction_msb = server_key.scalar_add_parallelized(&prediction_msb, 1);
        // Keyswitch and Bootstrap
        prediction = wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction_msb);
        let activation_lsb = wopbs_key.wopbs(&prediction, &exp_lut_lsb);
        let probability = wopbs_key.keyswitch_to_pbs_params(&activation_lsb);

        println!(
            "Finished inference in {:?} sec.",
            start.elapsed().as_secs_f64()
        );
        vec![probability]
    };

    // ------- Client side ------- //
    let mut total = 0;
    for (num, (target, probability)) in targets.iter().zip(all_probabilities.iter()).enumerate() {
        let ptxt_probability: u64 = client_key.decrypt(probability);
        let pr = (ptxt_probability as f64) / ((1<<precision) as f64);

        let class = (ptxt_probability > quantize(0.5, precision, bit_width)) as usize;
        println!("[{}] predicted {:?}, target {:?} (prediction probability {:?})", num, class, target, pr);
        if class == *target {
            total += 1;
        }
    }
    let accuracy = (total as f32 / num_samples as f32) * 100.0;
    println!("Accuracy {accuracy}%");
}
