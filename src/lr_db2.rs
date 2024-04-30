use std::time::Instant;

use clap::{App, Arg};
use rayon::prelude::*;
use ripple::common::*;
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

fn eval_lut(x: u64, lut_entries: &Vec<u64>) -> u64 {
    lut_entries[x as usize]
}

fn eval_lut_add_1(x: u64, lut_entries: &Vec<u64>, ptxt_space: u64) -> u64 {
    lut_entries[((x + 1) % ptxt_space) as usize]
}

fn eval_lut_add_2(x: u64, lut_entries: &Vec<u64>, ptxt_space: u64) -> u64 {
    lut_entries[((x + 2) % ptxt_space) as usize]
}

fn main() {
    let matches = App::new("Ripple")
        .about("Encrypted Logistic Regression with DB2 DWT LUTs")
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
    let bit_width = 24u8;
    let lut_bit_width = 18u8;
    let precision = 8;
    let j = 8; // wave depth
    assert!(precision <= bit_width / 2);

    let (lut_lsbs, lut_msb) = db2();

    // Number of blocks for full precision
    let nb_blocks = bit_width >> 1;

    // Number of blocks for J LSBs
    let nb_blocks_lsb = j >> 1;
    println!("Number of blocks for LSB path: {:?}", nb_blocks_lsb);

    // Number of blocks for n-J MSBs
    let nb_blocks_msb = (bit_width - j) >> 1;
    println!("Number of blocks for MSB path: {:?}", nb_blocks_msb);

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
                .iter()
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

    let lut_gen_start = Instant::now();
    println!("Generating LUT.");
    let mut dummy: RadixCiphertext = server_key.create_trivial_radix(2_u64, nb_blocks.into());
    for _ in 0..weights_int.len() {
        let dummy_2 = server_key.scalar_mul_parallelized(&dummy, 2_u64);
        dummy = server_key.add_parallelized(&dummy_2, &dummy);
    }
    let dummy_blocks = &dummy.into_blocks()[3..(nb_blocks as usize)];
    let dummy_blocks_lsb = &dummy_blocks[0..((j >> 1) as usize)];
    let dummy_blocks_msb = &dummy_blocks[((j >> 1) as usize)..((nb_blocks as usize) - 3)];
    let dummy_lsb = RadixCiphertext::from_blocks(dummy_blocks_lsb.to_vec());
    let dummy_msb = RadixCiphertext::from_blocks(dummy_blocks_msb.to_vec());
    let dummy_msb = server_key.scalar_add_parallelized(&dummy_msb, 1);
    let dummy_lsb = server_key.scalar_add_parallelized(&dummy_lsb, 1);
    let mut lsb_luts = Vec::new();
    let mut msb_luts = Vec::new();
    for lut_lsb in lut_lsbs.iter() {
        lsb_luts.push(wopbs_key.generate_lut_radix(&dummy_lsb, |x: u64| eval_lut(x, lut_lsb)));
    }
    msb_luts.push(wopbs_key.generate_lut_radix(&dummy_msb, |x: u64| eval_lut(x, &lut_msb)));
    msb_luts.push(wopbs_key.generate_lut_radix(&dummy_msb, |x: u64| {
        eval_lut_add_1(x, &lut_msb, 2u64.pow((lut_bit_width - j) as u32))
    }));
    msb_luts.push(wopbs_key.generate_lut_radix(&dummy_msb, |x: u64| {
        eval_lut_add_2(x, &lut_msb, 2u64.pow((lut_bit_width - j) as u32))
    }));
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
                println!("Started inference #{:?}.", cnt);

                let mut prediction = server_key.create_trivial_radix(bias_int, nb_blocks.into());
                for (s, &weight) in sample.iter_mut().zip(weights_int.iter()) {
                    let ct_prod = server_key.scalar_mul_parallelized(s, weight);
                    prediction = server_key.add_parallelized(&ct_prod, &prediction);
                }
                // Truncate 6 LSBs to reduce to 18 bits
                let prediction_blocks = &prediction.into_blocks()[3..(nb_blocks as usize)];
                let (prediction_msb, prediction_lsb) = rayon::join(
                    || {
                        let prediction_blocks_lsb = &prediction_blocks[0..((j >> 1) as usize)];
                        let prediction_lsb =
                            RadixCiphertext::from_blocks(prediction_blocks_lsb.to_vec());
                        let prediction_lsb = server_key.scalar_add_parallelized(&prediction_lsb, 1);
                        wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction_lsb)
                    },
                    || {
                        let prediction_blocks_msb =
                            &prediction_blocks[((j >> 1) as usize)..((nb_blocks as usize) - 3)];
                        let prediction_msb =
                            RadixCiphertext::from_blocks(prediction_blocks_msb.to_vec());
                        let prediction_msb = server_key.scalar_add_parallelized(&prediction_msb, 1);
                        wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction_msb)
                    },
                );
                let prods: Vec<RadixCiphertext> = (0..3)
                    .into_par_iter()
                    .map(|i| {
                        let activation_lsb = wopbs_key.wopbs(&prediction_lsb, &lsb_luts[i]);
                        let activation_msb = wopbs_key.wopbs(&prediction_msb, &msb_luts[i]);
                        let activation_lsb_blocks = wopbs_key
                            .keyswitch_to_pbs_params(&activation_lsb)
                            .into_blocks();
                        let activation_lsb = RadixCiphertext::from_blocks(activation_lsb_blocks);
                        let activation_msb = wopbs_key.keyswitch_to_pbs_params(&activation_msb);
                        // Multiply and pad to n bits
                        let mut ct_prod_blocks = server_key
                            .mul_parallelized(&activation_lsb, &activation_msb)
                            .into_blocks();
                        let padding: RadixCiphertext =
                            server_key.create_trivial_radix(0, (j >> 1).into());
                        let padding_blocks = padding.into_blocks();
                        ct_prod_blocks.extend(padding_blocks);
                        RadixCiphertext::from_blocks(ct_prod_blocks.to_vec())
                    })
                    .collect();
                // Sum products
                let probability = server_key.add_parallelized(&prods[0], &prods[1]);
                let probability = server_key.add_parallelized(&probability, &prods[2]);
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
        println!("Started inference.");

        let mut prediction = server_key.create_trivial_radix(bias_int, nb_blocks.into());
        for (s, &weight) in encrypted_dataset[0].iter_mut().zip(weights_int.iter()) {
            let ct_prod = server_key.scalar_mul_parallelized(s, weight);
            prediction = server_key.add_parallelized(&ct_prod, &prediction);
        }
        // Split into J LSBs and n-J MSBs
        let prediction_blocks = &prediction.into_blocks()[3..(nb_blocks as usize)];
        let (prediction_msb, prediction_lsb) = rayon::join(
            || {
                let prediction_blocks_lsb = &prediction_blocks[0..((j >> 1) as usize)];
                let prediction_lsb = RadixCiphertext::from_blocks(prediction_blocks_lsb.to_vec());
                let prediction_lsb = server_key.scalar_add_parallelized(&prediction_lsb, 1);
                wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction_lsb)
            },
            || {
                let prediction_blocks_msb =
                    &prediction_blocks[((j >> 1) as usize)..((nb_blocks as usize) - 3)];
                let prediction_msb = RadixCiphertext::from_blocks(prediction_blocks_msb.to_vec());
                let prediction_msb = server_key.scalar_add_parallelized(&prediction_msb, 1);
                wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction_msb)
            },
        );
        let prods: Vec<RadixCiphertext> = (0..3)
            .into_par_iter()
            .map(|i| {
                let activation_lsb = wopbs_key.wopbs(&prediction_lsb, &lsb_luts[i]);
                let activation_msb = wopbs_key.wopbs(&prediction_msb, &msb_luts[i]);
                let activation_lsb_blocks = wopbs_key
                    .keyswitch_to_pbs_params(&activation_lsb)
                    .into_blocks();
                let activation_lsb = RadixCiphertext::from_blocks(activation_lsb_blocks);
                let activation_msb = wopbs_key.keyswitch_to_pbs_params(&activation_msb);
                // Multiply and pad to n bits
                let mut ct_prod_blocks = server_key
                    .mul_parallelized(&activation_lsb, &activation_msb)
                    .into_blocks();
                let padding: RadixCiphertext = server_key.create_trivial_radix(0, (j >> 1).into());
                let padding_blocks = padding.into_blocks();
                ct_prod_blocks.extend(padding_blocks);
                RadixCiphertext::from_blocks(ct_prod_blocks.to_vec())
            })
            .collect();
        // Sum products
        let probability = server_key.add_parallelized(&prods[0], &prods[1]);
        let probability = server_key.add_parallelized(&probability, &prods[2]);
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
        let class = (ptxt_probability > quantize(0.5, precision, bit_width)) as usize;
        println!("[{}] predicted {:?}, target {:?}", num, class, target);
        if class == *target {
            total += 1;
        }
    }
    let accuracy = (total as f32 / num_samples as f32) * 100.0;
    println!("Accuracy {accuracy}%");
}
