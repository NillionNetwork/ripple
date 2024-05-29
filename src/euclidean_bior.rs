use std::time::Instant;

use rayon::prelude::*;
use ripple::common::*;
use tfhe::{
    integer::{
        gen_keys_radix, wopbs::*, IntegerCiphertext, IntegerRadixCiphertext, RadixCiphertext,
    },
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS, Degree,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

fn main() {
    let data = read_csv("data/euclidean.csv");
    let xs = &data[0];
    let num_iter = 3;

    // ------- Client side ------- //
    let bit_width = 16;
    let precision = 12;
    let j = 8;

    let (sqrt_lut_lsb, sqrt_lut_msb) = bior("data/bior_lut_sqrt_16.json", j as u8, bit_width);
    let (sqrt_lut_lsb_2, sqrt_lut_msb_2) = bior("data/bior_lut_sqrt_16_2.json", j as u8, bit_width);
    let (div_lut_lsb, div_lut_msb) = bior("data/bior_lut_div_16.json", j as u8, bit_width);
    let (div_lut_lsb_2, div_lut_msb_2) = bior("data/bior_lut_div_16_2.json", j as u8, bit_width);

    let sqrt_luts = [
        &sqrt_lut_lsb,
        &sqrt_lut_lsb_2,
        &sqrt_lut_msb,
        &sqrt_lut_msb_2,
    ];
    let div_luts = [&div_lut_lsb, &div_lut_lsb_2, &div_lut_msb, &div_lut_msb_2];

    // Number of blocks per ciphertext
    let nb_blocks = bit_width / 2;
    println!(
        "Number of blocks for the radix decomposition: {:?}",
        nb_blocks
    );

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

    let start = Instant::now();
    let xs_enc: Vec<_> = xs
        .par_iter() // Use par_iter() for parallel iteration
        .map(|&x| client_key.encrypt(quantize(x as f64, precision, bit_width)))
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //
    let lut_gen_start = Instant::now();
    println!("Generating LUT.");
    let dummy: RadixCiphertext = server_key.create_trivial_radix(0_u64, j >> 1);
    let mut dummy_blocks = dummy.into_blocks().to_vec();
    for block in &mut dummy_blocks {
        block.degree = Degree::new(3);
    }
    let dummy = RadixCiphertext::from_blocks(dummy_blocks);
    let dummy = wopbs_key.keyswitch_to_wopbs_params(&server_key, &dummy);
    let encoded_sqrt_luts = sqrt_luts
        .iter()
        .map(|lut| wopbs_key.generate_lut_radix(&dummy, |x: u64| eval_lut(x, &lut.to_vec())))
        .collect::<Vec<_>>();
    let encoded_div_luts = div_luts
        .iter()
        .map(|lut| wopbs_key.generate_lut_radix(&dummy, |x: u64| eval_lut(x, &lut.to_vec())))
        .collect::<Vec<_>>();
    println!(
        "LUT generation done in {:?} sec.",
        lut_gen_start.elapsed().as_secs_f64()
    );

    assert!(
        num_iter < data.len(),
        "Not enough columns in CSV for that many iterations"
    );

    let bench_start = Instant::now();
    let sum_dists = (1..num_iter + 1)
        .into_par_iter()
        .map(|i| {
            let ys = &data[i];
            // Compute the encrypted euclidean distance

            let start = Instant::now();
            println!("{}) Starting computing Squared Euclidean distance", i);

            let euclid_squared_enc = xs_enc
                .iter()
                .zip(ys.iter())
                .map(|(x_enc, &y)| {
                    let diff = server_key.scalar_sub_parallelized(x_enc, y);
                    server_key.mul_parallelized(&diff, &diff)
                })
                .fold(
                    server_key.create_trivial_radix(0_u64, nb_blocks.into()),
                    |acc: RadixCiphertext, diff| server_key.add_parallelized(&acc, &diff),
                );
            println!(
                "{}) Finished computing Squared Euclidean distance in {:?} sec.",
                i,
                start.elapsed().as_secs_f64()
            );
            // println!("euclid_squared_enc degree: {:?}", euclid_squared_enc.blocks()[0].degree);
            println!("{}) Starting computing square root", i);
            let distance_enc = ct_lut_eval_bior_no_gen(
                euclid_squared_enc,
                bit_width.into(),
                nb_blocks.into(),
                j,
                &wopbs_key,
                0_i32,
                &server_key,
                &encoded_sqrt_luts,
            );
            println!(
                "{}) Finished computing square root in {:?} sec.",
                i,
                start.elapsed().as_secs_f64()
            );

            distance_enc
        })
        .collect::<Vec<_>>()
        .into_iter()
        .fold(
            server_key.create_trivial_radix(0_u64, nb_blocks.into()),
            |acc: RadixCiphertext, diff| server_key.add_parallelized(&acc, &diff),
        );

    // println!("sum_dists degree: {:?}", sum_dists.blocks()[0].degree);
    let dists_mean_enc = ct_lut_eval_bior_no_gen(
        sum_dists,
        bit_width.into(),
        nb_blocks.into(),
        j,
        &wopbs_key,
        0_i32,
        &server_key,
        &encoded_div_luts,
    );
    println!(
        "Finished computing everything in {:?} sec.",
        bench_start.elapsed().as_secs_f64()
    );

    // ------- Client side ------- //
    let mean_distance: u64 = client_key.decrypt(&dists_mean_enc);
    let mean_distance: f64 = unquantize(mean_distance, precision, bit_width);

    println!(
        "Mean of {} Euclidean distances: {}",
        num_iter, mean_distance
    );
}
