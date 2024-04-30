use std::time::Instant;

use num_integer::Roots;
use rayon::prelude::*;
use ripple::common;
use tfhe::{
    integer::{gen_keys_radix, wopbs::*, RadixCiphertext},
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

/// d(x, y) = sqrt( sum((xi - yi)^2) )
fn euclidean(x: &[u32], y: &[u32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).pow(2) as f32)
        .sum::<f32>()
        .sqrt()
}

fn main() {
    let data = common::read_csv("data/euclidean.csv");
    let xs = &data[0];

    // ------- Client side ------- //
    let bit_width = 16;

    // Number of blocks per ciphertext
    let nb_blocks = bit_width / 2;
    println!(
        "Number of blocks for the radix decomposition: {:?}",
        nb_blocks
    );

    let start = Instant::now();
    // Generate radix keys
    let (client_key, server_key) = gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, nb_blocks);
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
        .map(|&x| client_key.encrypt(x))
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //
    // TODO: Move LUT gens up here

    let num_iter = 3;
    assert!(
        num_iter < data.len(),
        "Not enough columns in CSV for that many iterations"
    );

    let mut sum_dists = (1..num_iter + 1)
        .into_par_iter()
        .map(|i| {
            let ys = &data[i];

            let distance = euclidean(xs, ys);
            println!("{}) Ptxt Euclidean distance: {}", i, distance);

            // Compute the encrypted euclidean distance

            let start = Instant::now();
            println!("{}) Starting computing Squared Euclidean distance", i);

            let mut euclid_squared_enc = xs_enc
                .iter()
                .zip(ys.iter())
                .map(|(x_enc, &y)| {
                    let diff = server_key.scalar_sub_parallelized(x_enc, y);
                    server_key.mul_parallelized(&diff, &diff)
                })
                .fold(
                    server_key.create_trivial_radix(0_u64, nb_blocks),
                    |acc: RadixCiphertext, diff| server_key.add_parallelized(&acc, &diff),
                );
            println!(
                "{}) Finished computing Squared Euclidean distance in {:?} sec.",
                i,
                start.elapsed().as_secs_f64()
            );

            println!("{}) Starting computing square root", i);
            let sqrt_lut = wopbs_key.generate_lut_radix(&euclid_squared_enc, |x: u64| x.sqrt());
            euclid_squared_enc =
                wopbs_key.keyswitch_to_wopbs_params(&server_key, &euclid_squared_enc);
            let mut distance_enc = wopbs_key.wopbs(&euclid_squared_enc, &sqrt_lut);
            distance_enc = wopbs_key.keyswitch_to_pbs_params(&distance_enc);
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
            server_key.create_trivial_radix(0_u64, nb_blocks),
            |acc: RadixCiphertext, diff| server_key.add_parallelized(&acc, &diff),
        );

    let div_lut = wopbs_key.generate_lut_radix(&sum_dists, |x: u64| x / (num_iter as u64));
    sum_dists = wopbs_key.keyswitch_to_wopbs_params(&server_key, &sum_dists);
    let mut dists_mean_enc = wopbs_key.wopbs(&sum_dists, &div_lut);
    dists_mean_enc = wopbs_key.keyswitch_to_pbs_params(&dists_mean_enc);

    // ------- Client side ------- //
    let mean_distance: u64 = client_key.decrypt(&dists_mean_enc);
    println!(
        "Mean of {} Euclidean distances: {}",
        num_iter, mean_distance
    );
}
