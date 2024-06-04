use std::time::Instant;

use num_integer::Roots;
use rayon::prelude::*;
use ripple::common::{self, ct_lut_eval_quantized_no_gen};
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
    let data = common::read_csv("data/euclidean.csv");
    let xs = &data[0];
    let num_iter = 1;

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
    let lut_gen_start = Instant::now();
    println!("Generating LUT.");
    let mut dummy: RadixCiphertext = server_key.create_trivial_radix(2_u64, nb_blocks);
    dummy = wopbs_key.keyswitch_to_wopbs_params(&server_key, &dummy);
    let mut dummy_blocks = dummy.clone().into_blocks().to_vec();
    for block in &mut dummy_blocks {
        block.degree = Degree::new(3);
    }
    dummy = RadixCiphertext::from_blocks(dummy_blocks[0..(nb_blocks >> 1)].to_vec());
    let sqrt_lut = wopbs_key.generate_lut_radix(&dummy, |x: u64| x.sqrt());
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
            println!("Starting computing Squared Euclidean distance");

            let euclid_squared_enc = xs_enc
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
                "Finished computing Squared Euclidean distance in {:?} sec.",
                start.elapsed().as_secs_f64()
            );
            println!("Starting computing square root");
            let distance_enc = ct_lut_eval_quantized_no_gen(
                euclid_squared_enc,
                nb_blocks,
                &wopbs_key,
                &server_key,
                &sqrt_lut,
            );
            println!(
                "Finished computing square root in {:?} sec.",
                start.elapsed().as_secs_f64()
            );

            distance_enc
        })
        .collect::<Vec<_>>();
    println!(
        "Finished computing Euclidean distance in {:?} sec.",
        bench_start.elapsed().as_secs_f64()
    );
    // ------- Client side ------- //
    let mean_distance: u64 = client_key.decrypt(&sum_dists[0]);
    println!("Euclidean distance: {}", mean_distance);
}
