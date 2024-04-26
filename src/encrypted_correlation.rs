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

fn main() {
    let (experience, salaries) = common::read_csv_two_columns("data/correlation.csv");
    assert_eq!(
        experience.len(),
        salaries.len(),
        "The length of the two arrays must be equal"
    );
    let dataset_size = salaries.len() as f64;

    let mut salary_sorted = salaries.clone();
    salary_sorted.sort();

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
    let encrypted_salaries: Vec<_> = salary_sorted
        .par_iter() // Use par_iter() for parallel iteration
        .map(|&salary| client_key.encrypt(salary))
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //
    // TODO: Move LUT gens up here

    // Compute the encrypted correlation
    let start = Instant::now();
    println!("Starting computing mean");

    let experience_mean = experience.iter().map(|&exp| exp as f64).sum::<f64>() / dataset_size;
    let experience_variance: f64 = experience
        .iter()
        .map(|&exp| ((exp as f64) - experience_mean).powi(2))
        .sum();
    let experience_stddev = experience_variance.sqrt();

    let mut salaries_sum_enc = encrypted_salaries.iter().fold(
        server_key.create_trivial_radix(0_u64, nb_blocks),
        |acc: RadixCiphertext, salary| server_key.add_parallelized(&acc, salary),
    );
    let div_lut =
        wopbs_key.generate_lut_radix(&salaries_sum_enc, |x: u64| x / (dataset_size as u64));
    salaries_sum_enc = wopbs_key.keyswitch_to_wopbs_params(&server_key, &salaries_sum_enc);
    let mut salaries_mean_enc = wopbs_key.wopbs(&salaries_sum_enc, &div_lut);
    salaries_mean_enc = wopbs_key.keyswitch_to_pbs_params(&salaries_mean_enc);

    println!(
        "Finished computing mean in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    let covariance = encrypted_salaries
        .iter()
        .zip(experience.iter())
        .map(|(salary_enc, &exp)| {
            let x = server_key.sub_parallelized(salary_enc, &salaries_mean_enc);
            server_key.scalar_mul_parallelized(&x, exp - (experience_mean as u32))
        })
        .fold(
            server_key.create_trivial_radix(0_u64, nb_blocks),
            |acc: RadixCiphertext, diff| server_key.add_parallelized(&acc, &diff),
        );

    let mut salaries_variance_enc = encrypted_salaries
        .iter()
        .map(|salary_enc| {
            let x = server_key.sub_parallelized(salary_enc, &salaries_mean_enc);
            server_key.mul_parallelized(&x, &x)
        })
        .fold(
            server_key.create_trivial_radix(0_u64, nb_blocks),
            |acc: RadixCiphertext, diff| server_key.add_parallelized(&acc, &diff),
        );

    let sqrt_lut = wopbs_key.generate_lut_radix(&salaries_variance_enc, |x: u64| x.sqrt());

    salaries_variance_enc =
        wopbs_key.keyswitch_to_wopbs_params(&server_key, &salaries_variance_enc);
    let mut salaries_stddev_enc = wopbs_key.wopbs(&salaries_variance_enc, &sqrt_lut);
    salaries_stddev_enc = wopbs_key.keyswitch_to_pbs_params(&salaries_stddev_enc);

    let divisor_enc =
        server_key.scalar_mul_parallelized(&salaries_stddev_enc, experience_stddev as u32);
    let correlation_enc = server_key.div_parallelized(&covariance, &divisor_enc);

    // ------- Client side ------- //
    let correlation: u64 = client_key.decrypt(&correlation_enc);

    println!("correlation: {}", correlation);
}
