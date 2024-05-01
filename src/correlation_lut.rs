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
    let data = common::read_csv("data/correlation.csv");
    let experience = &data[0];
    let salaries = &data[1];
    let dataset_size = salaries.len() as f64;

    let mut salaries_sorted = salaries.clone();
    salaries_sorted.sort();

    // ------- Client side ------- //
    let bit_width = 20;

    // Number of blocks per ciphertext
    let nb_blocks = bit_width / 2;
    println!(
        "Number of blocks for the radix decomposition: {:?}",
        nb_blocks
    );

    // Scale factor
    let scale = 100_000_u64;

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
    let encrypted_salaries: Vec<_> = salaries_sorted
        .par_iter() // Use par_iter() for parallel iteration
        .map(|&salary| client_key.encrypt(salary))
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //

    // The experience vector is known to the server.
    let experience_mean = experience.iter().map(|&exp| exp as f64).sum::<f64>() / dataset_size;
    let experience_variance: f64 = experience
        .iter()
        .map(|&exp| ((exp as f64) - experience_mean).powi(2))
        .sum();
    let experience_stddev = experience_variance.sqrt();

    // Offline: LUT genaration is offline cost.
    let mut dummy_ct: RadixCiphertext = server_key.create_trivial_radix(0_u64, nb_blocks);
    let dummy_ct_2 = server_key.create_trivial_radix(0_u64, nb_blocks);
    for _ in 0..encrypted_salaries.len() {
        dummy_ct = server_key.add_parallelized(&dummy_ct, &dummy_ct_2);
    }
    let div_lut = wopbs_key.generate_lut_radix(&dummy_ct, |x: u64| x / (dataset_size as u64));
    dummy_ct = wopbs_key.keyswitch_to_wopbs_params(&server_key, &dummy_ct);
    dummy_ct = wopbs_key.wopbs(&dummy_ct, &div_lut);
    dummy_ct = wopbs_key.keyswitch_to_pbs_params(&dummy_ct);
    let mut dummy_acc: RadixCiphertext = server_key.create_trivial_radix(0_u64, nb_blocks);
    for _ in 0..encrypted_salaries.len() {
        let mut dummy_ct_3 = server_key.sub_parallelized(&dummy_ct_2, &dummy_ct);
        dummy_ct_3 = server_key.mul_parallelized(&dummy_ct_3, &dummy_ct_3);
        dummy_acc = server_key.add_parallelized(&dummy_acc, &dummy_ct_3);
    }
    let sqrt_lut = wopbs_key.generate_lut_radix(&dummy_acc, |x: u64| {
        if x == 0 {
            1 // avoid division with zero error.
        } else {
            scale / (x.sqrt() * experience_stddev as u64)
        }
    });

    // Online: Compute the encrypted correlation

    // The salaries vector is encrypted.
    println!("- Starting computing mean");
    let start = Instant::now();
    let total = start;
    let mut salaries_sum_enc = encrypted_salaries.iter().fold(
        server_key.create_trivial_radix(0_u64, nb_blocks),
        |acc: RadixCiphertext, salary| server_key.add_parallelized(&acc, salary),
    );
    salaries_sum_enc = wopbs_key.keyswitch_to_wopbs_params(&server_key, &salaries_sum_enc);
    let mut salaries_mean_enc = wopbs_key.wopbs(&salaries_sum_enc, &div_lut);
    salaries_mean_enc = wopbs_key.keyswitch_to_pbs_params(&salaries_mean_enc);
    println!(
        "- Finished computing mean in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // Cov = Sum_i^n (salary_i - mean(salary))(experience_i - mean(experience))
    println!("- Starting computing covariance");
    let start = Instant::now();
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
    println!(
        "- Finished computing covariance in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // Var_salary = Sum_i^n (salary_i - mean(salary))^2
    println!("- Starting computing variance");
    let start = Instant::now();
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
    println!(
        "- Finished computing variance in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // sigma_salary (or stddev) = sqrt(var_salary)
    salaries_variance_enc =
        wopbs_key.keyswitch_to_wopbs_params(&server_key, &salaries_variance_enc);
    println!("- Starting computing LUT");
    let start = Instant::now();
    let mut salaries_stddev_enc = wopbs_key.wopbs(&salaries_variance_enc, &sqrt_lut);
    salaries_stddev_enc = wopbs_key.keyswitch_to_pbs_params(&salaries_stddev_enc);
    println!(
        "- Finished computing LUT in {:?} sec.",
        start.elapsed().as_secs_f64()
    );
    let correlation_enc = server_key.mul_parallelized(&salaries_stddev_enc, &covariance);

    println!(
        "Finished computing correlation in {:?} sec.",
        total.elapsed().as_secs_f64()
    );

    // ------- Client side ------- //
    let correlation: u64 = client_key.decrypt(&correlation_enc);

    println!("Correlation: {}", (correlation as f64) / (scale as f64));
}
