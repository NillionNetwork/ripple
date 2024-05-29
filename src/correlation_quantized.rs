use std::time::Instant;

use num_integer::Roots;
use rayon::prelude::*;
use ripple::common::*;
use tfhe::{
    integer::{
        ciphertext::BaseRadixCiphertext, gen_keys_radix, wopbs::*, IntegerCiphertext,
        IntegerRadixCiphertext, RadixCiphertext,
    },
    shortint::{
        parameters::{
            parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS, Degree,
            PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        },
        Ciphertext,
    },
};

fn main() {
    let data = read_csv("data/correlation.csv");
    let experience = &data[0];
    let salaries = &data[1];
    let dataset_size = salaries.len() as f64;

    let mut salaries_sorted = salaries.clone();
    salaries_sorted.sort();

    // ------- Client side ------- //
    let bit_width = 16;

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
    let lut_gen_start = Instant::now();
    println!("Generating LUT.");
    let mut dummy: RadixCiphertext = server_key.create_trivial_radix(2_u64, nb_blocks);
    dummy = wopbs_key.keyswitch_to_wopbs_params(&server_key, &dummy);
    let mut dummy_blocks = dummy.clone().into_blocks().to_vec();
    for block in &mut dummy_blocks {
        block.degree = Degree::new(3);
    }
    dummy = RadixCiphertext::from_blocks(dummy_blocks[0..(nb_blocks >> 1)].to_vec());
    let sqrt_lut = wopbs_key.generate_lut_radix(&dummy, |x: u64| {
        if x == 0 {
            1 // avoid division with zero error.
        } else {
            scale / (x.sqrt() * experience_stddev as u64)
        }
    });
    let div_lut = wopbs_key.generate_lut_radix(&dummy, |x: u64| x / (dataset_size as u64));
    println!(
        "LUT generation done in {:?} sec.",
        lut_gen_start.elapsed().as_secs_f64()
    );
    // Online: Compute the encrypted correlation

    // The salaries vector is encrypted.
    let start = Instant::now();
    let total = start;
    let salaries_sum_enc = encrypted_salaries.iter().fold(
        server_key.create_trivial_radix(0_u64, nb_blocks),
        |acc: RadixCiphertext, salary| server_key.add_parallelized(&acc, salary),
    );
    // println!("salaries_sum_enc degree: {:?}", salaries_sum_enc.blocks()[0].degree);
    let salaries_mean_enc = ct_lut_eval_quantized_no_gen(
        salaries_sum_enc,
        nb_blocks,
        &wopbs_key,
        &server_key,
        &div_lut,
    );
    let mut salaries_mean_enc_blocks = salaries_mean_enc.clone().blocks().to_vec();
    let zero_ct: BaseRadixCiphertext<Ciphertext> =
        server_key.create_trivial_radix(0_u64, nb_blocks >> 1);
    let zero_ct_blocks = zero_ct.clone().blocks().to_vec();
    salaries_mean_enc_blocks.extend(zero_ct_blocks.clone());
    let salaries_mean_enc = RadixCiphertext::from_blocks(salaries_mean_enc_blocks);
    // Cov = Sum_i^n (salary_i - mean(salary))(experience_i - mean(experience))
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

    // Var_salary = Sum_i^n (salary_i - mean(salary))^2
    let salaries_variance_enc = encrypted_salaries
        .iter()
        .map(|salary_enc| {
            let x = server_key.sub_parallelized(salary_enc, &salaries_mean_enc);
            server_key.mul_parallelized(&x, &x)
        })
        .fold(
            server_key.create_trivial_radix(0_u64, nb_blocks),
            |acc: RadixCiphertext, diff| server_key.add_parallelized(&acc, &diff),
        );

    // sigma_salary (or stddev) = sqrt(var_salary)
    // println!("salaries_variance_enc degree: {:?}",
    // salaries_variance_enc.blocks()[0].degree);
    let salaries_stddev_enc = ct_lut_eval_quantized_no_gen(
        salaries_variance_enc,
        nb_blocks,
        &wopbs_key,
        &server_key,
        &sqrt_lut,
    );
    let mut salaries_stddev_enc_blocks = salaries_stddev_enc.clone().blocks().to_vec();
    salaries_stddev_enc_blocks.extend(zero_ct_blocks);
    let salaries_stddev_enc = RadixCiphertext::from_blocks(salaries_stddev_enc_blocks);
    let correlation_enc = server_key.mul_parallelized(&salaries_stddev_enc, &covariance);

    println!(
        "Finished computing correlation in {:?} sec.",
        total.elapsed().as_secs_f64()
    );

    // ------- Client side ------- //
    let correlation: u64 = client_key.decrypt(&correlation_enc);

    println!("Correlation: {}", (correlation as f64) / (scale as f64));
}
