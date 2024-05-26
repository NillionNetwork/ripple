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
    let data = read_csv("data/correlation.csv");
    let experience = &data[0];
    let salaries = &data[1];
    let dataset_size = salaries.len() as f64;

    let mut salaries_sorted = salaries.clone();
    salaries_sorted.sort();

    // ------- Client side ------- //
    let bit_width = 16;
    let precision = 12;
    let wave_depth = 8;

    // TODO: Replace with actual custom functions
    let (sqrt_lut_lsb, sqrt_lut_msb) =
        bior("data/bior_lut_sqrt_16.json", wave_depth as u8, bit_width);
    let (sqrt_lut_lsb_2, sqrt_lut_msb_2) =
        bior("data/bior_lut_sqrt_16_2.json", wave_depth as u8, bit_width);
    let (div_lut_lsb, div_lut_msb) = bior("data/bior_lut_div_16.json", wave_depth as u8, bit_width);
    let (div_lut_lsb_2, div_lut_msb_2) =
        bior("data/bior_lut_div_16_2.json", wave_depth as u8, bit_width);

    let sqrt_luts = vec![
        &sqrt_lut_lsb,
        &sqrt_lut_lsb_2,
        &sqrt_lut_msb,
        &sqrt_lut_msb_2,
    ];
    let div_luts = vec![&div_lut_lsb, &div_lut_lsb_2, &div_lut_msb, &div_lut_msb_2];

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
    let (client_key, server_key) =
        gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, nb_blocks as usize);
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
        .map(|&salary| client_key.encrypt(quantize(salary as f64, precision, bit_width as u8)))
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //

    // The experience vector is known to the server.
    let experience_mean = experience.iter().map(|&exp| exp as f64).sum::<f64>() / dataset_size;
    // let experience_variance: f64 = experience
    //     .iter()
    //     .map(|&exp| ((exp as f64) - experience_mean).powi(2))
    //     .sum();
    // let experience_stddev = experience_variance.sqrt();

    // Offline: LUT genaration is offline cost.
    let lut_gen_start = Instant::now();
    println!("Generating LUT.");
    let mut dummy: RadixCiphertext =
        server_key.create_trivial_radix(2_u64, (nb_blocks >> 1).into());
    dummy = wopbs_key.keyswitch_to_wopbs_params(&server_key, &dummy);
    let mut dummy_blocks = dummy.clone().into_blocks().to_vec();
    for block in &mut dummy_blocks {
        block.degree = Degree::new(3);
    }
    dummy = RadixCiphertext::from_blocks(dummy_blocks);
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
    // Online: Compute the encrypted correlation

    // The salaries vector is encrypted.
    let start = Instant::now();
    let total = start;
    let salaries_sum_enc = encrypted_salaries.iter().fold(
        server_key.create_trivial_radix(0_u64, nb_blocks as usize),
        |acc: RadixCiphertext, salary| server_key.add_parallelized(&acc, salary),
    );
    // println!("salaries_sum_enc degree: {:?}", salaries_sum_enc.blocks()[0].degree);
    let salaries_mean_enc = ct_lut_eval_bior_no_gen(
        salaries_sum_enc,
        bit_width as usize,
        nb_blocks as usize,
        wave_depth,
        &wopbs_key,
        0_i32,
        &server_key,
        &encoded_div_luts,
    );

    // Cov = Sum_i^n (salary_i - mean(salary))(experience_i - mean(experience))
    let covariance = encrypted_salaries
        .iter()
        .zip(experience.iter())
        .map(|(salary_enc, &exp)| {
            let x = server_key.sub_parallelized(salary_enc, &salaries_mean_enc);
            server_key.scalar_mul_parallelized(&x, exp - (experience_mean as u32))
        })
        .fold(
            server_key.create_trivial_radix(0_u64, nb_blocks as usize),
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
            server_key.create_trivial_radix(0_u64, nb_blocks as usize),
            |acc: RadixCiphertext, diff| server_key.add_parallelized(&acc, &diff),
        );

    // sigma_salary (or stddev) = sqrt(var_salary)
    // println!("salaries_variance_enc degree: {:?}", salaries_variance_enc.blocks()[0].degree);
    let salaries_stddev_enc = ct_lut_eval_bior_no_gen(
        salaries_variance_enc,
        bit_width as usize,
        nb_blocks as usize,
        wave_depth,
        &wopbs_key,
        0_i32,
        &server_key,
        &encoded_sqrt_luts,
    );
    let correlation_enc = server_key.mul_parallelized(&salaries_stddev_enc, &covariance);

    println!(
        "Finished computing correlation in {:?} sec.",
        total.elapsed().as_secs_f64()
    );

    // ------- Client side ------- //
    let correlation: u64 = client_key.decrypt(&correlation_enc);
    let correlation_final: f64 = unquantize(correlation, precision, bit_width as u8);

    println!("Correlation: {}", correlation_final / (scale as f64));
}
