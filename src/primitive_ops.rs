use std::time::Instant;

use libm::tanh;
use ripple::common::*;
use statrs::function::erf::erf;
use tfhe::{
    integer::{gen_keys_radix, wopbs::*},
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

/*
reductions for 32 bits
sqrt: 20 bits
reciprocal: 16 bits
log: 20 bits
inverse sqrt: 16 bits
erf: 18 bits
sigmoid: 16 bits
Truncation amount = 8 bits
tanh: 18 bits
Truncation amount = 6 bits

reductions for 24 bits
sqrt: 12 bits
Truncation amount = 6 bits
log: 15 bits
Truncation amount = 3 bits
inverse sqrt: 12 bits
Truncation amount = 6 bits
erf: 14 bits
Truncation amount = 4 bits
sigmoid: 12 bits
Truncation amount = 6 bits
tanh: 14 bits
Truncation amount = 4 bits
reciprocal: 12 bits
Truncation amount = 6 bits

reduction for 16 bits
sqrt: 10 bits
Truncation amount = 2 bits
log: 10 bits
Truncation amount = 2 bits
inverse sqrt: 8 bits
Truncation amount = 4 bits
erf: 9 bits
Truncation amount = 3 bits
sigmoid: 8 bits
Truncation amount = 4 bits
tanh: 9 bits
Truncation amount = 3 bits
reciprocal: 8 bits
Truncation amount = 4 bits
*/

fn main() {
    // ------- Client side ------- //
    let bit_width = 16; // W
    let precision = 12u8; // k
    let integer_size = 4u32;
    let wave_depth = bit_width >> 1; // J

    // Number of blocks per ciphertext
    let nb_blocks = bit_width >> 1;
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

    let x = quantize(2.0, precision, bit_width as u8);
    let x_ct = client_key.encrypt(x);
    let y_ct = client_key.encrypt(2_u64.pow(bit_width as u32) - 5_u64);
    // let y_ct = client_key.encrypt(5_u64);

    // ------- Server side ------- //

    // 1.1. Square root using LUT
    fn my_sqrt(value: f64) -> f64 {
        value.sqrt()
    }
    let (square_ct, lut_time) = ct_lut_eval(
        x_ct.clone(),
        precision,
        bit_width,
        &my_sqrt,
        &wopbs_key,
        &server_key,
    );
    let lut_prod: u64 = client_key.decrypt(&square_ct);
    println!("Square root (LUT) time: {:?}", lut_time);

    // 1.2. Square root using Haar DWT LUT
    let (square_ct_haar, dwt_time) = ct_lut_eval_haar(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &my_sqrt,
        &wopbs_key,
        &server_key,
    );
    let dwt_lut_prod: u64 = client_key.decrypt(&square_ct_haar);
    println!("Square root (Haar) time: {:?}", dwt_time);

    // 1.3 Square root using Quantized LUT
    let (square_ct_quant, lut_time_quant) = ct_lut_eval_quantized(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &my_sqrt,
        &wopbs_key,
        &server_key,
    );
    let lut_prod_quant: u64 = client_key.decrypt(&square_ct_quant);
    println!("Square root (Quantized LUT) time: {:?}", lut_time_quant);

    let (lsb_1, _msb_1) = bior(
        "./data/bior_lut_sqrt_16.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let (lsb_2, _msb_2) = bior(
        "./data/bior_lut_sqrt_16_2.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let luts = vec![&lsb_1, &lsb_2];

    // 1.4 Square root using Biorthogonal
    let (square_ct_bior, lut_time_bior) = ct_lut_eval_bior(
        x_ct.clone(),
        bit_width,
        nb_blocks,
        &luts,
        wave_depth,
        &wopbs_key,
        0_i32,
        &server_key,
    );
    let lut_sqrt_bior: u64 = client_key.decrypt(&square_ct_bior);
    println!("Square root (Bior LUT) time: {:?}", lut_time_bior);

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, Bior LUT: {:?}\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}, Bior LUT: {:?}",
        lut_prod,
        dwt_lut_prod,
        lut_prod_quant,
        lut_sqrt_bior >> 2,
        unquantize(lut_prod, precision, bit_width as u8),
        unquantize(dwt_lut_prod, precision, bit_width as u8),
        unquantize(lut_prod_quant, precision, bit_width as u8),
        unquantize(lut_sqrt_bior >> 2, precision, bit_width as u8),
    );

    // 2.1 1000/x using LUT
    fn scaled_inverse(value: f64) -> f64 {
        1000_f64 / value
    }
    let (inverse_ct, lut_time) = ct_lut_eval(
        x_ct.clone(),
        precision,
        bit_width,
        &scaled_inverse,
        &wopbs_key,
        &server_key,
    );
    let lut_inv: u64 = client_key.decrypt(&inverse_ct);
    println!("Scaled Inverse (LUT) time: {:?}", lut_time);

    // 2.2. 1000/x using Haar DWT LUT
    let (inverse_ct_haar, dwt_time) = ct_lut_eval_haar(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &scaled_inverse,
        &wopbs_key,
        &server_key,
    );
    let dwt_lut_inv: u64 = client_key.decrypt(&inverse_ct_haar);
    println!("Scaled Inverse (Haar) time: {:?}", dwt_time);

    // 2.3 1000/x using Quantized LUT
    let (inverse_ct_quant, lut_time_quant) = ct_lut_eval_quantized(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &scaled_inverse,
        &wopbs_key,
        &server_key,
    );
    let lut_inv_quant: u64 = client_key.decrypt(&inverse_ct_quant);
    println!("Scaled Inverse (Quantized LUT) time: {:?}", lut_time_quant);

    let (lsb_1, _msb_1) = bior(
        "./data/bior_lut_reciprocal_16.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let (lsb_2, _msb_2) = bior(
        "./data/bior_lut_reciprocal_16_2.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let luts = vec![&lsb_1, &lsb_2];

    // 2.4 1000/x using Biorthogonal
    let (inverse_ct_bior, lut_time_bior) = ct_lut_eval_bior(
        x_ct.clone(),
        bit_width,
        nb_blocks,
        &luts,
        wave_depth,
        &wopbs_key,
        0_i32,
        &server_key,
    );
    let lut_inverse_bior: u64 = client_key.decrypt(&inverse_ct_bior);
    println!("Scaled Inverse (Bior LUT) time: {:?}", lut_time_bior);

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, Bior LUT: {:?}\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}, Bior LUT: {:?}",
        lut_inv,
        dwt_lut_inv,
        lut_inv_quant,
        lut_inverse_bior >> 4,
        unquantize(lut_inv, precision, bit_width as u8),
        unquantize(dwt_lut_inv, precision, bit_width as u8),
        unquantize(lut_inv_quant, precision, bit_width as u8),
        unquantize(lut_inverse_bior >> 4, precision, bit_width as u8),
    );

    // 3.1 log2(x) using LUT
    fn my_log(value: f64) -> f64 {
        value.log2()
    }
    let (log_ct, lut_time) = ct_lut_eval(
        x_ct.clone(),
        precision,
        bit_width,
        &my_log,
        &wopbs_key,
        &server_key,
    );
    let lut_log: u64 = client_key.decrypt(&log_ct);
    println!("log2(x) (LUT) time: {:?}", lut_time);

    // 3.2. log2(x) using Haar DWT LUT
    let (log_ct_haar, dwt_time) = ct_lut_eval_haar(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &my_log,
        &wopbs_key,
        &server_key,
    );
    let dwt_lut_log: u64 = client_key.decrypt(&log_ct_haar);
    println!("log2(x) (Haar) time: {:?}", dwt_time);

    // 3.3 log2(x) using Quantized LUT
    let (log_ct_quant, lut_time_quant) = ct_lut_eval_quantized(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &my_log,
        &wopbs_key,
        &server_key,
    );
    let lut_log_quant: u64 = client_key.decrypt(&log_ct_quant);
    println!("log2(x) (Quantized LUT) time: {:?}", lut_time_quant);

    let (lsb_1, _msb_1) = bior(
        "./data/bior_lut_log_16.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let (lsb_2, _msb_2) = bior(
        "./data/bior_lut_log_16_2.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let luts = vec![&lsb_1, &lsb_2];

    // 3.4 log2(x) using Biorthogonal
    let (log_ct_bior, lut_time_bior) = ct_lut_eval_bior(
        x_ct.clone(),
        bit_width,
        nb_blocks,
        &luts,
        wave_depth,
        &wopbs_key,
        0_i32,
        &server_key,
    );
    let lut_log_bior: u64 = client_key.decrypt(&log_ct_bior);
    println!("log2(x) (Bior LUT) time: {:?}", lut_time_bior);

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, Bior LUT: {:?}\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}, Bior LUT: {:?}",
        lut_log,
        dwt_lut_log,
        lut_log_quant,
        lut_log_bior >> 2,
        unquantize(lut_log, precision, bit_width as u8),
        unquantize(dwt_lut_log, precision, bit_width as u8),
        unquantize(lut_log_quant, precision, bit_width as u8),
        unquantize(lut_log_bior >> 2, precision, bit_width as u8),
    );

    // // 4.1 sigmoid(x) using LUT
    fn sigmoid(value: f64) -> f64 {
        1f64 / (1f64 + (-value).exp())
    }
    fn sigmoid_naive_pt1(value: f64) -> f64 {
        -value.exp()
    }
    fn sigmoid_naive_pt2(value: f64) -> f64 {
        1f64 / (1f64 + value)
    }
    // let (sig_ct, lut_time_pt1) = ct_lut_eval(
    //     x_ct.clone(),
    //     precision,
    //     bit_width,
    //     &sigmoid_naive_pt1,
    //     &wopbs_key,
    //     &server_key,
    // );
    // let (sig_ct, lut_time_pt2) = ct_lut_eval(
    //     sig_ct,
    //     precision,
    //     bit_width,
    //     &sigmoid_naive_pt2,
    //     &wopbs_key,
    //     &server_key,
    // );
    let (sig_ct, lut_time_pt2) = ct_lut_eval(
        x_ct.clone(),
        precision,
        bit_width,
        &sigmoid,
        &wopbs_key,
        &server_key,
    );
    let lut_sig: u64 = client_key.decrypt(&sig_ct);
    println!("Sigmoid (LUT) time: {:?}", lut_time_pt2);

    // 4.2a sigmoid(x) using Haar DWT LUT
    let (sig_ct_haar, dwt_time) = ct_lut_eval_haar(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &sigmoid,
        &wopbs_key,
        &server_key,
    );
    let dwt_lut_sig: u64 = client_key.decrypt(&sig_ct_haar);
    println!("Sigmoid (Haar) time: {:?}", dwt_time);

    // 4.2b sigmoid(x) using Haar DWT LUT w/ bounded optimization
    let (sig_ct_haar, dwt_time) = ct_lut_eval_haar_bounded(
        x_ct.clone(),
        precision,
        bit_width,
        integer_size,
        nb_blocks,
        &sigmoid,
        &wopbs_key,
        &server_key,
        false,
        // &client_key,
    );
    let dwt_sig_bounded: u64 = client_key.decrypt(&sig_ct_haar);
    println!("Sigmoid (Haar) time (bounded): {:?}", dwt_time);

    // 4.3 sigmoid(x) using Quantized LUT
    let (sig_ct_quant, lut_time_quant_pt1) = ct_lut_eval_quantized(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &sigmoid_naive_pt1,
        &wopbs_key,
        &server_key,
    );
    let (sig_ct_quant, lut_time_quant_pt2) = ct_lut_eval(
        sig_ct_quant,
        precision,
        bit_width >> 1,
        &sigmoid_naive_pt2,
        &wopbs_key,
        &server_key,
    );
    let lut_sig_quant: u64 = client_key.decrypt(&sig_ct_quant);
    println!(
        "Sigmoid (Quantized LUT) time: {:?}",
        lut_time_quant_pt1 + lut_time_quant_pt2
    );

    let (lsb_1, _msb_1) = bior(
        "./data/bior_lut_sigmoid_16.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let (lsb_2, _msb_2) = bior(
        "./data/bior_lut_sigmoid_16_2.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let luts = vec![&lsb_1, &lsb_2];

    // 4.4 sigmoid(x) using Biorthogonal
    let (sig_ct_bior, lut_time_bior) = ct_lut_eval_bior(
        x_ct.clone(),
        bit_width,
        nb_blocks,
        &luts,
        wave_depth,
        &wopbs_key,
        -128_i32,
        &server_key,
    );
    let lut_sig_bior: u64 = client_key.decrypt(&sig_ct_bior);
    println!("Sigmoid (Bior LUT) time: {:?}", lut_time_bior);

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Bounded DWT LUT: {:?}, Quant LUT: {:?}, Bior LUT: {:?}\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Bounded DWT LUT: {:?}, Quant LUT {:?}, Bior LUT: {:?}",
        lut_sig,
        dwt_lut_sig,
        dwt_sig_bounded,
        lut_sig_quant,
        lut_sig_bior >> 4,
        unquantize(lut_sig, precision, bit_width as u8),
        unquantize(dwt_lut_sig, precision, bit_width as u8),
        unquantize(dwt_sig_bounded, precision, bit_width as u8),
        unquantize(lut_sig_quant, precision, bit_width as u8),
        unquantize(lut_sig_bior >> 4, precision, bit_width as u8),
    );

    // 5.1 1000/sqrt(x) using LUT
    fn inv(value: f64) -> f64 {
        1000_f64 / value
    }
    fn inv_sqrt(value: f64) -> f64 {
        1000_f64 / value.sqrt()
    }
    let (inv_sqrt_ct, lut_time_pt1) = ct_lut_eval(
        x_ct.clone(),
        precision,
        bit_width,
        &my_sqrt,
        &wopbs_key,
        &server_key,
    );
    let (inv_sqrt_ct, lut_time_pt2) = ct_lut_eval(
        inv_sqrt_ct,
        precision,
        bit_width,
        &inv,
        &wopbs_key,
        &server_key,
    );
    let lut_inv_sqrt: u64 = client_key.decrypt(&inv_sqrt_ct);
    println!(
        "Inverse Square Root (LUT) time: {:?}",
        lut_time_pt1 + lut_time_pt2
    );

    // 5.2. 1000/sqrt(x) using Haar DWT LUT
    let (inv_sqrt_ct_haar, dwt_time) = ct_lut_eval_haar(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &inv_sqrt,
        &wopbs_key,
        &server_key,
    );
    let dwt_inv_sqrt: u64 = client_key.decrypt(&inv_sqrt_ct_haar);
    println!("Inverse Square Root (Haar) time: {:?}", dwt_time);

    // 5.3 1000/sqrt(x) using Quantized LUT
    let (inv_sqrt_ct_quant, lut_time_quant_pt1) = ct_lut_eval_quantized(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &my_sqrt,
        &wopbs_key,
        &server_key,
    );
    let (inv_sqrt_ct_quant, lut_time_quant_pt2) = ct_lut_eval(
        inv_sqrt_ct_quant,
        precision,
        bit_width >> 1,
        &inv,
        &wopbs_key,
        &server_key,
    );
    let lut_inv_sqrt_quant: u64 = client_key.decrypt(&inv_sqrt_ct_quant);
    println!(
        "Inverse Square Root (Quantized LUT) time: {:?}",
        lut_time_quant_pt1 + lut_time_quant_pt2
    );

    let (lsb_1, _msb_1) = bior(
        "./data/bior_lut_inv_sqrt_16.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let (lsb_2, _msb_2) = bior(
        "./data/bior_lut_inv_sqrt_16_2.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let luts = vec![&lsb_1, &lsb_2];

    // 5.4 1000/sqrt(x) using Biorthogonal
    let (inv_sqrt_ct_bior, lut_time_bior) = ct_lut_eval_bior(
        x_ct.clone(),
        bit_width,
        nb_blocks,
        &luts,
        wave_depth,
        &wopbs_key,
        0_i32,
        &server_key,
    );
    let lut_inv_sqrt_bior: u64 = client_key.decrypt(&inv_sqrt_ct_bior);
    println!("Inverse Square Root (Bior LUT) time: {:?}", lut_time_bior);

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, Bior LUT: {:?}\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}, Bior LUT: {:?}",
        lut_inv_sqrt,
        dwt_inv_sqrt,
        lut_inv_sqrt_quant,
        lut_inv_sqrt_bior >> 4,
        unquantize(lut_inv_sqrt, precision, bit_width as u8),
        unquantize(dwt_inv_sqrt, precision, bit_width as u8),
        unquantize(lut_inv_sqrt_quant, precision, bit_width as u8),
        unquantize(lut_inv_sqrt_bior >> 4, precision, bit_width as u8),
    );

    // 6.1 ERF using LUT
    let (erf_ct, lut_time) = ct_lut_eval(
        x_ct.clone(),
        precision,
        bit_width,
        &erf,
        &wopbs_key,
        &server_key,
    );
    let lut_erf: u64 = client_key.decrypt(&erf_ct);
    println!("ERF (LUT) time: {:?}", lut_time);

    // 6.2 ERF using Haar DWT LUT
    let (erf_ct_haar, dwt_time) = ct_lut_eval_haar(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &erf,
        &wopbs_key,
        &server_key,
    );
    let dwt_erf: u64 = client_key.decrypt(&erf_ct_haar);
    println!("ERF (Haar) time: {:?}", dwt_time);

    // 6.3 ERF using Quantized LUT
    let (erf_ct_quant, lut_time_quant) = ct_lut_eval_quantized(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &erf,
        &wopbs_key,
        &server_key,
    );
    let lut_erf_quant: u64 = client_key.decrypt(&erf_ct_quant);
    println!("ERF (Quantized LUT) time: {:?}", lut_time_quant);

    let (lsb_1, _msb_1) = bior(
        "./data/bior_lut_erf_16.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let (lsb_2, _msb_2) = bior(
        "./data/bior_lut_erf_16_2.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let luts = vec![&lsb_1, &lsb_2];

    // 6.4 ERF using Biorthogonal
    let (erf_ct_bior, lut_time_bior) = ct_lut_eval_bior(
        x_ct.clone(),
        bit_width,
        nb_blocks,
        &luts,
        wave_depth,
        &wopbs_key,
        -128_i32,
        &server_key,
    );
    let lut_erf_bior: u64 = client_key.decrypt(&erf_ct_bior);
    println!("ERF (Bior LUT) time: {:?}", lut_time_bior);

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, Bior LUT: {:?}\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}, Bior LUT: {:?}",
        lut_erf,
        dwt_erf,
        lut_erf_quant,
        lut_erf_bior >> 2,
        unquantize(lut_erf, precision, bit_width as u8),
        unquantize(dwt_erf, precision, bit_width as u8),
        unquantize(lut_erf_quant, precision, bit_width as u8),
        unquantize(lut_erf_bior >> 2, precision, bit_width as u8),
    );

    // 7.1 tanh(x) using LUT
    let (tanh_ct, lut_time) = ct_lut_eval(
        x_ct.clone(),
        precision,
        bit_width,
        &tanh,
        &wopbs_key,
        &server_key,
    );
    let lut_tanh: u64 = client_key.decrypt(&tanh_ct);
    println!("Tanh (LUT) time: {:?}", lut_time);

    // 7.2a tanh(x) using Haar DWT LUT
    let (tanh_ct_haar, dwt_time) = ct_lut_eval_haar(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &tanh,
        &wopbs_key,
        &server_key,
    );
    let dwt_tanh: u64 = client_key.decrypt(&tanh_ct_haar);
    println!("Tanh (Haar) time: {:?}", dwt_time);

    // 7.2b tanh(x) using Haar DWT LUT w/ bounded optimization
    let (tanh_ct_haar, dwt_time) = ct_lut_eval_haar_bounded(
        y_ct.clone(),
        precision,
        bit_width,
        integer_size,
        nb_blocks,
        &tanh,
        &wopbs_key,
        &server_key,
        true,
        // &client_key,
    );
    let dwt_tanh_bounded: u64 = client_key.decrypt(&tanh_ct_haar);
    println!("Tanh (Haar) time (bounded): {:?}", dwt_time);

    // 7.3 tanh(x) using Quantized LUT
    let (tanh_ct_quant, lut_time_quant) = ct_lut_eval_quantized(
        x_ct.clone(),
        precision,
        bit_width,
        nb_blocks,
        &tanh,
        &wopbs_key,
        &server_key,
    );
    let lut_tanh_quant: u64 = client_key.decrypt(&tanh_ct_quant);
    println!("Tanh (Quantized LUT) time: {:?}", lut_time_quant);

    let (lsb_1, _msb_1) = bior(
        "./data/bior_lut_tanh_16.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let (lsb_2, _msb_2) = bior(
        "./data/bior_lut_tanh_16_2.json",
        wave_depth as u8,
        bit_width as u8,
    );
    let luts = vec![&lsb_1, &lsb_2];

    // 7.4 tanh(x) using Biorthogonal
    let (tanh_ct_bior, lut_time_bior) = ct_lut_eval_bior(
        x_ct.clone(),
        bit_width,
        nb_blocks,
        &luts,
        wave_depth,
        &wopbs_key,
        -128_i32,
        &server_key,
    );
    let lut_tanh_bior: u64 = client_key.decrypt(&tanh_ct_bior);
    println!("Tanh (Bior LUT) time: {:?}", lut_time_bior);

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Bounded DWT LUT: {:?}, Quant LUT: {:?}, Bior LUT: {:?}\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Bounded DWT LUT: {:?}, Quant LUT {:?}, Bior LUT: {:?}",
        lut_tanh,
        dwt_tanh,
        dwt_tanh_bounded,
        lut_tanh_quant,
        lut_tanh_bior >> 2,
        unquantize(lut_tanh, precision, bit_width as u8),
        unquantize(dwt_tanh, precision, bit_width as u8),
        unquantize(dwt_tanh_bounded, precision, bit_width as u8),
        unquantize(lut_tanh_quant, precision, bit_width as u8),
        unquantize(lut_tanh_bior >> 2, precision, bit_width as u8),
    );
}
