use std::time::Instant;

use libm::tanh;
use ripple::common::*;
use statrs::function::erf::erf;
use tfhe::{
    integer::{
        gen_keys_radix, wopbs::*, IntegerCiphertext, IntegerRadixCiphertext, RadixCiphertext,
        ServerKey,
    },
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS, Degree,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

fn eval_lut(x: u64, lut_map: &Vec<u64>) -> u64 {
    lut_map[x as usize]
}

fn ct_lut_eval(
    ct: RadixCiphertext,
    precision: u8,
    bit_width: usize,
    func: &dyn Fn(f64) -> f64,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
) -> (RadixCiphertext, f64) {
    let func_lut = wopbs_key.generate_lut_radix(&ct, |x: u64| {
        let x_unquantized = unquantize(x, precision, bit_width as u8);
        quantize(func(x_unquantized), precision, bit_width as u8)
    });
    let start = Instant::now();
    let ct_ks = wopbs_key.keyswitch_to_wopbs_params(server_key, &ct);
    let mut lut_ct = wopbs_key.wopbs(&ct_ks, &func_lut);
    lut_ct = wopbs_key.keyswitch_to_pbs_params(&lut_ct);
    (lut_ct, start.elapsed().as_secs_f64())
}

fn ct_lut_eval_quantized(
    ct: RadixCiphertext,
    precision: u8,
    bit_width: usize,
    nb_blocks: usize,
    func: &dyn Fn(f64) -> f64,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
) -> (RadixCiphertext, f64) {
    let quant_blocks = &ct.clone().into_blocks()[0..(nb_blocks >> 1)];
    let quantized_ct = RadixCiphertext::from_blocks(quant_blocks.to_vec());
    let quantized_lut = wopbs_key.generate_lut_radix(&quantized_ct, |x: u64| {
        let x_unquantized = unquantize(x, precision, (bit_width >> 1) as u8);
        quantize(func(x_unquantized), precision, (bit_width >> 1) as u8)
    });
    let start = Instant::now();
    let quant_blocks = &ct.into_blocks()[(nb_blocks >> 1)..nb_blocks];
    let quantized_ct = RadixCiphertext::from_blocks(quant_blocks.to_vec());
    let quantized_ct = wopbs_key.keyswitch_to_wopbs_params(server_key, &quantized_ct);
    let quantized_ct = wopbs_key.wopbs(&quantized_ct, &quantized_lut);
    (
        wopbs_key.keyswitch_to_pbs_params(&quantized_ct),
        start.elapsed().as_secs_f64(),
    )
}

fn ct_lut_eval_haar(
    ct: RadixCiphertext,
    precision: u8,
    bit_width: usize,
    nb_blocks: usize,
    func: &dyn Fn(f64) -> f64,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
) -> (RadixCiphertext, f64) {
    let (haar_lsb, haar_msb) = haar(precision, precision, bit_width as u8, &func);
    let dummy: RadixCiphertext = server_key.create_trivial_radix(0_u64, nb_blocks >> 1);
    let mut dummy_blocks = dummy.into_blocks().to_vec();
    for block in &mut dummy_blocks {
        block.degree = Degree::new(ct.blocks()[0].degree.get());
    }
    let dummy = RadixCiphertext::from_blocks(dummy_blocks);
    let dummy = wopbs_key.keyswitch_to_wopbs_params(server_key, &dummy);

    let haar_lsb_lut = wopbs_key.generate_lut_radix(&dummy, |x: u64| eval_lut(x, &haar_lsb));
    let haar_msb_lut = wopbs_key.generate_lut_radix(&dummy, |x: u64| eval_lut(x, &haar_msb));

    let start = Instant::now();
    // Truncate x
    let x_truncated_blocks = &ct.into_blocks()[(nb_blocks >> 1)..nb_blocks];
    let x_truncated = RadixCiphertext::from_blocks(x_truncated_blocks.to_vec());
    let x_truncated_ks = wopbs_key.keyswitch_to_wopbs_params(server_key, &x_truncated);

    let (haar_lsb, haar_msb) = rayon::join(
        || {
            let haar_lsb = wopbs_key.wopbs(&x_truncated_ks, &haar_lsb_lut);
            wopbs_key.keyswitch_to_pbs_params(&haar_lsb)
        },
        || {
            let haar_msb = wopbs_key.wopbs(&x_truncated_ks, &haar_msb_lut);
            wopbs_key.keyswitch_to_pbs_params(&haar_msb)
        },
    );
    let mut lsb_blocks = haar_lsb.into_blocks();
    lsb_blocks.extend(haar_msb.into_blocks());
    let haar_ct = RadixCiphertext::from_blocks(lsb_blocks.to_vec());
    (haar_ct, start.elapsed().as_secs_f64())
}

fn main() {
    // ------- Client side ------- //
    let bit_width = 24;

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

    let precision = 16u8;
    let x = quantize(64.0, precision, bit_width as u8);
    let x_ct = client_key.encrypt(x);

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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
        lut_prod,
        dwt_lut_prod,
        lut_prod_quant,
        unquantize(lut_prod, precision, bit_width as u8),
        unquantize(dwt_lut_prod, precision, bit_width as u8),
        unquantize(lut_prod_quant, precision, bit_width as u8),
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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
        lut_inv,
        dwt_lut_inv,
        lut_inv_quant,
        unquantize(lut_inv, precision, bit_width as u8),
        unquantize(dwt_lut_inv, precision, bit_width as u8),
        unquantize(lut_inv_quant, precision, bit_width as u8),
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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
        lut_log,
        dwt_lut_log,
        lut_log_quant,
        unquantize(lut_log, precision, bit_width as u8),
        unquantize(dwt_lut_log, precision, bit_width as u8),
        unquantize(lut_log_quant, precision, bit_width as u8),
    );

    // 4.1 sigmoid(x) using LUT
    fn sigmoid(value: f64) -> f64 {
        1f64 / (1f64 + (-value).exp())
    }
    fn sigmoid_naive_pt1(value: f64) -> f64 {
        -value.exp()
    }
    fn sigmoid_naive_pt2(value: f64) -> f64 {
        1f64 / (1f64 + value)
    }
    let (sig_ct, lut_time_pt1) = ct_lut_eval(
        x_ct.clone(),
        precision,
        bit_width,
        &sigmoid_naive_pt1,
        &wopbs_key,
        &server_key,
    );
    let (sig_ct, lut_time_pt2) = ct_lut_eval(
        sig_ct,
        precision,
        bit_width,
        &sigmoid_naive_pt2,
        &wopbs_key,
        &server_key,
    );
    let lut_sig: u64 = client_key.decrypt(&sig_ct);
    println!("Sigmoid (LUT) time: {:?}", lut_time_pt1 + lut_time_pt2);

    // 4.2. sigmoid(x) using Haar DWT LUT
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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
        lut_sig,
        dwt_lut_sig,
        lut_sig_quant,
        unquantize(lut_sig, precision, bit_width as u8),
        unquantize(dwt_lut_sig, precision, bit_width as u8),
        unquantize(lut_sig_quant, precision, bit_width as u8),
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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
        lut_inv_sqrt,
        dwt_inv_sqrt,
        lut_inv_sqrt_quant,
        unquantize(lut_inv_sqrt, precision, bit_width as u8),
        unquantize(dwt_inv_sqrt, precision, bit_width as u8),
        unquantize(lut_inv_sqrt_quant, precision, bit_width as u8),
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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
        lut_erf,
        dwt_erf,
        lut_erf_quant,
        unquantize(lut_erf, precision, bit_width as u8),
        unquantize(dwt_erf, precision, bit_width as u8),
        unquantize(lut_erf_quant, precision, bit_width as u8),
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

    // 6.2 tanh(x) using Haar DWT LUT
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

    // 6.3 tanh(x) using Quantized LUT
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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?}, \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
        lut_tanh,
        dwt_tanh,
        lut_tanh_quant,
        unquantize(lut_tanh, precision, bit_width as u8),
        unquantize(dwt_tanh, precision, bit_width as u8),
        unquantize(lut_tanh_quant, precision, bit_width as u8),
    );

    // let x_ct = client_key.encrypt(5_u64);
    // let x_neg_ct = client_key.encrypt(2_u64.pow(bit_width as u32)-5_u64);

    // let start = Instant::now();
    // let msb_block = vec![x_ct.blocks()[nb_blocks-1].clone(); 1];
    // let msb_radix = RadixCiphertext::from_blocks(msb_block);
    // let msb = server_key.scalar_right_shift_parallelized(&msb_radix, 1);
    // let sign_extract_time = start.elapsed().as_secs_f64();
    // let msb_dec: u64 = client_key.decrypt(&msb);
    // println!("Sign Extraction time: {:?}", sign_extract_time);
    // println!("Sign: {:?}", msb_dec);

    // let start = Instant::now();
    // let msb_block = vec![x_neg_ct.blocks()[nb_blocks-1].clone(); 1];
    // let msb_radix = RadixCiphertext::from_blocks(msb_block);
    // let msb = server_key.scalar_right_shift_parallelized(&msb_radix, 1);
    // let sign_extract_time = start.elapsed().as_secs_f64();
    // let msb_dec: u64 = client_key.decrypt(&msb);
    // println!("Sign Extraction time: {:?}", sign_extract_time);
    // println!("Sign: {:?}", msb_dec);
}
