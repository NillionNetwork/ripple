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
    let (haar_lsb, haar_msb) = haar(
        precision,
        precision,
        bit_width as u8,
        bit_width as u8,
        &func,
    );
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

fn ct_lut_eval_haar_bounded(
    ct: RadixCiphertext,
    precision: u8,
    bit_width: usize,
    integer_size: u32,
    nb_blocks: usize,
    func: &dyn Fn(f64) -> f64,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
    is_symmetric: bool,
) -> (RadixCiphertext, f64) {
    let (haar_lsb, haar_msb) = haar(
        precision,
        precision,
        precision + integer_size as u8,
        bit_width as u8,
        &func,
    );

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
    let ltz = server_key.scalar_right_shift_parallelized(&ct, bit_width - 1);
    let sign = server_key.sub_parallelized(
        &server_key.create_trivial_radix(1, nb_blocks),
        &server_key.scalar_left_shift_parallelized(&ltz, 1),
    );
    let abs = server_key.mul_parallelized(&sign, &ct);

    println!(
        "Absolute value done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // Truncate x
    let tmp = (precision as usize) + (integer_size as usize);
    let x_truncated_blocks = &abs.clone().into_blocks()[(tmp - (bit_width >> 1)) >> 1..tmp >> 1];
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
    let mut haar_ct = RadixCiphertext::from_blocks(lsb_blocks.to_vec());

    // For non-symmetric (around zero) functions like Sigmoid.
    if !is_symmetric {
        // ltz = (msb == 1)
        let precision_encoded =
            server_key.create_trivial_radix(2_u64.pow(precision as u32), nb_blocks);
        let ltz = server_key.mul_parallelized(&precision_encoded, &ltz);

        // eval = sign * eval + ltz
        let eval = server_key.add_parallelized(&server_key.mul_parallelized(&haar_ct, &sign), &ltz);
        let check_value = 2_u64.pow(precision as u32 + integer_size);
        let check = server_key.scalar_lt_parallelized(&abs, check_value); // abs < 2^{integer_size + precision}
        let check = check.into_radix(nb_blocks, server_key);
        // limit = 1 - ltz
        let limit = server_key.sub_parallelized(&precision_encoded, &ltz);
        // return limit + check * (eval - limit)
        haar_ct = server_key.add_parallelized(
            &limit,
            &server_key.mul_parallelized(&check, &server_key.sub_parallelized(&eval, &limit)),
        );
    }

    (haar_ct, start.elapsed().as_secs_f64())
}

fn ct_lut_eval_bior(
    ct: RadixCiphertext,
    bit_width: usize,
    nb_blocks: usize,
    luts: &Vec<&Vec<u64>>,
    wave_depth: usize,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
) -> (RadixCiphertext, f64) {
    let nb_blocks_lsb = (bit_width - wave_depth) >> 1;
    let dummy: RadixCiphertext = server_key.create_trivial_radix(0_u64, wave_depth >> 1);
    let mut dummy_blocks = dummy.into_blocks().to_vec();
    for block in &mut dummy_blocks {
        block.degree = Degree::new(ct.blocks()[0].degree.get());
    }
    let dummy = RadixCiphertext::from_blocks(dummy_blocks);
    let dummy = wopbs_key.keyswitch_to_wopbs_params(server_key, &dummy);
    let encoded_luts = luts
        .iter()
        .map(|lut| wopbs_key.generate_lut_radix(&dummy, |x: u64| eval_lut(x, &lut.to_vec())))
        .collect::<Vec<_>>();

    let start = Instant::now();
    // Split into wave_depth MSBs and n - wave_depth LSBs
    let ct_blocks = &ct.into_blocks();
    let (lsb, msb) = rayon::join(
        || {
            let prediction_blocks_lsb = &ct_blocks[0..nb_blocks_lsb];
            RadixCiphertext::from_blocks(prediction_blocks_lsb.to_vec())
        },
        || {
            let prediction_blocks_msb = &ct_blocks[nb_blocks_lsb..nb_blocks];
            let prediction_msb = RadixCiphertext::from_blocks(prediction_blocks_msb.to_vec());
            wopbs_key.keyswitch_to_wopbs_params(server_key, &prediction_msb)
        },
    );
    let (output_1, output_2) = rayon::join(
        || {
            // Eval LUT over MSBs
            let lut_lsb = wopbs_key.wopbs(&msb, &encoded_luts[0]);
            let mut lut_lsb_blocks = wopbs_key.keyswitch_to_pbs_params(&lut_lsb).into_blocks();

            // Pad LUT output and LSB by 6 bits to avoid overflows
            let padding_ct_block = server_key
                .create_trivial_zero_radix::<RadixCiphertext>(3)
                .into_blocks();
            lut_lsb_blocks.extend(padding_ct_block.clone());
            let mut lsb_blocks = lsb.clone().into_blocks();
            lsb_blocks.extend(padding_ct_block);
            let lut_combined = RadixCiphertext::from_blocks(lut_lsb_blocks);
            let lsb_extended = RadixCiphertext::from_blocks(lsb_blocks);

            // l1 = 2^J - lsb
            let scalar_l1: RadixCiphertext =
                server_key.create_trivial_radix(2u64.pow(wave_depth as u32), nb_blocks_lsb + 1);
            let scalar_l1 = server_key.sub_parallelized(&scalar_l1, &lsb_extended);

            // Multiply l1 by LUT output
            server_key.mul_parallelized(&lut_combined, &scalar_l1)
        },
        || {
            // Eval LUT over MSBs
            let lut_lsb = wopbs_key.wopbs(&msb, &encoded_luts[1]);
            let mut lut_lsb_blocks = wopbs_key.keyswitch_to_pbs_params(&lut_lsb).into_blocks();

            // Pad LUT output and LSB by 6 bits to avoid overflows
            let padding_ct_block = server_key
                .create_trivial_zero_radix::<RadixCiphertext>(3)
                .into_blocks();
            lut_lsb_blocks.extend(padding_ct_block.clone());
            let mut lsb_blocks = lsb.clone().into_blocks();
            lsb_blocks.extend(padding_ct_block);
            let lut_combined = RadixCiphertext::from_blocks(lut_lsb_blocks);
            let lsb_extended = RadixCiphertext::from_blocks(lsb_blocks);

            // l2 = lsb
            // Multiply MSBs and LSBs
            server_key.mul_parallelized(&lut_combined, &lsb_extended)
        },
    );
    let probability = server_key.add_parallelized(&output_1, &output_2);
    (probability, start.elapsed().as_secs_f64())
}

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

    let x = quantize(4.0, precision, bit_width as u8);
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
        lut_sqrt_bior,
        unquantize(lut_prod, precision, bit_width as u8),
        unquantize(dwt_lut_prod, precision, bit_width as u8),
        unquantize(lut_prod_quant, precision, bit_width as u8),
        unquantize(lut_sqrt_bior, precision, bit_width as u8),
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
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?},\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
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
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?},\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
        lut_log,
        dwt_lut_log,
        lut_log_quant,
        unquantize(lut_log, precision, bit_width as u8),
        unquantize(dwt_lut_log, precision, bit_width as u8),
        unquantize(lut_log_quant, precision, bit_width as u8),
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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, DWT LUT (Bounded): {:?}, Quant LUT: {:?}, \
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, DWT LUT (Bounded): {:?}, Quant LUT {:?}",
        lut_sig,
        dwt_lut_sig,
        dwt_sig_bounded,
        lut_sig_quant,
        unquantize(lut_sig, precision, bit_width as u8),
        unquantize(dwt_lut_sig, precision, bit_width as u8),
        unquantize(dwt_sig_bounded, precision, bit_width as u8),
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
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?},\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
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
        "--- LUT: {:?}, DWT LUT: {:?}, Quant LUT: {:?},\
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, Quant LUT {:?}",
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

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}, DWT LUT (Bounded): {:?}, Quant LUT: {:?}, \
         \n--- unq: LUT: {:?}, DWT LUT: {:?}, DWT LUT (Bounded): {:?}, Quant LUT {:?}",
        lut_tanh,
        dwt_tanh,
        dwt_tanh_bounded,
        lut_tanh_quant,
        unquantize(lut_tanh, precision, bit_width as u8),
        unquantize(dwt_tanh, precision, bit_width as u8),
        unquantize(dwt_tanh_bounded, precision, bit_width as u8),
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
