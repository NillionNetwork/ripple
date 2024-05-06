use std::time::Instant;

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

fn eval_lut(x: u64, lut_map: &Vec<u64>) -> u64 {
    lut_map[x as usize]
}

fn main() {
    // ------- Client side ------- //
    let bit_width = 12;

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

    let precision = 4u8;
    let x = quantize(9.0, precision, bit_width as u8);
    let x_ct = client_key.encrypt(x);

    // ------- Server side ------- //

    // 1.1. Square root using LUT
    let square_lut = wopbs_key.generate_lut_radix(&x_ct, |x: u64| {
        let x_unquantized = unquantize(x, precision, bit_width as u8);
        quantize(x_unquantized.sqrt(), precision, bit_width as u8)
    });
    let start = Instant::now();
    let x_ct_ks = wopbs_key.keyswitch_to_wopbs_params(&server_key, &x_ct);
    let mut square_ct = wopbs_key.wopbs(&x_ct_ks, &square_lut);
    square_ct = wopbs_key.keyswitch_to_pbs_params(&square_ct);
    println!("LUT Square in {:?} sec.", start.elapsed().as_secs_f64());
    let lut_prod: u64 = client_key.decrypt(&square_ct);

    // 1.2. Square root using Haar DWT LUT
    fn my_sqrt(value: f64) -> f64 {
        value.sqrt()
    }
    let (haar_lsb, haar_msb) = haar(precision, precision, bit_width as u8, &my_sqrt);
    let dummy: RadixCiphertext = server_key.create_trivial_radix(0_u64, nb_blocks >> 1);
    let mut dummy_blocks = dummy.into_blocks().to_vec();
    for block in &mut dummy_blocks {
        block.degree = Degree::new(x_ct.blocks()[0].degree.get());
    }
    let dummy = RadixCiphertext::from_blocks(dummy_blocks);
    let dummy = wopbs_key.keyswitch_to_wopbs_params(&server_key, &dummy);

    let haar_lsb_lut = wopbs_key.generate_lut_radix(&dummy, |x: u64| eval_lut(x, &haar_lsb));
    let haar_msb_lut = wopbs_key.generate_lut_radix(&dummy, |x: u64| eval_lut(x, &haar_msb));

    // Ptxt DWT Haar -- works correctly.
    // let x_truncated = x >> (bit_width >> 1);
    // let dwt_lut_prod = (haar_msb[x_truncated as usize] << (bit_width >> 1)) + haar_lsb[x_truncated as usize];

    let start = Instant::now();
    // Truncate x
    let x_truncated_blocks = &x_ct.into_blocks()[(nb_blocks >> 1)..nb_blocks];
    let x_truncated = RadixCiphertext::from_blocks(x_truncated_blocks.to_vec());
    let x_truncated_ks = wopbs_key.keyswitch_to_wopbs_params(&server_key, &x_truncated);

    let (square_lsb, square_msb) = rayon::join(
        || {
            let square_lsb = wopbs_key.wopbs(&x_truncated_ks, &haar_lsb_lut);
            wopbs_key.keyswitch_to_pbs_params(&square_lsb)
        },
        || {
            let square_msb = wopbs_key.wopbs(&x_truncated_ks, &haar_msb_lut);
            wopbs_key.keyswitch_to_pbs_params(&square_msb)
        },
    );
    let mut square_lsb_blocks = square_lsb.into_blocks();
    square_lsb_blocks.extend(square_msb.into_blocks());
    let square_ct_haar = RadixCiphertext::from_blocks(square_lsb_blocks.to_vec());

    println!(
        "Haar LUT Square in {:?} sec.",
        start.elapsed().as_secs_f64()
    );
    let dwt_lut_prod: u64 = client_key.decrypt(&square_ct_haar);

    println!(
        "--- LUT: {:?}, DWT LUT: {:?}\n --- unq: LUT: {:?}, DWT LUT: {:?}",
        lut_prod,
        dwt_lut_prod,
        unquantize(lut_prod, precision, bit_width as u8),
        unquantize(dwt_lut_prod, precision, bit_width as u8),
    );
}
