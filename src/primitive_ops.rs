use std::time::Instant;

use dwt::{transform, wavelet::Haar, Operation};
use ripple::common::*;
use tfhe::{
    integer::{
        gen_keys_radix, wopbs::*, IntegerCiphertext, IntegerRadixCiphertext, RadixCiphertext,
    },
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

pub fn haar_square(
    table_size: u8,
    input_precision: u8,
    output_precision: u8,
    bit_width: u8,
) -> (Vec<u64>, Vec<u64>) {
    let max = 1 << bit_width;
    let mut data = Vec::new();
    for x in 0..max {
        let x = unquantize(x, input_precision, bit_width);
        let square = x * x;
        data.push(square);
    }
    data.rotate_right(1 << (bit_width - 1));
    transform(
        &mut data,
        Operation::Forward,
        &Haar::new(),
        (bit_width - table_size) as usize,
    );
    let coef_len = 1 << table_size;
    let scalar = 2f64.powf(-((bit_width - table_size) as f64) / 2f64);
    let mut haar: Vec<u64> = data
        .get(0..coef_len)
        .unwrap()
        .iter()
        .map(|x| quantize(scalar * x, output_precision, bit_width))
        .collect();
    haar.rotate_right(1 << (table_size - 1));
    let mask = (1 << (bit_width / 2)) - 1;
    let lsb = haar.iter().map(|x| x & mask).collect();
    let msb = haar.iter().map(|x| x >> (bit_width / 2) & mask).collect();
    (lsb, msb)
}

fn eval_lut(x: u64, lut_map: &Vec<u64>) -> u64 {
    lut_map[x as usize]
}

fn main() {
    // ------- Client side ------- //
    let bit_width = 20;

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

    let x = 5_u64;
    // let y = 10_u64;
    let x_ct = client_key.encrypt(x);
    // let y_ct = client_key.encrypt(y);

    // ------- Server side ------- //

    // 1. Square
    println!("\n1. Square");

    // 1.1. Square using multiplication
    let start = Instant::now();
    let square_ct = server_key.mul_parallelized(&x_ct, &x_ct);
    println!("Ct-Ct Mult in {:?} sec.", start.elapsed().as_secs_f64());
    let prod: u64 = client_key.decrypt(&square_ct);

    // 1.2. Square using LUT
    let square_lut = wopbs_key.generate_lut_radix(&x_ct, |x: u64| x * x);
    let start = Instant::now();
    let x_ct_ks = wopbs_key.keyswitch_to_wopbs_params(&server_key, &x_ct);
    let mut square_ct = wopbs_key.wopbs(&x_ct_ks, &square_lut);
    square_ct = wopbs_key.keyswitch_to_pbs_params(&square_ct);
    println!("LUT Square in {:?} sec.", start.elapsed().as_secs_f64());
    let lut_prod: u64 = client_key.decrypt(&square_ct);

    // 1.3. Square using Haar DWT LUT
    let (haar_lsb, haar_msb) = haar_square((bit_width >> 1) as u8, 8_u8, 16_u8, bit_width as u8);
    dbg!(&haar_lsb);
    dbg!(&haar_msb);
    let dummy: RadixCiphertext = server_key.create_trivial_radix(0_u64, nb_blocks);
    let dummy_blocks = &dummy.into_blocks()[(nb_blocks >> 1)..nb_blocks];
    let dummy_msb = RadixCiphertext::from_blocks(dummy_blocks.to_vec());
    let dummy_msb = server_key.scalar_add_parallelized(&dummy_msb, 1);
    let haar_lsb_lut = wopbs_key.generate_lut_radix(&dummy_msb, |x: u64| eval_lut(x, &haar_lsb));
    let haar_msb_lut = wopbs_key.generate_lut_radix(&dummy_msb, |x: u64| eval_lut(x, &haar_msb));

    let start = Instant::now();
    // Truncate x
    let x_truncated_blocks = &x_ct.into_blocks()[(nb_blocks >> 1)..nb_blocks];
    let x_truncated = RadixCiphertext::from_blocks(x_truncated_blocks.to_vec());
    let x_truncated = server_key.scalar_add_parallelized(&x_truncated, 1);
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
        "--- Exact: {:?}, LUT: {:?}, DWT LUT: {:?}",
        prod, lut_prod, dwt_lut_prod
    );
}
