use std::time::Instant;

use rayon::prelude::*;
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

fn eval_lut(x: u64, lut: &Vec<u64>) -> u64 {
    lut[x as usize]
}

fn main() {
    // ------- Client side ------- //
    let bit_width = 16;
    let precision = 12;
    let table_size = 8;

    let (lut_lsb_plain, lut_msb_plain) = haar(table_size, precision, precision, bit_width);

    // Number of blocks per ciphertext
    let pbs_blocks = bit_width >> 2;
    println!("Number of blocks: {:?}", pbs_blocks);

    let start = Instant::now();
    // Generate radix keys
    let (client_key, server_key) =
        gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, pbs_blocks.into());

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

    let dataset: Vec<u64> = vec![0, 72, 1050, 1790, 10234, 60122, 65001, 65535];
    // Expected [2079, 2079, 2333, 2458, 3776,  847, 1888, 2015]
    // Received [2143, 2143, 2396, 2519, 3794,  890, 1951, 2079]

    // let mut dataset = Vec::new();
    // let max = 1 << 10;
    // for i in 0..max {
    //     dataset.push(i * (1 << 6));
    // }

    let start = Instant::now();
    let mut encrypted_dataset: Vec<_> = dataset
        .par_iter() // Use par_iter() for parallel iteration
        .map(|&sample| {
            let mut lsb = client_key
                .encrypt(sample & (1 << ((pbs_blocks << 1) - 1)))
                .into_blocks(); // Get LSBs
            let msb = client_key
                .encrypt(sample >> (pbs_blocks << 1))
                .into_blocks(); // Get MSBs
            lsb.extend(msb);
            RadixCiphertext::from_blocks(lsb)
        })
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //
    let all_probabilities = encrypted_dataset
        .par_iter_mut()
        .enumerate()
        .map(|(cnt, sample)| {
            let start = Instant::now();
            println!("Started inference #{:?}.", cnt);

            // Truncate
            let mut prediction = sample.clone();
            let prediction_blocks =
                &prediction.into_blocks()[(pbs_blocks as usize)..((pbs_blocks << 1) as usize)];
            let prediction_msb = RadixCiphertext::from_blocks(prediction_blocks.to_vec());
            // let prediction_msb = server_key.unchecked_scalar_add(&prediction_msb, 1);
            // Keyswitch and Bootstrap
            prediction = wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction_msb);
            let lut_lsb =
                wopbs_key.generate_lut_radix(&prediction, |x: u64| eval_lut(x, &lut_lsb_plain));
            let lut_msb =
                wopbs_key.generate_lut_radix(&prediction, |x: u64| eval_lut(x, &lut_msb_plain));
            let activation_lsb = wopbs_key.wopbs(&prediction, &lut_lsb);
            let mut lsb_blocks = wopbs_key
                .keyswitch_to_pbs_params(&activation_lsb)
                .into_blocks();
            let activation_msb = wopbs_key.wopbs(&prediction, &lut_msb);
            let msb_blocks = wopbs_key
                .keyswitch_to_pbs_params(&activation_msb)
                .into_blocks();
            lsb_blocks.extend(msb_blocks);
            let probability = RadixCiphertext::from_blocks(lsb_blocks);

            println!(
                "Finished inference #{:?} in {:?} sec.",
                cnt,
                start.elapsed().as_secs_f64()
            );
            probability
        })
        .collect::<Vec<_>>();

    // ------- Client side ------- //
    for (prob, data) in all_probabilities.iter().zip(dataset.iter()) {
        let res: u64 = client_key.decrypt(prob);
        let exp_lsb = lut_lsb_plain[(data >> (bit_width - table_size)) as usize];
        let exp_msb = lut_msb_plain[(data >> (bit_width - table_size)) as usize];
        let exp = (exp_msb << (bit_width / 2)) + exp_lsb;
        println!(
            "For {:?} got probability {:?}, expeced {:?}",
            data, res, exp
        );
    }
}
