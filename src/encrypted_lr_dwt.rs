use std::{collections::HashMap, fs::File, time::Instant};

use fhe_lut::common::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tfhe::{
    integer::{
        gen_keys_radix, wopbs::*, IntegerCiphertext, IntegerRadixCiphertext, RadixCiphertext,
    },
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

#[derive(Debug, Serialize, Deserialize)]
struct KeyValue {
    key: u64,
    value: u64,
}

fn eval_sigmoid(x: u64, sigmoid_map: &HashMap<u64, u64>) -> u64 {
    sigmoid_map[&x]
}

fn main() {
    let file = File::open("lut16_quantized_lsb.json").unwrap();
    let json = serde_json::from_reader(file).unwrap();
    let kv_lut: Vec<KeyValue> = serde_json::from_value(json).unwrap();
    let mut lut_lsb: HashMap<u64, u64> = HashMap::new();
    for entry in kv_lut {
        lut_lsb.insert(entry.key, entry.value);
    }
    let file = File::open("lut16_quantized_msb.json").unwrap();
    let json = serde_json::from_reader(file).unwrap();
    let kv_lut: Vec<KeyValue> = serde_json::from_value(json).unwrap();
    let mut lut_msb: HashMap<u64, u64> = HashMap::new();
    for entry in kv_lut {
        lut_msb.insert(entry.key, entry.value);
    }
    // ------- Client side ------- //
    let bit_width = 16u8;
    let precision = bit_width >> 2;
    assert!(precision <= bit_width / 2);

    // Number of blocks per ciphertext
    let nb_blocks = bit_width >> 2;
    println!("Number of blocks: {:?}", nb_blocks);

    let start = Instant::now();
    // Generate radix keys
    let (client_key, server_key) = gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, nb_blocks.into());

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

    let (weights, biases) = load_weights_and_biases();
    let (weights_int, bias_int) =
        quantize_weights_and_biases(&weights, &biases, precision, bit_width);
    let (iris_dataset, targets) = prepare_iris_dataset();
    let num_features = iris_dataset[0].len();
    let (means, stds) = means_and_stds(&iris_dataset, num_features);

    let start = Instant::now();
    let mut encrypted_dataset: Vec<Vec<_>> = iris_dataset
        .par_iter() // Use par_iter() for parallel iteration
        .map(|sample| {
            sample
                .par_iter()
                .zip(means.par_iter().zip(stds.par_iter()))
                .map(|(&s, (mean, std))| {
                    let quantized = quantize((s - mean) / std, precision, bit_width);
                    let mut lsb = client_key
                        .encrypt(quantized & (1 << ((nb_blocks << 1) - 1)))
                        .into_blocks(); // Get LSBs
                    let msb = client_key
                        .encrypt(quantized >> (nb_blocks << 1))
                        .into_blocks(); // Get MSBs
                    lsb.extend(msb);
                    RadixCiphertext::from_blocks(lsb)
                })
                .collect()
        })
        .collect();
    println!(
        "Encryption done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // ------- Server side ------- //

    // Dummy encryption for building LUTs
    let ct_dummy = client_key.encrypt(2_u64);
    let mut ct_test = ct_dummy.clone();
    let ct_dummy = wopbs_key.keyswitch_to_wopbs_params(&server_key, &ct_dummy);

    // Build LUT for Sigmoid
    let sigmoid_lut_lsb =
        wopbs_key.generate_lut_radix(&ct_dummy, |x: u64| eval_sigmoid(x, &lut_lsb));
    let sigmoid_lut_msb =
        wopbs_key.generate_lut_radix(&ct_dummy, |x: u64| eval_sigmoid(x, &lut_msb));

    // FIXME: LUT TEST
    let mut ct_dummy = wopbs_key.keyswitch_to_pbs_params(&ct_dummy);
    ct_dummy = server_key.smart_mul(&mut ct_dummy, &mut ct_test);
    let test: u64 = client_key.decrypt(&ct_dummy);
    println!("Test Input: {:?}", &test);
    println!("Expected activation (LSB): {:?}", &lut_lsb[&test]);
    println!("Expected activation (MSB): {:?}", &lut_msb[&test]);
    let ct_dummy = wopbs_key.keyswitch_to_wopbs_params(&server_key, &ct_dummy);
    let activation_lsb = wopbs_key.wopbs(&ct_dummy, &sigmoid_lut_lsb);
    let activation_lsb = wopbs_key.keyswitch_to_pbs_params(&activation_lsb);
    let test: u64 = client_key.decrypt(&activation_lsb);
    println!("Activation (LSB): {:?}", &test);
    let activation_msb = wopbs_key.wopbs(&ct_dummy, &sigmoid_lut_msb);
    let activation_msb = wopbs_key.keyswitch_to_pbs_params(&activation_msb);
    let test: u64 = client_key.decrypt(&activation_msb);
    println!("Activation (MSB): {:?}", &test);

    let encrypted_dataset_short = encrypted_dataset.get_mut(0..1).unwrap();
    let all_probabilities = encrypted_dataset_short
        .iter_mut()
        .enumerate()
        .map(|(cnt, sample)| {
            let start = Instant::now();
            let probabilities = weights_int
                .iter()
                .zip(bias_int.iter())
                // .par_iter()
                // .zip(bias_int.par_iter())
                .map(|(model, &bias)| {
                    let scaled_bias = mul(1 << precision, bias, bit_width);
                    let mut prediction =
                        server_key.create_trivial_radix(scaled_bias, (nb_blocks << 1).into());
                    for (s, &weight) in sample.iter_mut().zip(model.iter()) {
                        let ct_prod = server_key.smart_scalar_mul(s, weight);
                        prediction = server_key.unchecked_add(&ct_prod, &prediction);
                    }
                    let test: u64 = client_key.decrypt(&prediction);
                    println!("Original: {:?}", &test);
                    // Truncate
                    let prediction_blocks = &prediction.clone().into_blocks()
                        [(nb_blocks as usize)..((nb_blocks << 1) as usize)];
                    let prediction_msb = RadixCiphertext::from_blocks(prediction_blocks.to_vec());
                    // For some reason, the truncation is off by 1...
                    let prediction_msb = server_key.unchecked_scalar_add(&prediction_msb, 1);
                    let test: u64 = client_key.decrypt(&prediction_msb);
                    println!("Truncated: {:?}", &test);
                    println!("Expected activation (LSB): {:?}", &lut_lsb[&test]);
                    println!("Expected activation (MSB): {:?}", &lut_msb[&test]);
                    // Keyswitch and Bootstrap
                    prediction = wopbs_key.keyswitch_to_wopbs_params(&server_key, &prediction_msb);
                    let activation_lsb = wopbs_key.wopbs(&prediction, &sigmoid_lut_lsb);
                    let activation_lsb = wopbs_key.keyswitch_to_pbs_params(&activation_lsb);
                    let test: u64 = client_key.decrypt(&activation_lsb);
                    println!("Activation (LSB): {:?}", &test);
                    let activation_msb = wopbs_key.wopbs(&prediction, &sigmoid_lut_msb);
                    let activation_msb = wopbs_key.keyswitch_to_pbs_params(&activation_msb);
                    let test: u64 = client_key.decrypt(&activation_msb);
                    println!("Activation (MSB): {:?}", &test);

                    // let (activation_lsb, activation_msb) = rayon::join(
                    //   || {
                    //       let activation_lsb = wopbs_key.wopbs(&prediction, &sigmoid_lut_lsb);
                    //       wopbs_key.keyswitch_to_pbs_params(&activation_lsb)
                    //   },
                    //   || {
                    //     let activation_msb = wopbs_key.wopbs(&prediction, &sigmoid_lut_msb);
                    //     wopbs_key.keyswitch_to_pbs_params(&activation_msb)
                    //   }
                    // );
                    let mut lsb_blocks = activation_lsb.clone().into_blocks();
                    let msb_blocks = activation_msb.clone().into_blocks();
                    lsb_blocks.extend(msb_blocks);
                    RadixCiphertext::from_blocks(lsb_blocks)
                })
                .collect::<Vec<_>>();
            println!(
                "Finished inference #{:?} in {:?} sec.",
                cnt,
                start.elapsed().as_secs_f64()
            );
            probabilities
        })
        .collect::<Vec<_>>();
    // }

    // ------- Client side ------- //
    let mut total = 0;
    for (num, (target, probabilities)) in targets.iter().zip(all_probabilities.iter()).enumerate() {
        let ptxt_probabilities = probabilities
            .iter()
            .map(|p| client_key.decrypt(p))
            .collect::<Vec<u64>>();
        println!("{:?}", ptxt_probabilities);
        let class = argmax(&ptxt_probabilities).unwrap();
        println!("[{}] predicted {:?}, target {:?}", num, class, target);
        if class == *target {
            total += 1;
        }
    }
    let accuracy = (total as f32 / encrypted_dataset_short.len() as f32) * 100.0;
    println!("Accuracy {accuracy}%");
}
