use std::fs::File;

use tfhe::{
    integer::{gen_keys_radix, wopbs::*, RadixCiphertext},
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

fn quantize(x: f32, precision: u8) -> u64 {
    let mut tmp = (x * ((1 << precision) as f32)) as i32;
    tmp += 1 << (precision - 1);
    tmp as u64
}

fn sigmoid(x: u64) -> u64 {
    let x_f32 = x as f32;
    let exp = (-x_f32 / ((1 << 16) as f32)).exp();
    ((1.0 / (1.0 + exp)) * (1 << 8) as f32) as u64
}

fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

fn load_weights_and_biases() -> (Vec<Vec<f32>>, Vec<f32>) {
    let weights_csv = File::open("iris_weights.csv").unwrap();
    let mut reader = csv::Reader::from_reader(weights_csv);
    let mut weights = vec![];
    let mut biases = vec![];

    for result in reader.deserialize() {
        let res: Vec<f32> = result.expect("a CSV record");
        biases.push(res[0]);
        weights.push(res[1..].to_vec());
    }

    (weights, biases)
}

fn quantize_weights_and_biases(
    weights: &[Vec<f32>],
    biases: &[f32],
    precision: u8,
) -> (Vec<Vec<u64>>, Vec<u64>) {
    let weights_int = weights
        .iter()
        .map(|row| {
            row.iter()
                .map(|&w| quantize(w, precision))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bias_int = biases
        .iter()
        .map(|&w| quantize(w, precision))
        .collect::<Vec<_>>();

    (weights_int, bias_int)
}

fn prepare_iris_dataset() -> Vec<(Vec<f64>, usize)> {
    let iris = linfa_datasets::iris();
    let mut iris_dataset = vec![];

    for (sample, target) in iris.sample_iter() {
        iris_dataset.push((sample.to_vec(), *target.first().unwrap()));
    }

    iris_dataset
}

fn means_and_stds(dataset: &[(Vec<f64>, usize)], num_features: usize) -> (Vec<f64>, Vec<f64>) {
    let mut means = vec![0f64; num_features];
    let mut stds = vec![0f64; num_features];

    for (sample, _) in dataset.iter() {
        for (feature, s) in sample.iter().enumerate() {
            means[feature] += s;
        }
    }
    for mean in means.iter_mut() {
        *mean /= dataset.len() as f64;
    }
    for (sample, _) in dataset.iter() {
        for (feature, s) in sample.iter().enumerate() {
            let dev = s - means[feature];
            stds[feature] += dev * dev;
        }
    }
    for std in stds.iter_mut() {
        *std = (*std / dataset.len() as f64).sqrt();
    }

    (means, stds)
}

fn main() {
    // ------- Client side ------- //
    let precision = 8;
    // Number of blocks per ciphertext
    let nb_blocks = 4;

    // Generate radix keys
    let (cks, sks) = gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, nb_blocks);

    // Generate key for PBS (without padding)
    let wopbs_key = WopbsKey::new_wopbs_key(&cks, &sks, &WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS);

    // Get message modulus (i.e. max value representable by radix ctxt)
    let mut modulus = 1_u64;
    for _ in 0..nb_blocks {
        modulus *= cks.parameters().message_modulus().0 as u64;
    }
    println!("Ptxt Modulus: {:?}", modulus);

    let (weights, biases) = load_weights_and_biases();
    let (weights_int, bias_int) = quantize_weights_and_biases(&weights, &biases, precision);
    let iris_dataset = prepare_iris_dataset();
    let num_features = iris_dataset[0].0.len();
    let (means, stds) = means_and_stds(&iris_dataset, num_features);

    // TODO(@jimouris): par_iter and map
    let mut encrypted_dataset = vec![];
    for (sample, _) in iris_dataset.iter() {
        let mut input = vec![];
        for (&s, (mean, std)) in sample.iter().zip(means.iter().zip(stds.iter())) {
            let n = (s - mean) / std;
            let quantized = quantize(n as f32, precision);
            input.push(cks.encrypt(quantized));
        }
        encrypted_dataset.push(input);
    }

    // ------- Server side ------- //

    // Build LUT for Sigmoid
    let sigmoid_lut = wopbs_key.generate_lut_radix(&encrypted_dataset[0][0], |x: u64| sigmoid(x));

    let mut all_probabilities = vec![];
    let mut cnt = 0;
    for sample in encrypted_dataset.iter() {
        let mut probabilities = vec![];
        for (model, bias) in weights_int.iter().zip(bias_int.iter()) {
            let mut prediction: RadixCiphertext = sks.create_trivial_radix(*bias, nb_blocks);
            for ((encrypted_value, weight), (_, _)) in sample
                .iter()
                .zip(model.iter())
                .zip(means.iter().zip(stds.iter()))
            {
                // prediction += weight * encrypted_value;
                let ct_prod = sks.unchecked_small_scalar_mul(encrypted_value, *weight);
                prediction = sks.unchecked_add(&ct_prod, &prediction);
            }
            prediction = wopbs_key.keyswitch_to_wopbs_params(&sks, &prediction);
            let activation = wopbs_key.wopbs(&prediction, &sigmoid_lut);
            prediction = wopbs_key.keyswitch_to_pbs_params(&activation);
            probabilities.push(prediction);
        }
        println!("Finished inference #{:?}", cnt);
        all_probabilities.push(probabilities);
        cnt += 1;
        if cnt == 2 {
            break;
        }
    }

    // ------- Client side ------- //
    let mut total = 0;
    for (num, ((_, target), probabilities)) in iris_dataset
        .iter()
        .zip(all_probabilities.iter())
        .enumerate()
    {
        let ptxt_probabilities = probabilities
            .iter()
            .map(|p| cks.decrypt(p))
            .collect::<Vec<u64>>();
        let class = argmax(&ptxt_probabilities).unwrap();
        println!("[{}] predicted {:?}, target {:?}", num, class, target);
        if class == *target {
            total += 1;
        }
        if num == 2 {
            break;
        }
    }
    let accuracy = (total as f32 / iris_dataset.len() as f32) * 100.0;
    println!("Accuracy {accuracy}%");
}
