use std::{collections::HashMap, fs::File, io::BufReader, time::Instant};

use dwt::{transform, wavelet::Haar, Operation};
use tfhe::{
    integer::{wopbs::*, IntegerCiphertext, IntegerRadixCiphertext, RadixCiphertext, ServerKey},
    shortint::parameters::Degree,
};

pub fn to_signed(x: u64, bit_width: u8) -> i64 {
    if x >= (1u64 << (bit_width - 1)) {
        (x as i128 - (1i128 << bit_width)) as i64
    } else {
        x as i64
    }
}

pub fn from_signed(x: i64, bit_width: u8) -> u64 {
    (x as i128).rem_euclid(1i128 << bit_width) as u64
}

pub fn quantize(x: f64, precision: u8, bit_width: u8) -> u64 {
    from_signed((x * ((1u128 << precision) as f64)) as i64, bit_width)
}

pub fn unquantize(x: u64, precision: u8, bit_width: u8) -> f64 {
    to_signed(x, bit_width) as f64 / ((1u128 << precision) as f64)
}

pub fn add(a: u64, b: u64, bit_width: u8) -> u64 {
    (a as u128 + b as u128).rem_euclid(1u128 << bit_width) as u64
}

pub fn mul(a: u64, b: u64, bit_width: u8) -> u64 {
    (a as u128 * b as u128).rem_euclid(1u128 << bit_width) as u64
}

pub fn trunc(x: u64, bit_width: u8, truncation: u8) -> u64 {
    let y = to_signed(x, bit_width);
    let z = y >> truncation;
    from_signed(z, bit_width)
}

pub fn sigmoid(x: u64, input_precision: u8, output_precision: u8, bit_width: u8) -> u64 {
    let x = unquantize(x, input_precision, bit_width);
    let sig = 1f64 / (1f64 + (-x).exp());
    quantize(sig, output_precision, bit_width)
}

pub fn load_weights_and_biases() -> (Vec<f64>, f64) {
    let weights_csv = File::open("data/penguins_weight.csv").unwrap();
    let mut reader = csv::Reader::from_reader(weights_csv);
    let mut weights = vec![];
    let mut bias = 0f64;

    for result in reader.deserialize() {
        let res: Vec<f64> = result.expect("a CSV record");
        bias = res[0];
        weights = res[1..].to_vec();
    }

    (weights, bias)
}

pub fn quantize_weights_and_bias(
    weights: &[f64],
    bias: f64,
    precision: u8,
    bit_width: u8,
) -> (Vec<u64>, u64) {
    let weights_int = weights
        .iter()
        .map(|&w| quantize(w, precision, bit_width))
        .collect::<Vec<_>>();
    // Quantize and double precision as bias will be added to double precision terms
    let bias_int = mul(
        1 << precision,
        quantize(bias, precision, bit_width),
        bit_width,
    );

    (weights_int, bias_int)
}

pub fn prepare_penguins_dataset() -> (Vec<Vec<f64>>, Vec<usize>) {
    let data_csv = File::open("data/penguins_data.csv").unwrap();
    let mut reader = csv::Reader::from_reader(data_csv);
    let mut dataset = vec![];

    for result in reader.deserialize() {
        let res: Vec<f64> = result.expect("a CSV record");
        dataset.push(res);
    }

    let target_csv = File::open("data/penguins_target.csv").unwrap();
    let mut reader = csv::Reader::from_reader(target_csv);
    let mut targets = vec![];
    for result in reader.deserialize() {
        let res: Vec<f64> = result.expect("a CSV record");
        targets.push(res[0] as usize);
    }

    (dataset, targets)
}

pub fn means_and_stds(dataset: &[Vec<f64>], num_features: usize) -> (Vec<f64>, Vec<f64>) {
    let mut maxs = vec![0f64; num_features];
    let mut mins = vec![0f64; num_features];

    for sample in dataset.iter() {
        for (feature, s) in sample.iter().enumerate() {
            if maxs[feature] < *s {
                maxs[feature] = *s;
            }
            if mins[feature] > *s {
                mins[feature] = *s;
            }
        }
    }

    (mins, maxs)
}

pub fn haar(
    input_precision: u8,
    output_precision: u8,
    max_bit: u8,
    bit_width: u8,
    f: &dyn Fn(f64) -> f64,
) -> (Vec<u64>, Vec<u64>) {
    let table_size = bit_width >> 1;
    let max = 1 << max_bit;
    let mut data = Vec::new();
    let mut negatives = false;
    for x in 0..max {
        let x = unquantize(x, input_precision, bit_width);
        data.push(f(x));
        if x < 0.0 {
            negatives = true;
        }
    }
    if negatives {
        data.rotate_right(1 << (max_bit - 1));
    }
    transform(
        &mut data,
        Operation::Forward,
        &Haar::new(),
        (max_bit - table_size) as usize,
    );
    let coef_len = 1 << table_size;
    let scalar = 2f64.powf(-((max_bit - table_size) as f64) / 2f64);
    let mut haar: Vec<u64> = data
        .get(0..coef_len)
        .unwrap()
        .iter()
        .map(|x| quantize(scalar * x, output_precision, bit_width))
        .collect();
    if negatives {
        haar.rotate_right(1 << (table_size - 1));
    }
    let mask = (1 << (bit_width / 2)) - 1;
    let lsb = haar.iter().map(|x| x & mask).collect();
    let msb = haar.iter().map(|x| x >> (bit_width / 2) & mask).collect();
    (lsb, msb)
}

pub fn bior(lut_file: &str, table_size: u8, bit_width: u8) -> (Vec<u64>, Vec<u64>) {
    // Read Biorthogonal LUT
    let reader = BufReader::new(File::open(lut_file).unwrap());
    let bior_lut: HashMap<u64, u64> = serde_json::from_reader(reader).unwrap();

    // Convert to 1-D vector
    let mut bior_lut_vec = vec![];
    for i in 0..(2u64.pow(table_size as u32)) {
        bior_lut_vec.push(bior_lut[&(i as u64)]);
    }

    // Break into two LUTs
    let mask = (1 << (bit_width / 2)) - 1;
    let lsb = bior_lut_vec.iter().map(|x| x & mask).collect();
    let msb = bior_lut_vec
        .iter()
        .map(|x| x >> (bit_width / 2) & mask)
        .collect();
    (lsb, msb)
}

pub fn db2() -> (Vec<Vec<u64>>, Vec<u64>) {
    // Read DB2 LUTs
    let reader = BufReader::new(File::open("./data/db2_lut_1.json").unwrap());
    let db2_lut_1: HashMap<u64, u64> = serde_json::from_reader(reader).unwrap();
    let reader = BufReader::new(File::open("./data/db2_lut_2.json").unwrap());
    let db2_lut_2: HashMap<u64, u64> = serde_json::from_reader(reader).unwrap();
    let reader = BufReader::new(File::open("./data/db2_lut_3.json").unwrap());
    let db2_lut_3: HashMap<u64, u64> = serde_json::from_reader(reader).unwrap();
    let reader = BufReader::new(File::open("./data/db2_lut_4.json").unwrap());
    let db2_lut_4: HashMap<u64, u64> = serde_json::from_reader(reader).unwrap();

    // Convert LSB LUTs to 2-D vector
    let lut_lsb_vecs = vec![
        db2_lut_1.into_values().collect::<Vec<_>>(),
        db2_lut_2.into_values().collect::<Vec<_>>(),
        db2_lut_3.into_values().collect::<Vec<_>>(),
    ];

    // Convert MSB LUT to 1-D vector
    let lut_msb_vec = db2_lut_4.into_values().collect::<Vec<_>>();

    (lut_lsb_vecs, lut_msb_vec)
}

pub fn read_csv(filename: &str) -> Vec<Vec<u32>> {
    let csv = File::open(filename).unwrap();
    let mut reader = csv::Reader::from_reader(csv);

    let num_columns = reader.headers().unwrap().len();
    let mut data = vec![vec![]; num_columns];
    for line in reader.deserialize() {
        let record: Vec<u32> = line.unwrap();
        if record.len() != num_columns {
            panic!("Number of columns in row does not match header");
        }
        for (i, &value) in record.iter().enumerate() {
            data[i].push(value);
        }
    }

    data
}

pub fn quantized_table(
    table_size: u8,
    input_precision: u8,
    output_precision: u8,
    bit_width: u8,
) -> (Vec<u64>, Vec<u64>) {
    let mut data = Vec::new();
    let max = 1 << (table_size);
    for x in 0..max {
        let x = x << (bit_width - table_size);
        let xq = unquantize(x, input_precision, bit_width);
        let sig = 1f64 / (1f64 + (-xq).exp());
        data.push(quantize(sig, output_precision, bit_width));
    }
    let mask = (1 << (bit_width / 2)) - 1;
    let lsb = data.clone().iter().map(|x| x & mask).collect();
    let msb = data.iter().map(|x| x >> (bit_width / 2) & mask).collect();
    (lsb, msb)
}

fn eval_lut(x: u64, lut_map: &Vec<u64>) -> u64 {
    lut_map[x as usize]
}

pub fn ct_lut_eval(
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

pub fn ct_lut_eval_no_gen(
    ct: RadixCiphertext,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
    func_lut: &IntegerWopbsLUT,
) -> RadixCiphertext {
    let ct_ks = wopbs_key.keyswitch_to_wopbs_params(server_key, &ct);
    let mut lut_ct = wopbs_key.wopbs(&ct_ks, &func_lut);
    lut_ct = wopbs_key.keyswitch_to_pbs_params(&lut_ct);
    lut_ct
}

pub fn ct_lut_eval_quantized(
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

pub fn ct_lut_eval_quantized_no_gen(
    ct: RadixCiphertext,
    nb_blocks: usize,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
    quantized_lut: &IntegerWopbsLUT,
) -> RadixCiphertext {
    let quant_blocks = &ct.into_blocks()[(nb_blocks >> 1)..nb_blocks];
    let quantized_ct = RadixCiphertext::from_blocks(quant_blocks.to_vec());
    let quantized_ct = wopbs_key.keyswitch_to_wopbs_params(server_key, &quantized_ct);
    let quantized_ct = wopbs_key.wopbs(&quantized_ct, &quantized_lut);
    wopbs_key.keyswitch_to_pbs_params(&quantized_ct)
}

pub fn ct_lut_eval_haar(
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

pub fn ct_lut_eval_haar_no_gen(
    ct: RadixCiphertext,
    nb_blocks: usize,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
    haar_lsb_lut: &IntegerWopbsLUT,
    haar_msb_lut: &IntegerWopbsLUT,
) -> RadixCiphertext {
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
    haar_ct
}

pub fn ct_lut_eval_haar_bounded(
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

pub fn ct_lut_eval_haar_bounded_no_gen(
    ct: RadixCiphertext,
    precision: u8,
    bit_width: usize,
    integer_size: u32,
    nb_blocks: usize,
    wopbs_key: &WopbsKey,
    server_key: &ServerKey,
    is_symmetric: bool,
    haar_lsb_lut: &IntegerWopbsLUT,
    haar_msb_lut: &IntegerWopbsLUT,
) -> RadixCiphertext {
    let ltz = server_key.scalar_right_shift_parallelized(&ct, bit_width - 1);
    let sign = server_key.sub_parallelized(
        &server_key.create_trivial_radix(1, nb_blocks),
        &server_key.scalar_left_shift_parallelized(&ltz, 1),
    );
    let abs = server_key.mul_parallelized(&sign, &ct);

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
    haar_ct
}

pub fn ct_lut_eval_bior(
    ct: RadixCiphertext,
    bit_width: usize,
    nb_blocks: usize,
    luts: &Vec<&Vec<u64>>,
    wave_depth: usize,
    wopbs_key: &WopbsKey,
    offset: i32,
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
            // Eval additional LUT if output bit-width is greater than
            // wave_depth bits
            if encoded_luts.len() > 2 {
                let lut_msb = wopbs_key.wopbs(&msb, &encoded_luts[2]);
                let lut_msb_blocks = wopbs_key.keyswitch_to_pbs_params(&lut_msb).into_blocks();
                lut_lsb_blocks.extend(lut_msb_blocks);
            }
            // Pad LUT output and LSB by 6 bits to avoid overflows
            let padding_ct_block = server_key
                .create_trivial_zero_radix::<RadixCiphertext>(3)
                .into_blocks();
            lut_lsb_blocks.extend(padding_ct_block.clone());
            let mut lsb_blocks = lsb.clone().into_blocks();
            lsb_blocks.extend(padding_ct_block);
            let mut lut_combined = RadixCiphertext::from_blocks(lut_lsb_blocks);
            let lsb_extended = RadixCiphertext::from_blocks(lsb_blocks);

            // subtract offset (if necessary)
            if (offset.abs() as u64) > 0 {
                lut_combined =
                    server_key.scalar_sub_parallelized(&lut_combined, offset.abs() as u64);
            }

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
            // Eval additional LUT if output bit-width is greater than
            // wave_depth bits
            if encoded_luts.len() > 2 {
                let lut_msb = wopbs_key.wopbs(&msb, &encoded_luts[3]);
                let lut_msb_blocks = wopbs_key.keyswitch_to_pbs_params(&lut_msb).into_blocks();
                lut_lsb_blocks.extend(lut_msb_blocks);
            }
            // Pad LUT output and LSB by 6 bits to avoid overflows
            let padding_ct_block = server_key
                .create_trivial_zero_radix::<RadixCiphertext>(3)
                .into_blocks();
            lut_lsb_blocks.extend(padding_ct_block.clone());
            let mut lsb_blocks = lsb.clone().into_blocks();
            lsb_blocks.extend(padding_ct_block);
            let mut lut_combined = RadixCiphertext::from_blocks(lut_lsb_blocks);
            let lsb_extended = RadixCiphertext::from_blocks(lsb_blocks);

            // subtract offset (if necessary)
            if (offset.abs() as u64) > 0 {
                lut_combined =
                    server_key.scalar_sub_parallelized(&lut_combined, offset.abs() as u64);
            }
            // l2 = lsb
            // Multiply MSBs and LSBs
            server_key.mul_parallelized(&lut_combined, &lsb_extended)
        },
    );
    let probability = server_key.add_parallelized(&output_1, &output_2);
    (probability, start.elapsed().as_secs_f64())
}

pub fn ct_lut_eval_bior_no_gen(
    ct: RadixCiphertext,
    bit_width: usize,
    nb_blocks: usize,
    wave_depth: usize,
    wopbs_key: &WopbsKey,
    offset: i32,
    server_key: &ServerKey,
    encoded_luts: &Vec<IntegerWopbsLUT>,
) -> RadixCiphertext {
    let nb_blocks_lsb = (bit_width - wave_depth) >> 1;
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
            // Eval additional LUT if output bit-width is greater than
            // wave_depth bits
            if encoded_luts.len() > 2 {
                let lut_msb = wopbs_key.wopbs(&msb, &encoded_luts[2]);
                let lut_msb_blocks = wopbs_key.keyswitch_to_pbs_params(&lut_msb).into_blocks();
                lut_lsb_blocks.extend(lut_msb_blocks);
            }
            // Pad LUT output and LSB by 6 bits to avoid overflows
            let padding_ct_block = server_key
                .create_trivial_zero_radix::<RadixCiphertext>(3)
                .into_blocks();
            lut_lsb_blocks.extend(padding_ct_block.clone());
            let mut lsb_blocks = lsb.clone().into_blocks();
            lsb_blocks.extend(padding_ct_block);
            let mut lut_combined = RadixCiphertext::from_blocks(lut_lsb_blocks);
            let lsb_extended = RadixCiphertext::from_blocks(lsb_blocks);

            // subtract offset (if necessary)
            if (offset.abs() as u64) > 0 {
                lut_combined =
                    server_key.scalar_sub_parallelized(&lut_combined, offset.abs() as u64);
            }

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
            // Eval additional LUT if output bit-width is greater than
            // wave_depth bits
            if encoded_luts.len() > 2 {
                let lut_msb = wopbs_key.wopbs(&msb, &encoded_luts[3]);
                let lut_msb_blocks = wopbs_key.keyswitch_to_pbs_params(&lut_msb).into_blocks();
                lut_lsb_blocks.extend(lut_msb_blocks);
            }
            // Pad LUT output and LSB by 6 bits to avoid overflows
            let padding_ct_block = server_key
                .create_trivial_zero_radix::<RadixCiphertext>(3)
                .into_blocks();
            lut_lsb_blocks.extend(padding_ct_block.clone());
            let mut lsb_blocks = lsb.clone().into_blocks();
            lsb_blocks.extend(padding_ct_block);
            let mut lut_combined = RadixCiphertext::from_blocks(lut_lsb_blocks);
            let lsb_extended = RadixCiphertext::from_blocks(lsb_blocks);

            // subtract offset (if necessary)
            if (offset.abs() as u64) > 0 {
                lut_combined =
                    server_key.scalar_sub_parallelized(&lut_combined, offset.abs() as u64);
            }
            // l2 = lsb
            // Multiply MSBs and LSBs
            server_key.mul_parallelized(&lut_combined, &lsb_extended)
        },
    );
    server_key.add_parallelized(&output_1, &output_2)
}
