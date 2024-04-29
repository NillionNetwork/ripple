use std::{collections::HashMap, fs::File, io::BufReader};

use dwt::{transform, wavelet::Haar, Operation};

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
    table_size: u8,
    input_precision: u8,
    output_precision: u8,
    bit_width: u8,
) -> (Vec<u64>, Vec<u64>) {
    let max = 1 << bit_width;
    let mut data = Vec::new();
    for x in 0..max {
        let x = unquantize(x, input_precision, bit_width);
        let sig = 1f64 / (1f64 + (-x).exp());
        data.push(sig);
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

pub fn bior(table_size: u8, bit_width: u8) -> (Vec<u64>, Vec<u64>) {
    // Read Biorthogonal LUT
    let reader = BufReader::new(File::open("./data/bior_lut.json").unwrap());
    let bior_lut: HashMap<u64, u64> = serde_json::from_reader(reader).unwrap();

    // Convert to 1-D vector
    bior_lut.into_iter().map(|(_, v)| v).collect::<Vec<_>>()

    // Break into two LUTs
    bior_lut.rotate_right(1 << (table_size - 1));
    let mask = (1 << (bit_width / 2)) - 1;
    let lsb = bior_lut.iter().map(|x| x & mask).collect();
    let msb = bior_lut.iter().map(|x| x >> (bit_width / 2) & mask).collect();
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
        db2_lut_1.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
        db2_lut_2.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
        db2_lut_3.into_iter().map(|(_, v)| v).collect::<Vec<_>>(),
    ];

    // Convert MSB LUT to 1-D vector
    let lut_msb_vec = db2_lut_4.into_iter().map(|(_, v)| v).collect::<Vec<_>>()

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
