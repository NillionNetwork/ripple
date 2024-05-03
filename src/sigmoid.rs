use ripple::common::*;

fn main() {
    let bit_width = 16;
    let precision = 12;
    let table_size = 8;

    println!("Generating Lookup Tables");
    let (lut_lsb, lut_msb) = quantized_table(table_size, precision, precision, bit_width);

    fn my_sigmoid(value: f64) -> f64 {
        1f64 / (1f64 + (-value).exp())
    }
    let (lut_haar_lsb, lut_haar_msb) = haar(precision, precision, bit_width, &my_sigmoid);

    let mut diff_quant = Vec::new();
    let mut diff_haar = Vec::new();
    // let dataset: Vec<u64> = vec![0, 72, 1050, 1790, 10234, 60122, 65001, 65535];
    let max = (1 << bit_width) - 1;
    println!("Evaluating Sigmoid");
    for x in 0..max {
        let s0 = sigmoid(x, precision, precision, bit_width);
        let _s1 = sigmoid(
            trunc(x, bit_width, bit_width - table_size),
            precision - (bit_width - table_size),
            precision,
            bit_width,
        );
        let s2a = lut_lsb[(x >> (bit_width - table_size)) as usize];
        let s2b = lut_msb[(x >> (bit_width - table_size)) as usize];
        let s2 = s2a + (s2b << (bit_width / 2));
        let diff = s0 as i64 - s2 as i64;
        diff_quant.push(diff * diff);
        let s3a = lut_haar_lsb[(x >> (bit_width - table_size)) as usize];
        let s3b = lut_haar_msb[(x >> (bit_width - table_size)) as usize];
        let s3 = s3a + (s3b << (bit_width / 2));
        let diff = s0 as i64 - s3 as i64;
        diff_haar.push(diff * diff);
        // println!("{x} {s0} {_s1} {s2} {s3}");
    }
    println!(
        "=== Quantized ===\nMean Squared Error {:?} \nMaximum Absolute Error {:?}",
        (diff_quant.iter().sum::<i64>() as f64) / diff_quant.len() as f64,
        (*diff_quant.iter().max().unwrap() as f64).sqrt()
    );
    println!(
        "=== Haar DWT ====\nMean Squared Error {:?} \nMaximum Absolute Error {:?}",
        (diff_haar.iter().sum::<i64>() as f64) / diff_haar.len() as f64,
        (*diff_haar.iter().max().unwrap() as f64).sqrt()
    );
}
