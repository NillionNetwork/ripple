use fhe_lut::common::*;

fn main() {
    let bit_width = 16;
    let precision = 12;
    let table_size = 8;

    println!("Starting Haar");
    let (lut_lsb, lut_msb) = quantized_table(table_size, precision, bit_width);
    let (lut_haar_lsb, lut_haar_msb) = haar(table_size, precision, bit_width);
    println!("Haar");

    let mut diff_quant = Vec::new();
    let mut diff_haar = Vec::new();
    // let dataset: Vec<u64> = vec![0, 72, 1050, 1790, 10234, 60122, 65001, 65535];
    let max = (1 << bit_width) - 1;
    for x in 0..max {
        let s0 = sigmoid(x, precision, precision, bit_width);
        let s1 = sigmoid(
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
        println!("{x} {s0} {s1} {s2} {s3}");
    }
    println!(
        "quantized mse {:?} ulp {:?}",
        (diff_quant.iter().sum::<i64>() as f64).sqrt(),
        (*diff_quant.iter().max().unwrap() as f64).sqrt()
    );
    println!(
        "haar mse {:?} ulp {:?}",
        (diff_haar.iter().sum::<i64>() as f64).sqrt(),
        (*diff_haar.iter().max().unwrap() as f64).sqrt()
    );
}
