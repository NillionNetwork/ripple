use ripple::common;

/// d(x, y) = sqrt( sum((xi - yi)^2) )
fn euclidean(x: &[u32], y: &[u32]) -> f32 {
    x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - yi).pow(2) as f32)
        .sum::<f32>()
        .sqrt()
}

fn main() {
    let (x, y) = common::read_csv_two_columns("data/euclidean.csv");

    if x.len() != y.len() {
        panic!("The length of the two arrays must be equal");
    }
    let distance = euclidean(&x, &y);
    println!("Euclidean distance: {}", distance);
}
