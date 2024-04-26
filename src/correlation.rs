use ripple::common;

fn pearson_correlation(x: &[u32], y: &[u32]) -> f64 {
    let n = x.len() as f64;

    let x_mean = x.iter().map(|&xi| xi as f64).sum::<f64>() / n;
    let y_mean = y.iter().map(|&yi| yi as f64).sum::<f64>() / n;

    let sum_xy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| ((xi as f64) - x_mean) * ((yi as f64) - y_mean))
        .sum();
    let variance_x: f64 = x.iter().map(|&xi| ((xi as f64) - x_mean).powi(2)).sum();
    let variance_y: f64 = y.iter().map(|&yi| ((yi as f64) - y_mean).powi(2)).sum();

    sum_xy / (variance_x.sqrt() * variance_y.sqrt())
}

fn main() {
    let (experience, salary) = common::read_csv_two_columns("data/correlation.csv");
    if experience.len() != salary.len() {
        panic!("The length of the two arrays must be equal");
    }

    let mut salary_sorted = salary.clone();
    salary_sorted.sort();

    let result = pearson_correlation(&experience, &salary);
    println!("Pearson's r (unrelated): {}", result);

    let result = pearson_correlation(&experience, &salary_sorted);
    println!("Pearson's r (correlated): {}", result);
}
