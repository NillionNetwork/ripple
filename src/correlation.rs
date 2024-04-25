use ripple::common;

fn pearson_correlation(x: &[u32], y: &[u32]) -> Result<f64, &'static str> {
    if x.len() != y.len() {
        return Err("The length of the two arrays must be equal");
    }

    let n = x.len() as f64;

    let x_mean = x.iter().map(|&xi| xi as f64).sum::<f64>() / n;
    let y_mean = y.iter().map(|&yi| yi as f64).sum::<f64>() / n;

    let sum_xy: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| ((xi as f64) - x_mean) * ((yi as f64) - y_mean))
        .sum();
    let sum_x_squared: f64 = x.iter().map(|&xi| ((xi as f64) - x_mean).powi(2)).sum();
    let sum_y_squared: f64 = y.iter().map(|&yi| ((yi as f64) - y_mean).powi(2)).sum();

    Ok(sum_xy / (sum_x_squared.sqrt() * sum_y_squared.sqrt()))
}

fn main() {
    let (experience, salary) = common::read_correlation("data/correlation.csv");

    let mut salary_sorted = salary.clone();
    salary_sorted.sort();

    match pearson_correlation(&experience, &salary) {
        Ok(result) => println!("Pearson's r (unrelated): {}", result),
        Err(err) => println!("Error: {}", err),
    }

    match pearson_correlation(&experience, &salary_sorted) {
        Ok(result) => println!("Pearson's r (correlated): {}", result),
        Err(err) => println!("Error: {}", err),
    }
}
