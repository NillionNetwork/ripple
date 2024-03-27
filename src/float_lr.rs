use fhe_lut::common::*;

fn main() {
    let (model, bias) = load_weights_and_biases();
    let (dataset, targets) = prepare_penguins_dataset();

    let mut total = 0;
    for (num, (sample, target)) in (dataset.iter().zip(targets.iter())).enumerate() {
        let mut prediction = bias;
        for (&s, w) in sample.iter().zip(model.iter()) {
            prediction += w * s;
        }
        let sigmoid = 1f64 / (1f64 + (-prediction).exp());
        let class = (sigmoid > 0.5) as usize;
        println!(
            "[{}] predicted {:?}, target {:?} probabilities {:?}",
            num, class, target, sigmoid
        );
        if class == *target {
            total += 1;
        }
    }

    let accuracy = (total as f64 / dataset.len() as f64) * 100.0;
    println!("Accuracy {accuracy}%");
}
