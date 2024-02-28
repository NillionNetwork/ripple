use fhe_lut::common::*;

fn main() {
    let (weights, biases) = load_weights_and_biases();

    let (iris_dataset, targets) = prepare_iris_dataset();
    let num_features = iris_dataset[0].len();
    let (means, stds) = means_and_stds(&iris_dataset, num_features);

    let mut total = 0;
    for (num, (sample, target)) in (iris_dataset.iter().zip(targets.iter())).enumerate() {
        let mut probabilities = vec![];
        let mut sum_p = 0f64;
        for (model, bias) in weights.iter().zip(biases.iter()) {
            let mut prediction = *bias;
            for ((&s, w), (mean, std)) in sample
                .iter()
                .zip(model.iter())
                .zip(means.iter().zip(stds.iter()))
            {
                let n = (s - mean) / std;
                prediction += w * n;
            }
            let activation = prediction.exp();
            probabilities.push(activation);
            sum_p += activation;
        }
        let mut norm = Vec::new();
        for p in &probabilities {
            norm.push(p / sum_p);
        }
        let class = argmax(&probabilities).unwrap();
        println!(
            "[{}] predicted {:?}, target {:?} probabilities {:?}",
            num, class, target, norm
        );
        if class == *target {
            total += 1;
        }
    }

    let accuracy = (total as f64 / iris_dataset.len() as f64) * 100.0;
    println!("Accuracy {accuracy}%");
}
