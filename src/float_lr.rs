use std::fs::File;

fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

fn load_weights_and_biases() -> (Vec<Vec<f64>>, Vec<f64>) {
    let weights_csv = File::open("iris_weights.csv").unwrap();
    let mut reader = csv::Reader::from_reader(weights_csv);
    let mut weights = vec![];
    let mut biases = vec![];

    for result in reader.deserialize() {
        let res: Vec<f64> = result.expect("a CSV record");
        biases.push(res[0]);
        weights.push(res[1..].to_vec());
    }

    (weights, biases)
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
    let (weights, biases) = load_weights_and_biases();

    let iris_dataset = prepare_iris_dataset();
    let num_features = iris_dataset[0].0.len();
    let (means, stds) = means_and_stds(&iris_dataset, num_features);

    let mut total = 0;
    for (num, (sample, target)) in iris_dataset.iter().enumerate() {
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
        println!("[{}] predicted {:?}, target {:?} probabilities {:?}", num, class, target, norm);
        if class == *target {
            total += 1;
        }
    }

    let accuracy = (total as f64 / iris_dataset.len() as f64) * 100.0;
    println!("Accuracy {accuracy}%");
}
