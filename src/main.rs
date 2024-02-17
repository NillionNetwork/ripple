use std::fs::File;

fn quantize(x: f32, precision: u8) -> i32 {
    (x * ((1 << precision) as f32)) as i32
}

fn sigmoid(x: i32, input_precision: u8, output_precision: u8) -> i32 {
    let exp = (-x as f32 / ((1 << input_precision) as f32)).exp();
    ((1.0 / (1.0 + exp)) * (1 << output_precision) as f32) as i32
}

fn argmax<T: PartialOrd>(slice: &[T]) -> Option<usize> {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
}

fn load_weights_and_biases() -> (Vec<Vec<f32>>, Vec<f32>) {
    let weights_csv = File::open("iris_weights.csv").unwrap();
    let mut reader = csv::Reader::from_reader(weights_csv);
    let mut weights = vec![];
    let mut biases = vec![];

    for result in reader.deserialize() {
        let res: Vec<f32> = result.expect("a CSV record");
        biases.push(res[0]);
        weights.push(res[1..].to_vec());
    }

    (weights, biases)
}

fn quantize_weights_and_biases(
    weights: &[Vec<f32>],
    biases: &[f32],
    precision: u8,
) -> (Vec<Vec<i32>>, Vec<i32>) {
    let weights_int = weights
        .iter()
        .map(|row| {
            row.iter()
                .map(|&w| quantize(w, precision))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let bias_int = biases
        .iter()
        .map(|&w| quantize(w, precision))
        .collect::<Vec<_>>();

    (weights_int, bias_int)
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
    let precision = 8;
    let (weights, biases) = load_weights_and_biases();
    let (weights_int, bias_int) = quantize_weights_and_biases(&weights, &biases, precision);

    let iris_dataset = prepare_iris_dataset();
    let num_features = iris_dataset[0].0.len();
    let (means, stds) = means_and_stds(&iris_dataset, num_features);

    let mut total = 0;
    for (num, (sample, target)) in iris_dataset.iter().enumerate() {
        let mut probabilities = vec![];
        for (model, bias) in weights_int.iter().zip(bias_int.iter()) {
            let mut prediction = 0;
            for ((&s, w), (mean, std)) in sample
                .iter()
                .zip(model.iter())
                .zip(means.iter().zip(stds.iter()))
            {
                let n = (s - mean) / std;
                let quantized = quantize(n as f32, precision);
                prediction += w * quantized + bias;
            }
            let activation = sigmoid(prediction, precision * 2, precision);
            probabilities.push(activation);
        }
        let class = argmax(&probabilities).unwrap();
        println!("[{}] predicted {:?}, target {:?}", num, class, target);
        if class == *target {
            total += 1;
        }
    }

    let accuracy = (total as f32 / iris_dataset.len() as f32) * 100.0;
    println!("Accuracy {accuracy}%");
}
