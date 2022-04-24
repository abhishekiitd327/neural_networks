pub mod tensor;
pub mod neural_networks;

use crate::tensor::Tensor2D;
use crate::neural_networks::NeuralNetwork;

use std::fs;

pub fn current_millis() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

pub fn read_csv(
    filename: &str,
    input_size: usize,
    output_size: usize,
) -> (Vec<Tensor2D>, Vec<Tensor2D>) {
    let mut inputs = Vec::new();
    let mut labels = Vec::new();
    let content = fs::read_to_string(filename).expect("Error opening file");
    let lines: Vec<&str> = content.lines().collect();
    for idx in 0..lines.len() {
        let line = lines[idx];
        let values: Vec<&str> = line.split(",").collect();
        let mut input = Tensor2D::new(input_size, 1);
        let mut label = Tensor2D::new(output_size, 1);
        for value_index in 0..values.len() {
            if value_index < input_size {
                input[value_index][0] = values[value_index].parse::<f32>().unwrap();
            } else {
                label[value_index - input_size][0] =
                    values[value_index].parse::<f32>().unwrap();
            }
        }
        inputs.push(input);
        labels.push(label);
    }
    (inputs, labels)
}

pub fn get_accuracy(nn: &NeuralNetwork, filename: &str) -> f32 {
    let (inputs, labels) = read_csv(filename, nn.input_nodes, nn.output_nodes);
    let mut count: usize = 0;
    for i in 0..inputs.len() {
        if nn.predict(&inputs[i]).index_of_max() == labels[i].index_of_max() {
            count += 1;
        }
    }
    count as f32 / inputs.len() as f32
}

pub fn train(nn: &mut NeuralNetwork, filename: &str, num_epochs: u32) {
    let (inputs, labels) = read_csv(filename, nn.input_nodes, nn.output_nodes);
    let ts = current_millis();
    println!("Training started ...");
    for i in 0..num_epochs {
        for j in 0..inputs.len() {
            nn.train_one_epoch(&inputs[j], &labels[j]);
        }
        println!("Epoch [{} / {}] completed.", i + 1, num_epochs);
    }
    let tt = (current_millis() - ts) as f32 / 1000 as f32;
    println!("Training time: {}s", tt);
}
