use neural_networks::neural_networks::NeuralNetwork;
use neural_networks::{get_accuracy, train};

fn main() {
    let mut nn = NeuralNetwork::new(784, vec![8, 8], 10, 0.1, "sigmoid");
    train(&mut nn, "mnist_train.csv", 5);
    let acc = get_accuracy(&nn, "mnist_test.csv") * 100.0;
    println!("Accuarcy: {}%", acc);
}
