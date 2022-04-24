# neural_networks
A neural network library written from in Rust.

MNIST Example:

Step 1: Add neural_networks="0.0.1" in Cargo.toml under dependecies.

Step 2: Download mnist dataset from the dataset folder from the project github. Extract it.

Step 3: Now you can use neural_networks functions as shown in the following mnist example:

```rust
use neural_networks::neural_networks::NeuralNetwork;
use neural_networks::{get_accuracy, train};

fn main() {
    let mut nn = NeuralNetwork::new(784, vec![8, 8], 10, 0.1, "sigmoid");
    train(&mut nn, "mnist_train.csv", 5);
    let acc = get_accuracy(&nn, "mnist_test.csv") * 100.0;
    println!("Accuarcy: {}%", acc);
}
```
