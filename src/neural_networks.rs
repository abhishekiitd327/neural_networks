use std::fs;

use std::io::Read;
use std::io::Write;
use std::mem;

use crate::tensor::Tensor2D;

pub struct NeuralNetwork {
    pub input_nodes: usize,
    pub hidden_nodes: Vec<usize>,
    pub output_nodes: usize,
    input_to_hidden_weights: Tensor2D,
    hidden_to_hidden_weights: Vec<Tensor2D>,
    hidden_to_output_weights: Tensor2D,
    hidden_biases: Vec<Tensor2D>,
    output_bias: Tensor2D,
    learning_rate: f32,
    activation_function: fn(f32) -> f32,
    activation_function_derivative: fn(f32) -> f32,
}

impl NeuralNetwork {
    pub fn new(
        input_nodes: usize,
        hidden_nodes: Vec<usize>,
        output_nodes: usize,
        learning_rate: f32,
        activation_function_name: &str,
    ) -> NeuralNetwork {
        let mut hidden_to_hidden_weights = Vec::new();
        for hidden_index in 0..hidden_nodes.len() - 1 {
            hidden_to_hidden_weights.push(Tensor2D::new_rand(
                hidden_nodes[hidden_index + 1],
                hidden_nodes[hidden_index],
            ));
        }
        let mut hidden_biases = Vec::new();
        for hidden_index in 0..hidden_nodes.len() {
            hidden_biases.push(Tensor2D::new_rand(hidden_nodes[hidden_index], 1));
        }
        let (activation_function, activation_function_derivative) = match activation_function_name {
            "sigmoid" => (
                sigmoid as fn(f32) -> f32,
                sigmoid_derivative as fn(f32) -> f32,
            ),
            _ => panic!(
                "Error: Activation function {} is not supported.",
                activation_function_name
            ),
        };
        let nn = NeuralNetwork {
            input_to_hidden_weights: Tensor2D::new_rand(hidden_nodes[0], input_nodes),
            hidden_to_hidden_weights: hidden_to_hidden_weights,
            hidden_to_output_weights: Tensor2D::new_rand(
                output_nodes,
                hidden_nodes[hidden_nodes.len() - 1],
            ),
            hidden_biases: hidden_biases,
            output_bias: Tensor2D::new_rand(output_nodes, 1),
            input_nodes: input_nodes,
            hidden_nodes: hidden_nodes,
            output_nodes: output_nodes,
            learning_rate: learning_rate,
            activation_function: activation_function,
            activation_function_derivative: activation_function_derivative,
        };
        nn
    }

    pub fn train_one_epoch(&mut self, input: &Tensor2D, expected_output: &Tensor2D) {
        let first_hidden_state: Tensor2D =
            &(&self.input_to_hidden_weights * &input) + &self.hidden_biases[0];
        let first_hidden_activation: Tensor2D =
            first_hidden_state.apply_function(self.activation_function);
        let mut hidden_states = Vec::new();
        let mut hidden_activations = Vec::new();
        hidden_states.push(first_hidden_state);
        hidden_activations.push(first_hidden_activation);
        for hidden_index in 0..self.hidden_to_hidden_weights.len() {
            let next_hidden_state = &(&self.hidden_to_hidden_weights[hidden_index]
                * &hidden_activations[hidden_activations.len() - 1])
                + &self.hidden_biases[hidden_index + 1];
            let next_hidden_activation = next_hidden_state.apply_function(self.activation_function);
            hidden_states.push(next_hidden_state);
            hidden_activations.push(next_hidden_activation);
        }
        let output_state = &(&self.hidden_to_output_weights
            * &hidden_activations[hidden_activations.len() - 1])
            + &self.output_bias;
        let output_activation = output_state.apply_function(self.activation_function);
        // backprop
        let output_error = expected_output - &output_activation;

        let gradient = Tensor2D::hadamard(
            &output_error,
            &output_state.apply_function(self.activation_function_derivative),
        );
        let mut current_hidden_error = &self.hidden_to_output_weights.transpose() * &gradient;
        self.hidden_to_output_weights += (&gradient
            * &hidden_activations[hidden_activations.len() - 1].transpose())
            * self.learning_rate;
        self.output_bias += gradient * self.learning_rate;

        for hidden_index in (0..self.hidden_to_hidden_weights.len()).rev() {
            let gradient = Tensor2D::hadamard(
                &current_hidden_error,
                &hidden_states[hidden_index + 1]
                    .apply_function(self.activation_function_derivative),
            );
            current_hidden_error =
                &self.hidden_to_hidden_weights[hidden_index].transpose() * &gradient;
            self.hidden_to_hidden_weights[hidden_index] +=
                (&gradient * &hidden_activations[hidden_index].transpose()) * self.learning_rate;
            self.hidden_biases[hidden_index + 1] += gradient * self.learning_rate;
        }

        let gradient = Tensor2D::hadamard(
            &current_hidden_error,
            &hidden_states[0].apply_function(self.activation_function_derivative),
        );
        self.input_to_hidden_weights += (&gradient * &input.transpose()) * self.learning_rate;
        self.hidden_biases[0] += gradient * self.learning_rate;
    }

    pub fn predict(&self, input: &Tensor2D) -> Tensor2D {
        let first_hidden_state: Tensor2D =
            &(&self.input_to_hidden_weights * &input) + &self.hidden_biases[0];
        let first_hidden_activation: Tensor2D =
            first_hidden_state.apply_function(self.activation_function);
        let mut hidden_states = Vec::new();
        let mut hidden_activations = Vec::new();
        hidden_states.push(first_hidden_state);
        hidden_activations.push(first_hidden_activation);
        for hidden_index in 0..self.hidden_to_hidden_weights.len() {
            let next_hidden_state = &(&self.hidden_to_hidden_weights[hidden_index]
                * &hidden_activations[hidden_activations.len() - 1])
                + &self.hidden_biases[hidden_index + 1];
            let next_hidden_activation = next_hidden_state.apply_function(self.activation_function);
            hidden_states.push(next_hidden_state);
            hidden_activations.push(next_hidden_activation);
        }
        let output_state = &(&self.hidden_to_output_weights
            * &hidden_activations[hidden_activations.len() - 1])
            + &self.output_bias;
        let output_activation = output_state.apply_function(self.activation_function);
        output_activation
    }

    pub fn save_model(&self, filename: &str) {
        // helper functions
        fn write_usize(file: &mut std::fs::File, value: &usize) {
            file.write(&value.to_be_bytes()).ok();
        }
        fn write_f32(file: &mut std::fs::File, value: &f32) {
            file.write(&value.to_be_bytes()).ok();
        }
        fn write_matrix(file: &mut std::fs::File, mat: &Tensor2D) {
            write_usize(file, &mat.rows);
            write_usize(file, &mat.cols);
            for r in 0..mat.rows {
                for c in 0..mat.cols {
                    write_f32(file, &mat[r][c]);
                }
            }
        }

        let mut file = fs::File::create(filename).expect("Error: Can't save model.");
        // write model
        write_usize(&mut file, &self.input_nodes);
        write_usize(&mut file, &self.hidden_nodes.len());
        for current_hidden_layer_size in &self.hidden_nodes {
            write_usize(&mut file, &current_hidden_layer_size);
        }
        write_usize(&mut file, &self.output_nodes);
        write_f32(&mut file, &self.learning_rate);
        let mut activation_function_id: usize = 0;
        if self.activation_function == sigmoid {
            activation_function_id = 0
        }
        write_usize(&mut file, &activation_function_id);
        // write weights
        write_matrix(&mut file, &self.input_to_hidden_weights);
        for mat in &self.hidden_to_hidden_weights {
            write_matrix(&mut file, &mat);
        }
        write_matrix(&mut file, &self.hidden_to_output_weights);
        // write biases
        for mat in &self.hidden_biases {
            write_matrix(&mut file, &mat);
        }
        write_matrix(&mut file, &self.output_bias);
    }

    pub fn load_model(filename: &str) -> NeuralNetwork {
        
        fn read_usize(file: &mut std::fs::File) -> usize {
            let mut buffer = [0; mem::size_of::<usize>()];
            let _ = &file.read_exact(&mut buffer).ok();
            usize::from_be_bytes(buffer)
        }
        fn read_f32(file: &mut std::fs::File) -> f32 {
            let mut buffer = [0; mem::size_of::<f32>()];
            let _ = &file.read_exact(&mut buffer).ok();
            f32::from_be_bytes(buffer)
        }
        fn read_matrix(file: &mut std::fs::File) -> Tensor2D {
            let rows = read_usize(file);
            let cols = read_usize(file);
            let mut mat = Tensor2D::new(rows, cols);
            for r in 0..mat.rows {
                for c in 0..mat.cols {
                    mat[r][c] = read_f32(file);
                }
            }
            mat
        }
        let mut file = fs::File::open(filename).expect("Error: Can't load model.");
        // read model
        let input_nodes = read_usize(&mut file);
        let num_hidden_layers = read_usize(&mut file);
        let mut hidden_nodes = Vec::new();
        for _ in 0..num_hidden_layers {
            hidden_nodes.push(read_usize(&mut file));
        }
        let output_nodes = read_usize(&mut file);
        let learning_rate = read_f32(&mut file);
        let (activation_function, activation_function_derivative) = match read_usize(&mut file) {
            0 => (
                sigmoid as fn(f32) -> f32,
                sigmoid_derivative as fn(f32) -> f32,
            ),
            _ => panic!("Error: Can't load model."),
        };
        // read weights
        let input_to_hidden_weights = read_matrix(&mut file);
        let mut hidden_to_hidden_weights = Vec::new();
        for _ in 0..num_hidden_layers - 1 {
            hidden_to_hidden_weights.push(read_matrix(&mut file));
        }
        let hidden_to_output_weights = read_matrix(&mut file);
        // read biases
        let mut hidden_biases = Vec::new();
        for _ in 0..num_hidden_layers {
            hidden_biases.push(read_matrix(&mut file));
        }
        let output_bias = read_matrix(&mut file);
        let nn = NeuralNetwork {
            input_to_hidden_weights: input_to_hidden_weights,
            hidden_to_hidden_weights: hidden_to_hidden_weights,
            hidden_to_output_weights: hidden_to_output_weights,
            hidden_biases: hidden_biases,
            output_bias: output_bias,
            input_nodes: input_nodes,
            hidden_nodes: hidden_nodes,
            output_nodes: output_nodes,
            learning_rate: learning_rate,
            activation_function: activation_function,
            activation_function_derivative: activation_function_derivative,
        };
        nn
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f32) -> f32 {
    (1.0 - sigmoid(x)) * sigmoid(x)
}
