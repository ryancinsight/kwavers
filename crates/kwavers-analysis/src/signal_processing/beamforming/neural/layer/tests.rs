use super::*;

#[test]
fn test_neural_layer_creation() {
    let layer = NeuralLayer::new(64, 32).unwrap();
    assert_eq!(layer.input_size(), 64);
    assert_eq!(layer.output_size(), 32);
    assert_eq!(layer.weights().shape(), [64, 32]);
    assert_eq!(layer.biases().len(), 32);
}

#[test]
fn test_neural_layer_zero_sizes() {
    assert!(NeuralLayer::new(0, 32).is_err());
    assert!(NeuralLayer::new(64, 0).is_err());
    assert!(NeuralLayer::new(0, 0).is_err());
}

#[test]
fn test_xavier_initialization_bounds() {
    let layer = NeuralLayer::new(64, 32).unwrap();
    let limit = (6.0_f64 / (64.0_f64 + 32.0_f64)).sqrt();

    // All weights should be within [-limit, limit]
    for &weight in layer.weights().iter() {
        assert!(weight >= -limit as f32);
        assert!(weight <= limit as f32);
    }

    // All biases should be zero
    for &bias in layer.biases().iter() {
        assert_eq!(bias, 0.0);
    }
}

#[test]
fn test_neural_layer_forward() {
    let layer = NeuralLayer::new(8, 4).unwrap();
    let input = leto::Array3::ones((2, 3, 8)); // Batch=(2×3)=6, Features=8

    let output = layer.forward(&input).unwrap();

    assert_eq!(output.shape(), [2, 3, 4]);
    assert!(output.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_neural_layer_activation_range() {
    let layer = NeuralLayer::new(4, 4).unwrap();
    let input = leto::Array3::from_elem((2, 2, 4), 100.0); // Very large input

    let output = layer.forward(&input).unwrap();

    // Tanh saturates: output must be in [-1, 1]
    for &val in output.iter() {
        assert!(val >= -1.0, "Output {} < -1.0", val);
        assert!(val <= 1.0, "Output {} > 1.0", val);
    }
}

#[test]
fn test_neural_layer_dimension_mismatch() {
    use kwavers_core::error::KwaversError;

    let layer = NeuralLayer::new(8, 4).unwrap();
    let wrong_input = leto::Array3::ones((2, 3, 16)); // Wrong feature size (16 instead of 8)

    let result = layer.forward(&wrong_input);
    assert!(result.is_err());

    if let Err(KwaversError::DimensionMismatch(msg)) = result {
        assert!(msg.contains("expects input size 8"));
        assert!(msg.contains("got 16"));
    } else {
        panic!("Expected DimensionMismatch error");
    }
}

#[test]
fn test_neural_layer_forward_shape_preservation() {
    let layer = NeuralLayer::new(16, 8).unwrap();
    let input = leto::Array3::from_elem((5, 7, 16), 0.5);

    let output = layer.forward(&input).unwrap();

    // Spatial dimensions preserved, feature dimension transformed
    assert_eq!(output.shape()[0], 5);
    assert_eq!(output.shape()[1], 7);
    assert_eq!(output.shape()[2], 8);
}

#[test]
fn test_neural_layer_adaptation() {
    let mut layer = NeuralLayer::new(4, 2).unwrap();
    let initial_weights = layer.weights().clone();
    let initial_biases = layer.biases().clone();

    layer.adapt(0.5).unwrap();

    let scale = 1.0 - 0.01 * 0.5;
    let expected_weights = initial_weights.mapv(|w| w * scale);
    let expected_biases = initial_biases.mapv(|b| b * scale);

    assert_eq!(expected_weights, *layer.weights());
    assert_eq!(expected_biases, *layer.biases());
}

#[test]
fn test_neural_layer_adaptation_zero_gradient_is_noop() {
    let mut layer = NeuralLayer::new(3, 3).unwrap();
    let initial_weights = layer.weights().clone();
    let initial_biases = layer.biases().clone();

    layer.adapt(0.0).unwrap();

    assert_eq!(initial_weights, *layer.weights());
    assert_eq!(initial_biases, *layer.biases());
}

#[test]
fn test_tanh_activation_zero_input() {
    let layer = NeuralLayer::new(4, 4).unwrap();
    let input = leto::Array3::zeros((2, 2, 4));

    let output = layer.forward(&input).unwrap();

    // For zero input: tanh(b) where b ≈ 0 → tanh(0) = 0
    // (biases initialized to zero)
    assert!(output.iter().all(|&x| x.abs() < 0.1));
}

#[test]
fn test_layer_linearity_before_activation() {
    // Test that doubling input approximately doubles pre-activation output
    // (ignoring saturation effects)
    let layer = NeuralLayer::new(4, 4).unwrap();
    let input1 = leto::Array3::from_elem((2, 2, 4), 0.01);
    let input2 = leto::Array3::from_elem((2, 2, 4), 0.02);

    let output1 = layer.forward(&input1).unwrap();
    let output2 = layer.forward(&input2).unwrap();

    // For small inputs, tanh(x) ≈ x (linear regime)
    // So output should be approximately linear in input
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..4 {
                let ratio = output2[[i, j, k]] / (output1[[i, j, k]] + 1e-8);
                // Should be close to 2.0 in linear regime
                assert!(ratio > 1.5 && ratio < 2.5, "Ratio: {}", ratio);
            }
        }
    }
}
