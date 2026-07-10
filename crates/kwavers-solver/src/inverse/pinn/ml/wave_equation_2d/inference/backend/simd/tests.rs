use super::SimdExecutor;
use crate::inverse::pinn::ml::wave_equation_2d::inference::types::{
    ActivationType, QuantizedNetwork, WaveInferenceMemoryPool2D,
};

/// Reference scalar implementation for validation
fn matmul_scalar_quantized(
    input: &[f32],
    weights: &[i8],
    weight_scale: f32,
    biases: &[i8],
    bias_scale: f32,
    batch_size: usize,
    input_size: usize,
    output_size: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; batch_size * output_size];

    for batch_idx in 0..batch_size {
        for out_idx in 0..output_size {
            let mut sum = 0.0;

            for i in 0..input_size {
                let input_val = input[batch_idx * input_size + i];
                let weight_val = weights[out_idx * input_size + i] as f32 * weight_scale;
                sum += input_val * weight_val;
            }

            let bias_val = biases[out_idx] as f32 * bias_scale;
            sum += bias_val;

            output[batch_idx * output_size + out_idx] = sum;
        }
    }

    output
}

/// Test SIMD matmul with input_size=3, output_size=3
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_matmul_simd_3x3() {
    let executor = SimdExecutor::new(16);

    let batch_size = 2;
    let input_size = 3;
    let output_size = 3;

    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let weights = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let weight_scale = 0.1;
    let biases = vec![10, 20, 30];
    let bias_scale = 0.01;

    let simd_result = executor
        .matmul_simd_quantized(
            &input,
            &weights,
            weight_scale,
            &biases,
            bias_scale,
            batch_size,
            input_size,
            output_size,
        )
        .unwrap();

    let scalar_result = matmul_scalar_quantized(
        &input,
        &weights,
        weight_scale,
        &biases,
        bias_scale,
        batch_size,
        input_size,
        output_size,
    );

    assert_eq!((simd_result.len()), (scalar_result.len()));
    for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
        assert!(
            (simd_val - scalar_val).abs() < 1e-5,
            "SIMD output {} != scalar output {}",
            simd_val,
            scalar_val
        );
    }
}

/// Test SIMD matmul with input_size=3, output_size=8 (hidden layer)
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_matmul_simd_3x8() {
    let executor = SimdExecutor::new(16);

    let batch_size = 2;
    let input_size = 3;
    let output_size = 8;

    let input = vec![1.0, 2.0, 3.0, 0.5, 1.5, 2.5];
    let weights: Vec<i8> = (0..24).map(|i| (i % 127) as i8).collect();
    let weight_scale = 0.05;
    let biases: Vec<i8> = (0..8).map(|i| (i * 5) as i8).collect();
    let bias_scale = 0.02;

    let simd_result = executor
        .matmul_simd_quantized(
            &input,
            &weights,
            weight_scale,
            &biases,
            bias_scale,
            batch_size,
            input_size,
            output_size,
        )
        .unwrap();

    let scalar_result = matmul_scalar_quantized(
        &input,
        &weights,
        weight_scale,
        &biases,
        bias_scale,
        batch_size,
        input_size,
        output_size,
    );

    assert_eq!((simd_result.len()), (scalar_result.len()));
    for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
        assert!(
            (simd_val - scalar_val).abs() < 1e-5,
            "SIMD output {} != scalar output {}",
            simd_val,
            scalar_val
        );
    }
}

/// Test SIMD matmul with input_size=16, output_size=16 (larger hidden layer)
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_matmul_simd_16x16() {
    let executor = SimdExecutor::new(16);

    let batch_size = 4;
    let input_size = 16;
    let output_size = 16;

    let input: Vec<f32> = (0..batch_size * input_size)
        .map(|i| (i as f32) * 0.1)
        .collect();
    let weights: Vec<i8> = (0..input_size * output_size)
        .map(|i| ((i * 7) % 127) as i8)
        .collect();
    let weight_scale = 0.03;
    let biases: Vec<i8> = (0..output_size).map(|i| (i * 3) as i8).collect();
    let bias_scale = 0.015;

    let simd_result = executor
        .matmul_simd_quantized(
            &input,
            &weights,
            weight_scale,
            &biases,
            bias_scale,
            batch_size,
            input_size,
            output_size,
        )
        .unwrap();

    let scalar_result = matmul_scalar_quantized(
        &input,
        &weights,
        weight_scale,
        &biases,
        bias_scale,
        batch_size,
        input_size,
        output_size,
    );

    assert_eq!((simd_result.len()), (scalar_result.len()));
    for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
        assert!(
            (simd_val - scalar_val).abs() < 1e-4,
            "SIMD output {} != scalar output {}",
            simd_val,
            scalar_val
        );
    }
}

/// Test SIMD matmul with input_size=32, output_size=1 (output layer from large hidden)
/// # Panics
/// - Panics if an internal invariant assumed to hold at this call site is violated.
///
#[test]
fn test_matmul_simd_32x1() {
    let executor = SimdExecutor::new(16);

    let batch_size = 8;
    let input_size = 32;
    let output_size = 1;

    let input: Vec<f32> = (0..batch_size * input_size)
        .map(|i| ((i as f32) * 0.05).sin())
        .collect();
    let weights: Vec<i8> = (0..input_size).map(|i| ((i * 11) % 127) as i8).collect();
    let weight_scale = 0.04;
    let biases = vec![42];
    let bias_scale = 0.01;

    let simd_result = executor
        .matmul_simd_quantized(
            &input,
            &weights,
            weight_scale,
            &biases,
            bias_scale,
            batch_size,
            input_size,
            output_size,
        )
        .unwrap();

    let scalar_result = matmul_scalar_quantized(
        &input,
        &weights,
        weight_scale,
        &biases,
        bias_scale,
        batch_size,
        input_size,
        output_size,
    );

    assert_eq!((simd_result.len()), (scalar_result.len()));
    for (simd_val, scalar_val) in simd_result.iter().zip(scalar_result.iter()) {
        assert!(
            (simd_val - scalar_val).abs() < 1e-4,
            "SIMD output {} != scalar output {}",
            simd_val,
            scalar_val
        );
    }
}

/// Integration test: full forward pass with multi-layer network
/// # Panics
/// - Panics if `Forward pass should succeed`.
///
#[test]
fn test_forward_simd_multilayer() {
    let mut executor = SimdExecutor::new(16);

    // Network: 3 → 8 → 4 → 1
    let layer_sizes = vec![3, 8, 4, 1];
    let num_layers = (layer_sizes.len()) - 1;

    // Create quantized network
    let mut weights = Vec::new();
    let mut weight_scales = Vec::new();
    let mut biases = Vec::new();
    let mut bias_scales = Vec::new();
    let mut activations = Vec::new();

    for i in 0..num_layers {
        let in_size = layer_sizes[i];
        let out_size = layer_sizes[i + 1];
        let w: Vec<i8> = (0..in_size * out_size)
            .map(|j| ((j * 13 + i * 7) % 127) as i8)
            .collect();
        let b: Vec<i8> = (0..out_size).map(|j| ((j * 5) % 50) as i8).collect();

        weights.push(w);
        weight_scales.push(0.05);
        biases.push(b);
        bias_scales.push(0.01);
        activations.push(if i < num_layers - 1 {
            ActivationType::Tanh
        } else {
            ActivationType::Linear
        });
    }

    let network = QuantizedNetwork {
        weights,
        weight_scales,
        biases,
        bias_scales,
        layer_sizes: layer_sizes.clone(),
        activations,
    };

    let mut memory_pool = WaveInferenceMemoryPool2D {
        buffers: vec![vec![0.0; 256]; num_layers],
        _buffer_sizes: layer_sizes[1..].to_vec(),
    };

    // Test inputs
    let x = vec![1.0, 2.0];
    let y = vec![3.0, 4.0];
    let t = vec![0.5, 1.0];

    let result = executor.predict(&network, &mut memory_pool, &x, &y, &t);

    let (predictions, uncertainties) = result.expect("Forward pass should succeed");
    assert_eq!((predictions.len()), 2);
    assert_eq!((uncertainties.len()), 2);
    assert!(predictions.iter().all(|&p| p.is_finite()));
}
