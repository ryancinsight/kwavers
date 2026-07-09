use std::collections::HashMap;

use super::*;

#[test]
fn test_quantizer_creation() {
    let quantizer = MlQuantizer::new(QuantizationScheme::Dynamic8Bit);
    assert_eq!(quantizer.accuracy_tolerance, 0.05);
}

#[test]
fn test_quantization_schemes() {
    let schemes = vec![
        QuantizationScheme::None,
        QuantizationScheme::Dynamic8Bit,
        QuantizationScheme::Static8Bit {
            calibration_data: vec![vec![1.0, 2.0, 3.0]],
        },
        QuantizationScheme::MixedPrecision {
            weight_bits: 8,
            activation_bits: 8,
        },
        QuantizationScheme::Adaptive {
            accuracy_threshold: 0.05,
            max_bits: 8,
        },
    ];

    for scheme in schemes {
        let quantizer = MlQuantizer::new(scheme);
        assert!(quantizer.accuracy_tolerance >= 0.0);
    }
}

#[test]
fn test_quantized_tensor_creation() {
    let data = vec![1.0, -1.0, 0.5, -0.5];
    let shape = vec![1, 4];

    let quantizer = MlQuantizer::new(QuantizationScheme::Dynamic8Bit);
    let result = quantizer.quantize_tensor(&data, &shape);

    let quantized = result.unwrap();
    assert_eq!((quantized.shape()[0] * quantized.shape()[1] * quantized.shape()[2]), 4);
    assert!(quantized.scale > 0.0);
}

#[test]
fn test_compression_ratio_calculation() {
    let model = QuantizedModel {
        original_layers: vec![LayerInfo {
            name: "test".to_string(),
            input_size: 10,
            output_size: 10,
            activation: "tanh".to_string(),
        }],
        quantized_weights: vec![QuantizedTensor {
            data: QuantizedData::I8(vec![1i8; 110]),
            scale: 1.0,
            zero_point: 0,
            shape: vec![10, 11],
        }],
        quantization_params: QuantizationParams {
            global_scale: 1.0,
            layer_scales: HashMap::new(),
            scheme: QuantizationScheme::Dynamic8Bit,
        },
        metadata: QuantizationModelMetadata {
            original_accuracy: 0.95,
            quantized_accuracy: 0.92,
            compression_ratio: 4.0,
            inference_speedup: 2.5,
        },
    };

    let ratio = model.compression_ratio();
    assert!(ratio > 1.0);
}
