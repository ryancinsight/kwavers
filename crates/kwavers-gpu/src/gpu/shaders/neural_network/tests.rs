use super::*;
use eunomia::assert_relative_eq;
use hephaestus_core::DevicePreference;

fn create_test_gpu_device() -> Option<GpuDevice> {
    pollster::block_on(GpuDevice::create(DevicePreference::HighPerformance)).ok()
}

fn expected_activation(input: &[f32], activation_type: u32) -> Vec<f32> {
    match ActivationKind::from_u32(activation_type) {
        Some(ActivationKind::Relu) => input.iter().copied().map(|x| x.max(0.0)).collect(),
        Some(ActivationKind::Sigmoid) => input
            .iter()
            .copied()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect(),
        Some(ActivationKind::Tanh) => input.iter().copied().map(f32::tanh).collect(),
        Some(ActivationKind::Linear) => input.to_vec(),
        None => unreachable!("activation_type is validated in the test"),
    }
}

#[test]
fn test_gpu_activation_matches_contract() {
    let Some(device) = create_test_gpu_device() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    let shader = pollster::block_on(NeuralNetworkShader::new(&device)).unwrap();
    let input = vec![-2.0_f32, -0.5, 0.0, 0.5, 2.0];

    for activation_type in 0..=3 {
        let gpu = shader.activate(&input, activation_type).unwrap();
        let expected = expected_activation(&input, activation_type);
        assert_eq!(gpu.len(), expected.len());
        for (actual, reference) in gpu.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, reference, epsilon = 1e-5);
        }
    }
}

#[test]
fn test_gpu_activation_rejects_unknown_type() {
    let Some(device) = create_test_gpu_device() else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    let shader = pollster::block_on(NeuralNetworkShader::new(&device)).unwrap();
    let err = shader.activate(&[1.0_f32, 2.0, 3.0], 99).unwrap_err();
    assert!(format!("{err:?}").contains("Unknown activation type"));
}
