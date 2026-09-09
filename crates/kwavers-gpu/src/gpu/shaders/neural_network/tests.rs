use super::*;
use eunomia::assert_relative_eq;
use hephaestus_core::DevicePreference;

fn create_test_gpu_device() -> Option<GpuDevice> {
    match GpuDevice::create(DevicePreference::HighPerformance) {
        Ok(device) => Some(device),
        Err(error) if error.is_gpu_absent() => {
            eprintln!("no compatible GPU adapter on this host, skipping");
            None
        }
        Err(error) => panic!(
            "a present GPU adapter failed device creation, which is a defect rather than absent hardware: {error}"
        ),
    }
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
        return;
    };

    let shader = NeuralNetworkShader::new(&device).unwrap();
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
        return;
    };

    let shader = NeuralNetworkShader::new(&device).unwrap();
    let err = shader.activate(&[1.0_f32, 2.0, 3.0], 99).unwrap_err();
    assert!(format!("{err:?}").contains("Unknown activation type"));
}
