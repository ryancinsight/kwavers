use super::*;
use approx::assert_relative_eq;

async fn create_test_gpu_device() -> Option<GpuDevice> {
    GpuDevice::create(wgpu::PowerPreference::HighPerformance)
        .await
        .ok()
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

#[tokio::test]
async fn test_gpu_activation_matches_contract() {
    let Some(device) = create_test_gpu_device().await else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    let shader = NeuralNetworkShader::new(&device).await.unwrap();
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

#[tokio::test]
async fn test_gpu_activation_rejects_unknown_type() {
    let Some(device) = create_test_gpu_device().await else {
        eprintln!("GPU not available, skipping test");
        return;
    };

    let shader = NeuralNetworkShader::new(&device).await.unwrap();
    let err = shader.activate(&[1.0_f32, 2.0, 3.0], 99).unwrap_err();
    assert!(format!("{err:?}").contains("Unknown activation type"));
}
