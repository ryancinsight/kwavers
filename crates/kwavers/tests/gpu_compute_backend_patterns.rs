//! Provider-backed GPU compute backend contract tests.
//!
//! This target intentionally exercises the public Kwavers/Hephaestus provider
//! seam instead of direct WGPU adapter/device construction. WGPU is the current
//! real kernel provider; CUDA belongs behind the same operation traits once
//! real CUDA kernels exist.

#![cfg(feature = "gpu")]

use kwavers_gpu::backend::GPUBackend;
use kwavers_solver::backend::traits::{BackendType, ComputeBackend, GpuProvider};
use leto::Array3 as LetoArray3;

#[test]
fn gpu_backend_reports_provider_identity_when_available() {
    let Ok(backend) = GPUBackend::new() else {
        return;
    };

    assert_eq!(backend.backend_type(), BackendType::GPU(GpuProvider::Wgpu));
    assert_eq!(backend.gpu_provider(), Some(GpuProvider::Wgpu));
}

#[test]
fn gpu_backend_rejects_unknown_device_id_when_available() {
    let Ok(mut backend) = GPUBackend::new() else {
        return;
    };

    let err = backend
        .select_device(1)
        .expect_err("provider-backed GPU backend exposes only selected device 0");

    assert!(
        err.to_string().contains("device_id"),
        "unexpected device-selection error: {err}"
    );
}

#[test]
fn gpu_backend_dispatches_provider_native_elementwise_when_available() {
    let Ok(backend) = GPUBackend::new() else {
        return;
    };

    let a = LetoArray3::from_shape_vec([2, 2, 1], vec![1.0, -2.0, 3.0, -4.0])
        .expect("shape matches input length");
    let b = LetoArray3::from_shape_vec([2, 2, 1], vec![5.0, 6.0, -7.0, -8.0])
        .expect("shape matches input length");
    let mut out = LetoArray3::<f32>::zeros([2, 2, 1]);

    ComputeBackend::element_wise_multiply(&backend, &a, &b, &mut out)
        .expect("provider-backed elementwise multiply must dispatch");

    assert_eq!(
        out.as_slice().expect("dense provider-native output"),
        &[5.0, -12.0, -21.0, 32.0]
    );
}
