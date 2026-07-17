use super::*;
use crate::backend::init::GpuProviderContext;
use crate::gpu::GpuDeviceProvider;
use kwavers_solver::backend::traits::{BackendType, ComputeBackend, GpuProvider};
use leto::Array3 as LetoArray3;

#[test]
fn gpu_backend_is_generic_over_provider_trait() {
    fn assert_backend<P>()
    where
        P: GpuComputeProvider,
        P::Device: GpuDeviceProvider,
    {
    }

    assert_backend::<WgpuComputeProvider>();
    assert!(std::mem::size_of::<GPUBackend<WgpuComputeProvider>>() > 0);
}

#[test]
fn gpu_compute_contract_is_composed_from_operation_traits() {
    fn assert_kernel<P>()
    where
        P: GpuKernelProvider,
        P::Device: GpuDeviceProvider,
    {
    }

    fn assert_elementwise<P>()
    where
        P: ElementWiseMultiplyProvider<Scalar = f32>,
    {
    }

    fn assert_spatial_derivative<P>()
    where
        P: SpatialDerivativeProvider<Scalar = f32>,
    {
    }

    assert_kernel::<WgpuComputeProvider>();
    assert_elementwise::<WgpuComputeProvider>();
    assert_spatial_derivative::<WgpuComputeProvider>();
}

#[test]
fn solver_compute_backend_uses_provider_native_scalar() {
    fn assert_scalar<B>()
    where
        B: ComputeBackend<Scalar = f32>,
    {
    }

    assert_scalar::<GPUBackend<WgpuComputeProvider>>();
}

#[test]
fn gpu_provider_identity_is_separate_from_kernel_dispatch() {
    fn assert_provider<P>()
    where
        P: GpuProviderBackend,
        P::Device: GpuDeviceProvider,
    {
    }

    assert_provider::<GpuProviderContext<hephaestus_wgpu::WgpuDevice>>();
    assert_provider::<WgpuComputeProvider>();
    assert_eq!(
        <hephaestus_wgpu::WgpuDevice as GpuDeviceProvider>::provider_kind(),
        GpuProvider::Wgpu
    );
}

#[cfg(feature = "cuda-provider")]
#[test]
fn cuda_satisfies_provider_identity_without_fake_kernels() {
    use hephaestus_cuda::CudaDevice;

    fn assert_provider<P>()
    where
        P: GpuProviderBackend,
        P::Device: GpuDeviceProvider,
    {
    }

    assert_provider::<GpuProviderContext<CudaDevice>>();
    assert_eq!(
        <CudaDevice as GpuDeviceProvider>::provider_kind(),
        GpuProvider::Cuda
    );
}

#[cfg(feature = "cuda-provider")]
#[test]
fn cuda_elementwise_provider_satisfies_only_real_operation_family() {
    fn assert_kernel<P>()
    where
        P: GpuKernelProvider<Scalar = f32>,
        P::Device: GpuDeviceProvider,
    {
    }

    fn assert_elementwise<P>()
    where
        P: ElementWiseMultiplyProvider<Scalar = f32>,
    {
    }

    assert_kernel::<CudaElementWiseProvider>();
    assert_elementwise::<CudaElementWiseProvider>();
    let _constructor: fn() -> kwavers_core::error::KwaversResult<CudaElementWiseProvider> =
        CudaElementWiseProvider::new;
}

#[cfg(feature = "cuda-provider")]
#[test]
fn cuda_elementwise_multiply_matches_provider_native_values_when_available() {
    let Ok(provider) = CudaElementWiseProvider::new() else {
        return;
    };

    let a = LetoArray3::from_shape_vec(
        [2, 2, 2],
        vec![1.5_f32, -2.0, 0.25, 4.0, 8.0, -0.5, 3.0, -1.0],
    )
    .expect("invariant: test shape matches value count");
    let b = LetoArray3::from_shape_vec(
        [2, 2, 2],
        vec![2.0_f32, 4.0, 8.0, -0.5, 0.25, -6.0, -3.0, -7.0],
    )
    .expect("invariant: test shape matches value count");
    let mut out = LetoArray3::<f32>::zeros([2, 2, 2]);

    provider
        .element_wise_multiply(&a, &b, &mut out)
        .expect("invariant: CUDA elementwise dispatch succeeds when CUDA is available");

    assert_eq!(
        out.as_slice()
            .expect("invariant: provider-native Leto output is dense"),
        &[3.0, -8.0, 2.0, -2.0, 2.0, 3.0, -9.0, 7.0]
    );
}

#[test]
fn test_gpu_backend_creation() {
    // May fail if GPU not available, which is OK
    match GPUBackend::new() {
        Ok(backend) => {
            assert_eq!(backend.backend_type(), BackendType::GPU(GpuProvider::Wgpu));
            assert_eq!(backend.gpu_provider(), Some(GpuProvider::Wgpu));
            assert!(backend.is_available());
            println!("GPU backend initialized: {}", backend.device_name());
        }
        Err(e) => {
            println!("GPU backend unavailable (expected on some systems): {}", e);
        }
    }
}

#[test]
fn test_gpu_capabilities() {
    if let Ok(backend) = GPUBackend::new() {
        let caps = backend.capabilities();
        assert!(!caps.supports_fft);
        assert!(caps.supports_f32);
        assert!(!caps.supports_f64);
        assert!(caps.supports_async);
        assert!(caps.max_parallelism > 0);
    }
}

#[test]
fn test_gpu_synchronize() {
    if let Ok(backend) = GPUBackend::new() {
        backend.synchronize().unwrap();
    }
}

#[test]
fn test_gpu_devices() {
    if let Ok(backend) = GPUBackend::new() {
        let devices = backend.devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, 0);
        assert_eq!(devices[0].backend_type, BackendType::GPU(GpuProvider::Wgpu));
    }
}

#[test]
fn test_performance_estimation() {
    if let Ok(backend) = GPUBackend::new() {
        let perf_small = backend.estimate_performance((64, 64, 64));
        let perf_large = backend.estimate_performance((256, 256, 256));
        let device_peak = backend.devices()[0].peak_performance;

        assert_eq!(perf_small, device_peak);
        assert_eq!(perf_large, device_peak);
    }
}

#[test]
fn spatial_derivative_shader_has_real_finite_difference_body() {
    let shader = include_str!("shaders/operators.wgsl");

    assert!(shader.contains("derivative_params.direction == 0u"));
    assert!(shader.contains("input[idx + yz] - input[idx]"));
    assert!(!shader.contains("placeholder"));
    assert!(!shader.contains("copy input to output"));
}

#[test]
fn elementwise_multiply_matches_provider_native_values_when_gpu_available() {
    let Ok(backend) = GPUBackend::new() else {
        return;
    };

    let a = LetoArray3::from_shape_vec(
        [2, 2, 2],
        vec![1.5_f32, -2.0, 0.25, 4.0, 8.0, -0.5, 3.0, -1.0],
    )
    .expect("invariant: test shape matches value count");
    let b = LetoArray3::from_shape_vec(
        [2, 2, 2],
        vec![2.0_f32, 4.0, 8.0, -0.5, 0.25, -6.0, -3.0, -7.0],
    )
    .expect("invariant: test shape matches value count");
    let mut out = LetoArray3::<f32>::zeros([2, 2, 2]);

    backend
        .dispatch_element_wise_multiply(&a, &b, &mut out)
        .expect("invariant: WGPU elementwise dispatch succeeds for exact provider-native values");

    assert_eq!(
        out.as_slice()
            .expect("invariant: provider-native Leto output is dense"),
        &[3.0, -8.0, 2.0, -2.0, 2.0, 3.0, -9.0, 7.0]
    );
}

#[test]
fn elementwise_multiply_rejects_shape_mismatch_when_gpu_available() {
    let Ok(backend) = GPUBackend::new() else {
        return;
    };

    let a = LetoArray3::<f32>::from_elem([2, 2, 2], 1.0);
    let b = LetoArray3::<f32>::from_elem([2, 2, 1], 1.0);
    let mut out = LetoArray3::<f32>::zeros([2, 2, 2]);

    let err = backend
        .dispatch_element_wise_multiply(&a, &b, &mut out)
        .expect_err("invariant: shape mismatch is rejected before dispatch");

    assert!(err.to_string().contains("rhs shape"));
}

#[test]
fn spatial_derivative_matches_affine_field_when_gpu_available() {
    let Ok(backend) = GPUBackend::new() else {
        return;
    };

    let field = LetoArray3::from_shape_fn([4, 3, 5], |[i, j, k]| (6 * i + 2 * j + k) as f32);

    for (direction, expected) in [(0, 6.0_f32), (1, 2.0_f32), (2, 1.0_f32)] {
        let mut out = LetoArray3::<f32>::zeros(field.shape());
        backend
            .dispatch_spatial_derivative(&field, direction, &mut out)
            .expect("invariant: WGPU derivative dispatch succeeds for affine field");

        for value in out.iter() {
            assert_eq!(*value, expected);
        }
    }
}

#[test]
fn solver_compute_backend_dispatches_provider_native_values_when_gpu_available() {
    let Ok(backend) = GPUBackend::new() else {
        return;
    };

    let a = LetoArray3::<f32>::from_elem([1, 1, 1], 1.5);
    let b = LetoArray3::<f32>::from_elem([1, 1, 1], 2.0);
    let mut out = LetoArray3::<f32>::zeros([1, 1, 1]);

    ComputeBackend::element_wise_multiply(&backend, &a, &b, &mut out)
        .expect("invariant: solver ComputeBackend dispatch uses the provider-native scalar");

    assert_eq!(
        out.as_slice()
            .expect("invariant: provider-native Leto output is dense"),
        &[3.0]
    );
}
