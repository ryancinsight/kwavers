use super::*;

#[test]
fn test_gpu_backend_creation() {
    // May fail if GPU not available, which is OK
    match GPUBackend::new() {
        Ok(backend) => {
            assert_eq!(backend.backend_type(), BackendType::GPU);
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
        assert!(caps.supports_fft);
        assert!(caps.supports_f32);
        assert!(caps.supports_async);
        assert!(caps.max_parallelism > 0);
    }
}

#[test]
fn test_gpu_synchronize() {
    if let Ok(backend) = GPUBackend::new() {
        let result = backend.synchronize();
        assert!(result.is_ok());
    }
}

#[test]
fn test_gpu_devices() {
    if let Ok(backend) = GPUBackend::new() {
        let devices = backend.devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, 0);
        assert_eq!(devices[0].backend_type, BackendType::GPU);
    }
}

#[test]
fn test_performance_estimation() {
    if let Ok(backend) = GPUBackend::new() {
        // Small problem
        let perf_small = backend.estimate_performance((64, 64, 64));
        // Large problem
        let perf_large = backend.estimate_performance((256, 256, 256));

        // Large problem should have higher estimated performance
        assert!(perf_large > perf_small);
    }
}
