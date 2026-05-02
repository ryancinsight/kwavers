use super::{
    Architecture, DataType, EdgeRuntime, ExecutionKernel, IOSpecification, MemoryAllocator,
};

#[test]
fn test_edge_runtime_creation() {
    let runtime = EdgeRuntime::new(64);
    assert_eq!(runtime.allocator.total_memory, 64 * 1024 * 1024);
    assert!(runtime.model.is_none());
}

#[test]
fn test_memory_allocator() {
    let mut allocator = MemoryAllocator::new(1024 * 1024);

    let address = allocator.allocate_block(1024, 64).unwrap();
    assert_eq!(address, 0);

    assert_eq!(allocator.get_allocated_memory(), 1024);
}

#[test]
fn test_hardware_capabilities() {
    let caps = EdgeRuntime::detect_hardware_capabilities();

    match caps.architecture {
        Architecture::ARM64 | Architecture::ARM => {
            assert!(caps.has_fpu, "ARM architectures should have FPU");
        }
        Architecture::X86_64 | Architecture::X86 => {
            assert!(caps.has_fpu, "x86 architectures should have FPU");
        }
        Architecture::RISCV | Architecture::Other(_) => {}
    }

    assert!(
        caps.simd_width >= 64 || matches!(caps.architecture, Architecture::Other(_)),
        "Should have at least basic SIMD or be other architecture"
    );
    assert!(caps.total_memory_mb > 0, "Should detect non-zero memory");
}

#[test]
fn test_data_type_quantization() {
    let runtime = EdgeRuntime::new(64);
    let input = vec![1.0, -1.0, 0.5, -0.5];
    let kernel = ExecutionKernel {
        id: "test".to_string(),
        io_spec: IOSpecification {
            input_shape: vec![4],
            output_shape: vec![4],
            input_dtype: DataType::Int8,
            output_dtype: DataType::Float32,
        },
        estimated_time_us: 100.0,
        memory_required: 1024,
    };

    let result = runtime.software_quantize(&input, &kernel);
    assert!(result.is_ok());

    let quantized = result.unwrap();
    assert_eq!(quantized.len(), 4);

    for &val in &quantized {
        assert!((-127.0f32..=127.0f32).contains(&val));
    }
}
