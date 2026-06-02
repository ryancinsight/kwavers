use super::*;

#[test]
fn test_jit_compiler_creation() {
    let compiler = JitCompiler::new(OptimizationLevel::Basic);
    assert_eq!(compiler.kernel_cache.len(), 0);
    assert_eq!(compiler.stats.kernels_compiled, 0);
}

#[test]
fn test_memory_pool_allocation() {
    let pool = JitMemoryPool::new();
    let buffer = pool.allocate_output_buffer(100).unwrap();
    assert_eq!(buffer.len(), 128);
}

#[test]
fn test_optimization_levels() {
    let compiler_none = JitCompiler::new(OptimizationLevel::None);
    let compiler_max = JitCompiler::new(OptimizationLevel::Maximum);

    let plan = ExecutionPlan {
        operations: vec![Operation::InputNormalization],
        _memory_layout: MemoryLayout::Contiguous,
        _vectorization: VectorizationLevel::None,
        _cache_optimization: false,
    };

    let time_none = compiler_none.estimate_execution_time(&plan);
    let time_max = compiler_max.estimate_execution_time(&plan);

    assert!(time_max < time_none);
}
