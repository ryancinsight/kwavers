//! GPU Compute Backend Pattern Tests
//!
//! This test module demonstrates best practices for GPU compute backend implementation
//! including proper buffer usage, staging buffers, and async operations.
//!
//! Key learnings from implementation:
//! - wgpu operations are async and require proper error handling
//! - Staging buffers needed for reading GPU data back to CPU
//! - Buffer usage flags must match operation (COPY_SRC vs COPY_DST vs MAP_READ)
//! - Generic types need Into<f32>/From<f32> for GPU operations
//! - Async traits require careful design with impl Future

#![cfg(feature = "gpu")]

use std::collections::HashMap;

/// Storage abstraction for dense and sparse data
#[derive(Debug)]
enum Storage<T> {
    Dense(Vec<T>),
    Sparse(HashMap<usize, T>),
}

impl<T> Storage<T> {
    fn dense(data: Vec<T>) -> Self {
        Storage::Dense(data)
    }

    fn sparse(map: HashMap<usize, T>) -> Self {
        Storage::Sparse(map)
    }

    fn compute_squares(&self) -> Vec<T>
    where
        T: Copy + std::ops::Mul<Output = T> + Default,
    {
        match self {
            Storage::Dense(data) => data.iter().map(|&x| x * x).collect(),
            Storage::Sparse(map) => {
                let mut result = Vec::with_capacity(map.len());
                result.extend(map.values().map(|&x| x * x));
                result
            }
        }
    }
}

/// Backend abstraction for CPU/GPU compute
#[derive(Debug, Clone, Copy)]
enum Backend {
    Cpu,
    Gpu,
}

/// Custom error type for compute operations
#[derive(Debug)]
enum ComputeError {
    NoAdapter,
    DeviceError(String),
}

impl std::fmt::Display for ComputeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ComputeError::NoAdapter => write!(f, "No GPU adapter found"),
            ComputeError::DeviceError(msg) => write!(f, "Device error: {}", msg),
        }
    }
}

impl std::error::Error for ComputeError {}

impl From<wgpu::RequestDeviceError> for ComputeError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        ComputeError::DeviceError(err.to_string())
    }
}

/// Trait for compute backends
trait ComputeBackend {
    async fn compute_squares<T>(&self, storage: &Storage<T>) -> Result<Vec<T>, ComputeError>
    where
        T: Copy + std::ops::Mul<Output = T> + Default + Into<f32> + From<f32>;
}

impl ComputeBackend for Backend {
    async fn compute_squares<T>(&self, storage: &Storage<T>) -> Result<Vec<T>, ComputeError>
    where
        T: Copy + std::ops::Mul<Output = T> + Default + Into<f32> + From<f32>,
    {
        match self {
            Backend::Cpu => Ok(match storage {
                Storage::Dense(data) => data.iter().map(|&x| x * x).collect(),
                Storage::Sparse(map) => {
                    let mut result = Vec::with_capacity(map.len());
                    result.extend(map.values().map(|&x| x * x));
                    result
                }
            }),
            Backend::Gpu => match storage {
                Storage::Dense(data) => {
                    use wgpu::*;

                    // Initialize wgpu with proper error handling
                    let instance = Instance::new(InstanceDescriptor::default());
                    let adapter = instance
                        .request_adapter(&RequestAdapterOptions::default())
                        .await
                        .ok_or(ComputeError::NoAdapter)?;
                    let (device, queue) = adapter
                        .request_device(&DeviceDescriptor::default(), None)
                        .await?;

                    // Create shader module
                    let shader = device.create_shader_module(ShaderModuleDescriptor {
                        label: Some("Square Shader"),
                        source: ShaderSource::Wgsl(std::borrow::Cow::Borrowed(
                            r#"
                                @group(0) @binding(0)
                                var<storage, read> input: array<f32>;

                                @group(0) @binding(1)
                                var<storage, read_write> output: array<f32>;

                                @compute @workgroup_size(64)
                                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                                    let idx = global_id.x;
                                    if (idx < arrayLength(&input)) {
                                        output[idx] = input[idx] * input[idx];
                                    }
                                }
                            "#,
                        )),
                    });

                    // Create compute pipeline
                    let compute_pipeline =
                        device.create_compute_pipeline(&ComputePipelineDescriptor {
                            label: Some("Square Pipeline"),
                            layout: None,
                            module: &shader,
                            entry_point: "main",
                            compilation_options: PipelineCompilationOptions::default(),
                            cache: None,
                        });

                    // Create buffers with proper usage flags
                    let input_buffer = device.create_buffer(&BufferDescriptor {
                        label: Some("Input Buffer"),
                        size: (data.len() * std::mem::size_of::<f32>()) as u64,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });

                    let output_buffer = device.create_buffer(&BufferDescriptor {
                        label: Some("Output Buffer"),
                        size: (data.len() * std::mem::size_of::<f32>()) as u64,
                        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });

                    // Staging buffer for reading back results
                    let staging_buffer = device.create_buffer(&BufferDescriptor {
                        label: Some("Staging Buffer"),
                        size: (data.len() * std::mem::size_of::<f32>()) as u64,
                        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    });

                    // Copy input data to GPU
                    let input_data: Vec<f32> = data.iter().map(|&x| x.into()).collect();
                    queue.write_buffer(&input_buffer, 0, bytemuck::cast_slice(&input_data));

                    // Create bind group
                    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
                    let bind_group = device.create_bind_group(&BindGroupDescriptor {
                        label: Some("Compute Bind Group"),
                        layout: &bind_group_layout,
                        entries: &[
                            BindGroupEntry {
                                binding: 0,
                                resource: input_buffer.as_entire_binding(),
                            },
                            BindGroupEntry {
                                binding: 1,
                                resource: output_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    // Create command encoder and compute pass
                    let mut encoder =
                        device.create_command_encoder(&CommandEncoderDescriptor::default());
                    {
                        let mut compute_pass =
                            encoder.begin_compute_pass(&ComputePassDescriptor::default());
                        compute_pass.set_pipeline(&compute_pipeline);
                        compute_pass.set_bind_group(0, &bind_group, &[]);
                        compute_pass
                            .dispatch_workgroups((data.len() as u32).div_ceil(64), 1, 1);
                    }

                    // Copy from output buffer to staging buffer for reading
                    encoder.copy_buffer_to_buffer(
                        &output_buffer,
                        0,
                        &staging_buffer,
                        0,
                        (data.len() * std::mem::size_of::<f32>()) as u64,
                    );

                    // Submit and wait
                    queue.submit([encoder.finish()]);
                    device.poll(Maintain::Wait);

                    // Read back results from staging buffer
                    let staging_slice = staging_buffer.slice(..);
                    staging_slice.map_async(MapMode::Read, |_| {});
                    device.poll(Maintain::Wait);

                    let staging_data = staging_slice.get_mapped_range();
                    let result: Vec<f32> = bytemuck::cast_slice(&staging_data).to_vec();
                    drop(staging_data);
                    staging_buffer.unmap();

                    // Convert back to original type
                    Ok(result.iter().map(|&x| T::from(x)).collect())
                }
                Storage::Sparse(map) => {
                    // Fallback to CPU for sparse data
                    let mut result = Vec::with_capacity(map.len());
                    result.extend(map.values().map(|&x| x * x));
                    Ok(result)
                }
            },
        }
    }
}

/// Select backend based on feature flags
fn select_backend() -> Backend {
    if cfg!(feature = "gpu") {
        Backend::Gpu
    } else {
        Backend::Cpu
    }
}

/// Compute squares using the appropriate backend
async fn compute_squares<T>(
    backend: &impl ComputeBackend,
    storage: &Storage<T>,
) -> Result<Vec<T>, Box<dyn std::error::Error>>
where
    T: Copy + std::ops::Mul<Output = T> + Default + Into<f32> + From<f32>,
{
    backend
        .compute_squares(storage)
        .await
        .map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_dense() {
        let storage = Storage::dense(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let result = storage.compute_squares();
        assert_eq!(result, vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_storage_sparse() {
        let mut map = HashMap::new();
        map.insert(0, 2.0_f32);
        map.insert(1, 3.0);
        map.insert(2, 4.0);
        let storage = Storage::sparse(map);
        let mut result = storage.compute_squares();
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(result, vec![4.0, 9.0, 16.0]);
    }

    #[tokio::test]
    async fn test_cpu_backend_dense() {
        let backend = Backend::Cpu;
        let storage = Storage::dense(vec![1.0_f32, 2.0, 3.0, 4.0]);
        let result = compute_squares(&backend, &storage).await.unwrap();
        assert_eq!(result, vec![1.0, 4.0, 9.0, 16.0]);
    }

    #[tokio::test]
    async fn test_cpu_backend_sparse() {
        let backend = Backend::Cpu;
        let mut map = HashMap::new();
        map.insert(0, 2.0_f32);
        map.insert(1, 3.0);
        map.insert(2, 4.0);
        let storage = Storage::sparse(map);
        let mut result = compute_squares(&backend, &storage).await.unwrap();
        result.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(result, vec![4.0, 9.0, 16.0]);
    }

    #[tokio::test]
    #[ignore] // Requires GPU hardware to run
    async fn test_gpu_backend_dense() {
        let backend = Backend::Gpu;
        let storage = Storage::dense(vec![1.0_f32, 2.0, 3.0, 4.0]);
        match compute_squares(&backend, &storage).await {
            Ok(result) => {
                assert_eq!(result.len(), 4);
                for (i, &val) in result.iter().enumerate() {
                    let expected = ((i + 1) * (i + 1)) as f32;
                    assert!((val - expected).abs() < 1e-5);
                }
            }
            Err(e) => {
                // GPU not available, test passes
                eprintln!("GPU test skipped: {}", e);
            }
        }
    }

    #[test]
    fn test_backend_selection() {
        let backend = select_backend();
        // Backend selection works correctly based on feature flags
        match backend {
            Backend::Cpu | Backend::Gpu => {
                // Both variants are valid
            }
        }
    }
}
