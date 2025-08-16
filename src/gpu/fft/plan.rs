//! FFT planning and workspace management
//!
//! Provides structures for FFT plan creation and workspace allocation

use crate::error::KwaversResult;
use crate::gpu::{GpuBuffer, memory::BufferType};
use std::time::Instant;

/// FFT direction enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FftDirection {
    Forward,
    Inverse,
}

/// FFT plan for efficient repeated transforms
#[derive(Debug)]
pub struct GpuFftPlan {
    /// FFT dimensions
    pub dimensions: (usize, usize, usize),
    /// Transform direction
    pub direction: FftDirection,
    /// Workspace for intermediate results
    pub workspace: FftWorkspace,
    /// Plan creation time for profiling
    pub created_at: Instant,
}

/// FFT workspace buffers
#[derive(Debug)]
pub struct FftWorkspace {
    /// Input buffer (real or complex)
    pub input: GpuBuffer,
    /// Output buffer (complex)
    pub output: GpuBuffer,
    /// Temporary buffer for intermediate results
    pub temp: Option<GpuBuffer>,
    /// Twiddle factors for FFT
    pub twiddle_factors: GpuBuffer,
}

impl FftWorkspace {
    /// Create a new FFT workspace
    pub fn new(nx: usize, ny: usize, nz: usize) -> KwaversResult<Self> {
        let size = nx * ny * nz;
        let complex_size = size * std::mem::size_of::<[f32; 2]>();
        
        // Create workspace buffers
        let input = GpuBuffer {
            id: 0,
            size_bytes: complex_size,
            device_ptr: None,
            host_ptr: None,
            is_pinned: false,
            allocation_time: Instant::now(),
            last_access_time: Instant::now(),
            access_count: 0,
            buffer_type: BufferType::FFT,
        };
        
        let output = GpuBuffer {
            id: 1,
            size_bytes: complex_size,
            device_ptr: None,
            host_ptr: None,
            is_pinned: false,
            allocation_time: Instant::now(),
            last_access_time: Instant::now(),
            access_count: 0,
            buffer_type: BufferType::FFT,
        };
        
        let twiddle_factors = GpuBuffer {
            id: 2,
            size_bytes: size * std::mem::size_of::<[f32; 2]>(),
            device_ptr: None,
            host_ptr: None,
            is_pinned: false,
            allocation_time: Instant::now(),
            last_access_time: Instant::now(),
            access_count: 0,
            buffer_type: BufferType::FFT,
        };
        
        Ok(Self {
            input,
            output,
            temp: None,
            twiddle_factors,
        })
    }
    
    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        self.input.size_bytes + 
        self.output.size_bytes + 
        self.twiddle_factors.size_bytes +
        self.temp.as_ref().map_or(0, |t| t.size_bytes)
    }
}

impl GpuFftPlan {
    /// Create a new FFT plan
    pub fn new(nx: usize, ny: usize, nz: usize, direction: FftDirection) -> KwaversResult<Self> {
        let workspace = FftWorkspace::new(nx, ny, nz)?;
        
        Ok(Self {
            dimensions: (nx, ny, nz),
            direction,
            workspace,
            created_at: Instant::now(),
        })
    }
    
    /// Get the total number of elements
    pub fn size(&self) -> usize {
        self.dimensions.0 * self.dimensions.1 * self.dimensions.2
    }
    
    /// Check if this is a forward transform
    pub fn is_forward(&self) -> bool {
        self.direction == FftDirection::Forward
    }
}