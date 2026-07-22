//! GPU buffer registry: name-keyed `GpuBufferManager`.
//!
//! This module provides [`GpuBufferManager`], a **named registry** of [`GpuBufferData`]
//! instances allocated and looked up by string key. The buffer primitive itself
//! lives in [`crate::gpu::buffer`]; this module only manages the registry layer.
//!
//! ## Relationship to `solver::backend::gpu::buffers::GpuBufferManager`
//!
//! The codebase intentionally contains **two `GpuBufferManager` types** with
//! distinct responsibilities — this is **not** a DRY violation:
//!
//! | Type                                            | Layer       | Key                            | Purpose                                                                  |
//! |-------------------------------------------------|-------------|--------------------------------|--------------------------------------------------------------------------|
//! | `crate::gpu::buffers::GpuBufferManager` (here)     | gpu module  | `String` (stable name)         | Named registry: persistent, per-context buffers (`CoreGpuContext`, `MultiGpuContext`) |
//! | `solver::backend::gpu::buffers::GpuBufferManager`  | solver layer| `(size, usage)` (allocation key) | Allocation pool: ephemeral compute scratch buffers reused across kernel dispatches |
//!
//! The registry tracks named state (one `pressure_field` buffer per context).
//! The pool recycles size-matched buffers between dispatches. Merging would
//! conflate persistence semantics with reuse semantics. The split follows SRP:
//! each `GpuBufferManager` changes for one reason only.
//!
//! SRP: changes here when the registry naming or memory-budget strategy
//! changes. Changes to individual buffer lifecycle or readback belong in
//! `crate::gpu::buffer`. Changes to dispatch-pool reuse semantics belong in
//! `solver::backend::gpu::buffers`.

use crate::gpu::buffer::GpuBufferData;
use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;

/// Named GPU buffer pool with aggregate memory tracking.
///
/// Allocates [`GpuBufferData`] instances keyed by string name and tracks total
/// device-side memory. All allocations use the canonical [`GpuBufferData`] type
/// from [`crate::gpu::buffer`].
///
/// # Invariants
///
/// - `total_memory` equals the sum of `buf.size()` for every live buffer.
/// - No two entries share the same name; `allocate` returns `Err` on collision.
#[derive(Debug)]
pub struct GpuBufferManager {
    buffers: HashMap<String, GpuBufferData>,
    total_memory: u64,
    _max_memory: u64,
}

impl GpuBufferManager {
    /// Create a new buffer manager.
    ///
    /// `max_memory` is initialised from the device's `max_buffer_size` limit
    /// and recorded for future eviction / budget enforcement.
    pub fn new(device: &wgpu::Device) -> Self {
        let limits = device.limits();
        Self {
            buffers: HashMap::new(),
            total_memory: 0,
            _max_memory: limits.max_buffer_size,
        }
    }

    /// Allocate a new buffer with the given `name`, `size`, and `usage`.
    ///
    /// Returns `Err` if a buffer named `name` already exists.
    /// The new buffer is accessible via [`GpuBufferManager::get`].
    /// # Errors
    /// - Returns `KwaversError::System` if the precondition for a System-class constraint is violated.
    ///
    pub fn allocate(
        &mut self,
        device: &wgpu::Device,
        name: &str,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> KwaversResult<&GpuBufferData> {
        if self.buffers.contains_key(name) {
            return Err(KwaversError::System(
                kwavers_core::error::SystemError::InvalidOperation {
                    operation: format!("Buffer '{}' creation", name),
                    reason: "Buffer already exists".to_string(),
                },
            ));
        }

        // `GpuBufferData::new` is infallible; wgpu panics on OOM rather than returning Err.
        let buffer = GpuBufferData::new(device, name, size as usize, usage);
        self.total_memory += size;
        self.buffers.insert(name.to_string(), buffer);

        self.buffers.get(name).ok_or_else(|| {
            KwaversError::System(kwavers_core::error::SystemError::ResourceExhausted {
                resource: format!("GPU buffer '{}'", name),
                reason: "Buffer not found after creation".to_string(),
            })
        })
    }

    /// Get a buffer by name.
    pub fn get(&self, name: &str) -> Option<&GpuBufferData> {
        self.buffers.get(name)
    }

    /// Get a mutable buffer by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut GpuBufferData> {
        self.buffers.get_mut(name)
    }

    /// Release (deallocate) a buffer by name, updating the memory counter.
    pub fn release(&mut self, name: &str) {
        if let Some(buffer) = self.buffers.remove(name) {
            self.total_memory = self.total_memory.saturating_sub(buffer.size() as u64);
        }
    }

    /// Total bytes currently allocated across all live buffers.
    pub fn total_memory(&self) -> u64 {
        self.total_memory
    }
}