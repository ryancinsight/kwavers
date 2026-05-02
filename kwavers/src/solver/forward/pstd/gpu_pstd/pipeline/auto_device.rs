//! `GpuPstdSolver::with_auto_device` — automatic GPU adapter selection.
//!
//! SRP: changes when adapter selection policy or device descriptor changes.

use super::super::GpuPstdSolver;
use crate::domain::grid::Grid;
use std::sync::Arc;

impl GpuPstdSolver {
    /// Create a `GpuPstdSolver` by automatically selecting the best available
    /// GPU adapter.  Returns `Err` if no adapter is found or device creation fails.
    ///
    /// This constructor owns the wgpu device lifecycle, so callers do not need
    /// to add `wgpu` or `pollster` as direct dependencies.
    #[allow(clippy::too_many_arguments)]
    pub fn with_auto_device(
        grid: &Grid,
        c0_flat: &[f32],
        rho0_flat: &[f32],
        dt: f64,
        nt: usize,
        c_ref: f64,
        pml_x: &[f32],
        pml_y: &[f32],
        pml_z: &[f32],
        pml_sgx: &[f32],
        pml_sgy: &[f32],
        pml_sgz: &[f32],
        bon_a_flat: &[f32],
        absorb_nabla1: &[f32],
        absorb_nabla2: &[f32],
        absorb_tau: &[f32],
        absorb_eta: &[f32],
        nonlinear: bool,
        absorbing: bool,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|_| "No GPU adapter available".to_string())?;

        // Query adapter's native limits; use them as a ceiling so we don't
        // exceed what the hardware supports but do raise the WebGPU defaults
        // (e.g. max_storage_buffers_per_shader_stage is 8 by default but the
        // PSTD kspace bind group uses 10 storage buffers).
        let native_limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pstd_auto"),
            required_features: wgpu::Features::PUSH_CONSTANTS,
            required_limits: wgpu::Limits {
                max_push_constant_size: 128,
                max_storage_buffers_per_shader_stage:
                    native_limits.max_storage_buffers_per_shader_stage,
                ..wgpu::Limits::default()
            },
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("GPU device creation failed: {e}"))?;

        Self::new(
            Arc::new(device),
            Arc::new(queue),
            grid,
            c0_flat,
            rho0_flat,
            dt,
            nt,
            c_ref,
            pml_x,
            pml_y,
            pml_z,
            pml_sgx,
            pml_sgy,
            pml_sgz,
            bon_a_flat,
            absorb_nabla1,
            absorb_nabla2,
            absorb_tau,
            absorb_eta,
            nonlinear,
            absorbing,
        )
    }
}
