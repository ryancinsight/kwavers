//! WebGPU context management

use crate::error::{KwaversError, KwaversResult};
use crate::gpu::GpuFieldOps;
use crate::grid::Grid;
use ndarray::Array3;

#[cfg(feature = "wgpu")]
use wgpu::{ComputePipeline, Device, Queue};

/// WebGPU execution context
pub struct WebGpuContext {
    #[cfg(feature = "wgpu")]
    pub(crate) device: Device,
    #[cfg(feature = "wgpu")]
    pub(crate) queue: Queue,
    #[cfg(feature = "wgpu")]
    pub(crate) acoustic_pipeline: Option<ComputePipeline>,
    #[cfg(feature = "wgpu")]
    pub(crate) thermal_pipeline: Option<ComputePipeline>,
    #[cfg(not(feature = "wgpu"))]
    _phantom: std::marker::PhantomData<()>,
}

impl WebGpuContext {
    /// Create new WebGPU context
    pub async fn new() -> KwaversResult<Self> {
        #[cfg(feature = "wgpu")]
        {
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: wgpu::Backends::all(),
                ..Default::default()
            });

            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok_or_else(|| KwaversError::Gpu(crate::error::GpuError::NoDevicesFound))?;

            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: Some("kwavers-gpu"),
                        required_features: wgpu::Features::empty(),
                        required_limits: wgpu::Limits::default(),
                        memory_hints: wgpu::MemoryHints::default(),
                    },
                    None,
                )
                .await?;

            Ok(Self {
                device,
                queue,
                acoustic_pipeline: None,
                thermal_pipeline: None,
            })
        }

        #[cfg(not(feature = "wgpu"))]
        {
            Ok(Self {
                _phantom: std::marker::PhantomData,
            })
        }
    }

    /// Initialize compute pipelines
    pub fn initialize_pipelines(&mut self) -> KwaversResult<()> {
        #[cfg(feature = "wgpu")]
        {
            use super::shaders;

            // Create acoustic pipeline
            self.acoustic_pipeline = Some(shaders::create_acoustic_pipeline(&self.device)?);

            // Create thermal pipeline
            self.thermal_pipeline = Some(shaders::create_thermal_pipeline(&self.device)?);
        }

        Ok(())
    }
}

impl GpuFieldOps for WebGpuContext {
    fn update_acoustic_field(
        &mut self,
        pressure: &mut Array3<f64>,
        velocity_x: &mut Array3<f64>,
        velocity_y: &mut Array3<f64>,
        velocity_z: &mut Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        #[cfg(feature = "wgpu")]
        {
            use super::kernels;
            kernels::update_acoustic_field(
                self, pressure, velocity_x, velocity_y, velocity_z, grid, dt,
            )
        }

        #[cfg(not(feature = "wgpu"))]
        {
            Err(KwaversError::NotImplemented(
                "WebGPU acoustic field update".to_string(),
            ))
        }
    }

    fn update_thermal_field(
        &mut self,
        temperature: &mut Array3<f64>,
        heat_rate: &Array3<f64>,
        thermal_conductivity: &Array3<f64>,
        specific_heat: &Array3<f64>,
        density: &Array3<f64>,
        grid: &Grid,
        dt: f64,
    ) -> KwaversResult<()> {
        #[cfg(feature = "wgpu")]
        {
            use super::kernels;
            kernels::update_thermal_field(
                self,
                temperature,
                heat_rate,
                thermal_conductivity,
                specific_heat,
                density,
                grid,
                dt,
            )
        }

        #[cfg(not(feature = "wgpu"))]
        {
            Err(KwaversError::NotImplemented(
                "WebGPU thermal field update".to_string(),
            ))
        }
    }
}
