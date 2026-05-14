//! Burn-based GPU accelerator: struct definition, tensor conversion, and trait impl.

mod acoustic;
mod electromagnetic;
mod pde_residuals;

use super::super::types::{GpuConfig, PhysicsParameters};
use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::forward::fdtd::FdtdGpuAccelerator;
use burn::prelude::*;
use burn::tensor::backend::Backend;
use ndarray::Array3;
use std::marker::PhantomData;

/// Burn-based GPU accelerator for scientific computing.
#[derive(Debug)]
pub struct BurnGpuAccelerator<B: Backend> {
    /// Burn device for tensor operations
    pub(super) device: B::Device,
    /// Backend type marker
    pub(super) _backend: PhantomData<B>,
}

impl<B: Backend> BurnGpuAccelerator<B> {
    /// Create new GPU accelerator.
    /// # Errors
    /// - Returns [`KwaversError::System`] if `config.enable_gpu` is false.
    ///
    pub fn new(config: &GpuConfig) -> KwaversResult<Self>
    where
        B::Device: Default,
    {
        if !config.enable_gpu {
            return Err(KwaversError::System(
                crate::core::error::SystemError::ResourceUnavailable {
                    resource: "GPU acceleration disabled".to_owned(),
                },
            ));
        }
        Ok(Self {
            device: B::Device::default(),
            _backend: PhantomData,
        })
    }

    /// Get the Burn device.
    pub fn device(&self) -> &B::Device {
        &self.device
    }

    /// Convert ndarray `Array3<f64>` to a Burn 3-D tensor (via `f32`).
    pub fn array_to_tensor(&self, array: &Array3<f64>) -> Tensor<B, 3> {
        let shape = array.shape();
        let data: Vec<f32> = array.iter().map(|&x| x as f32).collect();
        Tensor::<B, 1>::from_data(TensorData::from(data.as_slice()), &self.device)
            .reshape([shape[0], shape[1], shape[2]])
    }

    /// Convert Burn 3-D tensor to ndarray `Array3<f64>` (via `f32`).
    /// # Panics
    /// - Panics if internal tensor shape invariants are violated.
    ///
    pub fn tensor_to_array(&self, tensor: Tensor<B, 3>) -> Array3<f64> {
        let shape = tensor.shape();
        let data = tensor.into_data();
        let slice = data.as_slice::<f32>().unwrap();
        Array3::from_shape_vec(
            (shape[0], shape[1], shape[2]),
            slice.iter().map(|&x| x as f64).collect(),
        )
        .unwrap()
    }

    // ── Shared gradient helpers ───────────────────────────────────────────────

    pub(super) fn compute_gradient_x(&self, field: &Tensor<B, 3>, dx: f32) -> Tensor<B, 3> {
        let shifted_right = field.clone().roll(&[1], &[0]);
        let shifted_left = field.clone().roll(&[-1], &[0]);
        (shifted_right - shifted_left).div_scalar(2.0 * dx)
    }

    pub(super) fn compute_gradient_y(&self, field: &Tensor<B, 3>, dy: f32) -> Tensor<B, 3> {
        let shifted_up = field.clone().roll(&[1], &[1]);
        let shifted_down = field.clone().roll(&[-1], &[1]);
        (shifted_up - shifted_down).div_scalar(2.0 * dy)
    }

    pub(super) fn compute_gradient_z(&self, field: &Tensor<B, 3>, dz: f32) -> Tensor<B, 3> {
        let shifted_front = field.clone().roll(&[1], &[2]);
        let shifted_back = field.clone().roll(&[-1], &[2]);
        (shifted_front - shifted_back).div_scalar(2.0 * dz)
    }
}

impl<B: Backend> FdtdGpuAccelerator for BurnGpuAccelerator<B> {
    fn propagate_acoustic_wave(
        &self,
        pressure: &Array3<f64>,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        dt: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Array3<f64>> {
        BurnGpuAccelerator::propagate_acoustic_wave(
            self,
            pressure,
            velocity_x,
            velocity_y,
            velocity_z,
            density,
            sound_speed,
            dt,
            dx,
            dy,
            dz,
        )
    }
}
