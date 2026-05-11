//! External GPU-backend trait surface for FDTD acoustic propagation.
//!
//! Implementors are responsible for one fused velocity+pressure update
//! per call and return a fresh pressure volume as `Array3<f64>`.

use ndarray::Array3;

use crate::core::error::KwaversResult;

pub trait FdtdGpuAccelerator: Send + Sync + std::fmt::Debug {
    /// Propagate one acoustic time step on the GPU device.
    ///
    /// # Errors
    /// - Returns [`Err`] if the GPU kernel launch fails or a device error occurs.
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
    ) -> KwaversResult<Array3<f64>>;
}
