//! External GPU-backend trait surface for FDTD acoustic propagation.
//!
//! Implementors are responsible for one fused velocity+pressure update
//! per call and return a fresh pressure volume as `leto::Array3<f64>`.

use leto::Array3 as LetoArray3;
use ndarray::Array3 as NdArray3;

use kwavers_core::error::KwaversResult;

pub trait FdtdGpuAccelerator: Send + Sync + std::fmt::Debug {
    /// Propagate one acoustic time step on the GPU device.
    ///
    /// # Errors
    /// - Returns [`Err`] if the GPU kernel launch fails or a device error occurs.
    #[allow(clippy::too_many_arguments)]
    fn propagate_acoustic_wave(
        &self,
        pressure: &LetoArray3<f64>,
        velocity_x: &LetoArray3<f64>,
        velocity_y: &LetoArray3<f64>,
        velocity_z: &LetoArray3<f64>,
        density: &NdArray3<f64>,
        sound_speed: &NdArray3<f64>,
        dt: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<LetoArray3<f64>>;
}
