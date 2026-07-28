//! SIMD-accelerated FDTD pressure / velocity update kernels.
//!
//! `FdtdSimdOps` is a thin dispatcher that forwards to hermes-backed
//! kernels in the `pressure` and `velocity` sub-modules.  Runtime ISA
//! selection (AVX-512 / AVX2 / NEON / scalar) is performed inside
//! `hermes_simd_intrinsics` — no manual dispatch is needed here.

mod pressure;
mod velocity;

/// SIMD-accelerated FDTD operations backed by `hermes-simd`.
#[derive(Debug, Default)]
pub struct FdtdSimdOps;

impl FdtdSimdOps {
    /// Create a new `FdtdSimdOps` dispatcher.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Hermes-dispatched 3-D pressure update.
    ///
    /// Computes `p^{n+1} = 2·p^n − p^{n−1} + c²Δt²·∇²p^n` for interior
    /// points `(i, j, k)` with `1 ≤ i < nx−1`, `1 ≤ j < ny−1`, `1 ≤ k < nz−1`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_pressure_3d(
        &self,
        pressure: &mut [f32],
        pressure_prev: &[f32],
        laplacian: &[f32],
        c_squared_dt_squared: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        self.update_pressure_hermes(
            pressure,
            pressure_prev,
            laplacian,
            c_squared_dt_squared,
            nx,
            ny,
            nz,
        );
    }

    /// Hermes-dispatched 3-D velocity update.
    ///
    /// Computes `v^{n+1} = v^n − (Δt/ρ)·∇p^n` for interior points.
    #[allow(clippy::too_many_arguments)]
    pub fn update_velocity_3d(
        &self,
        velocity: &mut [f32],
        velocity_prev: &[f32],
        pressure_gradient: &[f32],
        dt_over_rho: f32,
        nx: usize,
        ny: usize,
        nz: usize,
    ) {
        self.update_velocity_hermes(
            velocity,
            velocity_prev,
            pressure_gradient,
            dt_over_rho,
            nx,
            ny,
            nz,
        );
    }
}
