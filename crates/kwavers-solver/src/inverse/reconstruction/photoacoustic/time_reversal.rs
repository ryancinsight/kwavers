//! Time reversal reconstruction for photoacoustic imaging
//!
//! Implements the k-space pseudospectral time-reversal method (TR) for
//! photoacoustic image reconstruction.
//!
//! ## Mathematical Foundation
//!
//! ### Wave equation time-reversal symmetry
//!
//! The lossless scalar wave equation `∂²p/∂t² = c²∇²p` is invariant under
//! `t → −t` (Fink 1992). Consequently, the initial pressure distribution
//! `p₀(r)` can be recovered by:
//!
//! 1. Recording sensor data `p(r_s, t)` for `t ∈ [0, T]`.
//! 2. Running the wave equation **forward** from `t = 0` to `t = T` with the
//!    **time-reversed** sensor signal `p(r_s, T − t)` applied as a **hard**
//!    (Dirichlet) source at each sensor location at every step.
//! 3. The field at `t = T` equals `p₀(r)` (Treeby et al. 2010, §2.3).
//!
//! ### k-space propagator
//!
//! Exact (dispersion-free) propagation via the leapfrog update in k-space:
//! ```text
//! p̂(k, t+dt) = 2·cos(c·|k|·dt)·p̂(k, t) − p̂(k, t−dt)
//! ```
//! where `cos(c·|k|·dt)` is **real**. After multiplying by this real factor
//! and inverse-FFTing, the result is a real pressure field.
//! (Tabei et al. 2002, Eq. 2; Treeby & Cox 2010, Eq. 15.)
//!
//! ### Source injection — hard source (Dirichlet)
//!
//! At each TR time step `t`, the time-reversed sensor value
//! `p(r_s, T − t)` **replaces** (not adds to) the field at the nearest
//! grid node of each sensor location. This imposes a Dirichlet condition
//! that drives the reconstruction.
//! (Treeby et al. 2010, §2.3; k-Wave MATLAB source `kspaceFirstOrder3D.m`,
//! `source.p_mode = 'dirichlet'`.)
//!
//! ## References
//!
//! - Fink M (1992). "Time reversal of ultrasonic fields — Part I: Basic
//!   principles." IEEE Trans. UFFC 39(5):555–566. DOI:10.1109/58.156174
//! - Tabei M, Mast TD, Waag RC (2002). "A k-space method for coupled
//!   first-order acoustic propagation equations."
//!   J. Acoust. Soc. Am. 111(1):53–63. DOI:10.1121/1.1421344
//! - Treeby BE, Zhang EZ, Cox BT (2010). "Photoacoustic tomography in
//!   absorbing acoustic media using combined photoacoustic and ultrasound
//!   imaging." New J. Phys. 12, 055008. DOI:10.1088/1367-2630/12/5/055008

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_math::fft::{get_fft_for_grid, Fft3dInOutExt};
use moirai_parallel::{enumerate_mut_with, Adaptive};
use ndarray::{Array3, ArrayView2};
use num_complex::Complex64;

/// k-space pseudospectral time-reversal reconstructor.
///
/// Propagates a real pressure field via the exact cosine leapfrog update
/// in the Fourier domain, injecting time-reversed sensor signals as hard
/// Dirichlet sources.
#[derive(Debug)]
pub struct PhotoacousticTimeReversal {
    grid_size: [usize; 3],
    sound_speed: f64,
    sampling_frequency: f64,
}

impl PhotoacousticTimeReversal {
    /// Create a new time-reversal reconstructor.
    ///
    /// # Arguments
    /// * `grid_size`          – `[nx, ny, nz]` grid dimensions
    /// * `sound_speed`        – homogeneous sound speed `c₀` (m/s)
    /// * `sampling_frequency` – temporal sampling rate (Hz)
    /// * `_time_steps`        – (unused; kept for API compatibility)
    pub fn new(
        grid_size: [usize; 3],
        sound_speed: f64,
        sampling_frequency: f64,
        _time_steps: usize,
    ) -> Self {
        Self {
            grid_size,
            sound_speed,
            sampling_frequency,
        }
    }

    /// Reconstruct the initial pressure distribution via time reversal.
    ///
    /// Runs the forward-in-pseudotime loop with time-reversed sensor data
    /// injected as hard (Dirichlet) sources. Returns the reconstructed
    /// pressure field `p₀(r)`.
    ///
    /// # Errors
    /// Returns `Err` if the CFL condition is violated or FFT allocation fails.
    pub fn reconstruct(
        &self,
        sensor_data: ArrayView2<f64>,
        sensor_positions: &[[f64; 3]],
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        let (n_time, _n_sensors) = sensor_data.dim();
        let [nx, ny, nz] = self.grid_size;
        let fft = get_fft_for_grid(nx, ny, nz);

        // Time step from sampling frequency (s).
        let dt = 1.0 / self.sampling_frequency;
        let c0 = self.sound_speed;

        // CFL stability check: c₀·dt/dx_min ≤ 1/√3 for 3-D pseudospectral
        // (Tabei et al. 2002, Appendix A).
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let cfl = c0 * dt / dx_min;
        if cfl > 1.0 / f64::sqrt(3.0) {
            return Err(kwavers_core::error::KwaversError::Numerical(
                kwavers_core::error::NumericalError::Instability {
                    operation: "Time reversal CFL".to_owned(),
                    condition: cfl,
                },
            ));
        }

        // ── k-vectors (rad/m) ─────────────────────────────────────────────
        // Standard DFT frequency ordering: k_n = n·(2π)/(N·dx) for n ≤ N/2,
        // k_n = (n − N)·(2π)/(N·dx) for n > N/2.
        let kx = Self::k_vector(nx, grid.dx);
        let ky = Self::k_vector(ny, grid.dy);
        let kz = Self::k_vector(nz, grid.dz);

        // ── Real cosine propagator cos(c₀·|k|·dt) ─────────────────────────
        // Tabei et al. 2002, Eq. 2: p̂(t+dt) = 2·cos(c₀·|k|·dt)·p̂(t) − p̂(t−dt).
        // The propagator is purely real; storing it as f64 avoids spurious
        // imaginary contributions in the update step.
        let propagator: Array3<f64> = {
            let mut prop = Array3::zeros((nx, ny, nz));
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let k2 = kx[i].mul_add(kx[i], ky[j].mul_add(ky[j], kz[k] * kz[k]));
                        let arg = c0 * dt * k2.sqrt();
                        prop[[i, j, k]] = arg.cos();
                    }
                }
            }
            prop
        };

        // ── Pre-compute sensor grid indices (nearest-neighbour) ────────────
        let sensor_indices: Vec<[usize; 3]> = sensor_positions
            .iter()
            .map(|pos| {
                [
                    ((pos[0] / grid.dx).round() as usize).min(nx - 1),
                    ((pos[1] / grid.dy).round() as usize).min(ny - 1),
                    ((pos[2] / grid.dz).round() as usize).min(nz - 1),
                ]
            })
            .collect();

        // ── Time-reversal loop ─────────────────────────────────────────────
        // Forward-time index `s` ∈ [0, n_time) maps to reversed sensor index
        // `n_time − 1 − s`. At pseudo-time step `s`, inject sensor_data row
        // `n_time − 1 − s` (Treeby et al. 2010, §2.3).
        let mut pressure = Array3::<f64>::zeros((nx, ny, nz));
        let mut pressure_prev = Array3::<f64>::zeros((nx, ny, nz));

        for s in 0..n_time {
            // Reversed sensor time index.
            let sensor_t = n_time - 1 - s;
            let sensor_row = sensor_data.row(sensor_t);

            // Hard Dirichlet source: replace field values at sensor locations.
            // (Treeby et al. 2010, §2.3; k-Wave `source.p_mode = 'dirichlet'`.)
            for (idx, &[si, sj, sk]) in sensor_indices.iter().enumerate() {
                pressure[[si, sj, sk]] = sensor_row[idx];
            }

            // k-space leapfrog: p_next = 2·cos(c·|k|·dt)·p − p_prev.
            let pressure_next =
                self.k_space_step(&pressure, &pressure_prev, &propagator, fft.as_ref())?;

            pressure_prev = pressure;
            pressure = pressure_next;

            // Re-apply hard source after propagation so the source is active
            // at every step (overrides what the propagator wrote at sensor sites).
            for (idx, &[si, sj, sk]) in sensor_indices.iter().enumerate() {
                pressure[[si, sj, sk]] = sensor_row[idx];
            }
        }

        // At the end of `n_time` steps the field equals p₀(r).
        Ok(pressure)
    }

    /// Construct the standard DFT k-vector for `n` points and spacing `dx`.
    ///
    /// `k[i] = i·(2π/(n·dx))` for `i ≤ n/2`, and `(i−n)·(2π/(n·dx))` otherwise.
    fn k_vector(n: usize, dx: f64) -> Vec<f64> {
        let dk = TWO_PI / (n as f64 * dx);
        (0..n)
            .map(|i| {
                if i <= n / 2 {
                    i as f64 * dk
                } else {
                    f64::from(i as i32 - n as i32) * dk
                }
            })
            .collect()
    }

    /// One k-space leapfrog time step.
    ///
    /// Computes `p_next(r) = IFFT{ 2·cos(c·|k|·dt)·FFT{p}(k) − FFT{p_prev}(k) }`.
    ///
    /// The propagator `cos(c·|k|·dt)` is real, so the update is exact and
    /// dispersion-free. (Tabei et al. 2002, Eq. 2.)
    fn k_space_step(
        &self,
        pressure: &Array3<f64>,
        pressure_prev: &Array3<f64>,
        propagator: &Array3<f64>,
        fft: &kwavers_math::fft::Fft3d,
    ) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = self.grid_size;

        // Forward FFT of current and previous pressure fields.
        let mut p_hat = Array3::<Complex64>::zeros((nx, ny, nz));
        let mut p_prev_hat = Array3::<Complex64>::zeros((nx, ny, nz));
        fft.forward_into(pressure, &mut p_hat);
        fft.forward_into(pressure_prev, &mut p_prev_hat);

        // Leapfrog update in k-space: p̂_next = 2·cos·p̂ − p̂_prev.
        // `propagator` is real (f64); multiply real cosine by complex spectrum.
        let mut p_next_hat = Array3::<Complex64>::zeros((nx, ny, nz));
        if let (Some(next_values), Some(propagator_values), Some(p_values), Some(prev_values)) = (
            p_next_hat.as_slice_memory_order_mut(),
            propagator.as_slice_memory_order(),
            p_hat.as_slice_memory_order(),
            p_prev_hat.as_slice_memory_order(),
        ) {
            enumerate_mut_with::<Adaptive, _, _>(next_values, |idx, next| {
                *next = Complex64::from(2.0 * propagator_values[idx]) * p_values[idx]
                    - prev_values[idx];
            });
        } else {
            for (((next, &cos_val), &p), &prev) in p_next_hat
                .iter_mut()
                .zip(propagator.iter())
                .zip(p_hat.iter())
                .zip(p_prev_hat.iter())
            {
                *next = Complex64::from(2.0 * cos_val) * p - prev;
            }
        }

        // Inverse FFT → real output.
        let mut out = Array3::<f64>::zeros((nx, ny, nz));
        let mut scratch = Array3::<Complex64>::zeros((nx, ny, nz));
        fft.inverse_into(&p_next_hat, &mut out, &mut scratch);
        Ok(out)
    }
}
