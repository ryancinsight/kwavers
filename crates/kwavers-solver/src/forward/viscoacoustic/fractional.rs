//! Fractional k-space viscoacoustic absorption operator (Treeby & Cox 2010).
//!
//! Applies the exact power-law absorption filter in the spectral domain:
//!
//! ```text
//! H_abs(k) = exp(−α₀ |k|^(y−1) c₀ Δt)
//! ```
//!
//! where `|k|² = kx² + ky² + kz²` is the squared wavenumber magnitude.
//! For `y = 2` this recovers the classical Stokes thermoviscous absorption
//! `exp(−δ ω² Δt / (2c₀³))`.
//!
//! ## References
//!
//! - Treeby BE, Cox BT (2010). "Modeling power law absorption and dispersion
//!   for acoustic propagation using the fractional Laplacian."
//!   J. Acoust. Soc. Am. 127(5), 2741–2748. DOI: 10.1121/1.3377056
//! - Szabo TL (1994). "Time domain wave equations for lossy media obeying a
//!   frequency power law." J. Acoust. Soc. Am. 96(1), 491–500.

use std::f64::consts::PI;
use std::sync::Arc;

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d};
use leto::Array3;
use moirai_parallel::{enumerate_mut_with, Adaptive};

fn is_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn k_vector(n: usize, spacing: f64) -> Vec<f64> {
    let dk = TWO_PI / (n as f64 * spacing);
    (0..n)
        .map(|i| {
            if i <= n / 2 {
                i as f64 * dk
            } else {
                (i as f64 - n as f64) * dk
            }
        })
        .collect()
}

/// Exact power-law absorption/dispersion filter for a homogeneous medium.
#[derive(Debug, Clone)]
pub struct FractionalAbsorptionOperator {
    /// Pre-computed filter mask `H_abs(k) = exp(-alpha0 * |k|^(y-1) * c0 * dt)`,
    /// shape `[nx, ny, nz]`, applied in spectral domain.
    h_abs: Array3<f64>,
    /// Pre-computed companion dispersion phase for causal propagation.
    ///
    /// Stored as the phase angle θ(k) of `exp(i θ(k))`.
    h_disp: Array3<f64>,
    fft: Arc<Fft3d>,
}

impl FractionalAbsorptionOperator {
    /// Build the exact k-space absorption and causal-dispersion masks.
    ///
    /// # Errors
    /// Returns [`Err`] when a grid spacing, material parameter, or timestep is
    /// non-finite or physically invalid.
    #[must_use]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        alpha0_np_m: f64,
        y: f64,
        c0: f64,
        dt: f64,
    ) -> KwaversResult<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(
                "fractional viscoacoustic operator requires non-zero grid dimensions".to_owned(),
            ));
        }
        if !is_positive_finite(dx)
            || !is_positive_finite(dy)
            || !is_positive_finite(dz)
            || !is_positive_finite(c0)
            || !is_positive_finite(dt)
        {
            return Err(KwaversError::InvalidInput(
                "fractional viscoacoustic operator requires positive finite spacings, c0, and dt"
                    .to_owned(),
            ));
        }
        if !alpha0_np_m.is_finite() || alpha0_np_m < 0.0 {
            return Err(KwaversError::InvalidInput(
                "fractional viscoacoustic alpha0 must be finite and non-negative".to_owned(),
            ));
        }
        if !y.is_finite() {
            return Err(KwaversError::InvalidInput(
                "fractional viscoacoustic exponent must be finite".to_owned(),
            ));
        }

        let dispersion_factor = (PI * y / 2.0).tan();
        if !dispersion_factor.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "fractional viscoacoustic exponent y={y} makes tan(pi*y/2) non-finite"
            )));
        }

        let kx = k_vector(nx, dx);
        let ky = k_vector(ny, dy);
        let kz = k_vector(nz, dz);
        let mut h_abs = Array3::<f64>::zeros((nx, ny, nz));
        let mut h_disp = Array3::<f64>::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let k_mag = kx[i]
                        .mul_add(kx[i], ky[j].mul_add(ky[j], kz[k] * kz[k]))
                        .sqrt();
                    if k_mag == 0.0 {
                        h_abs[[i, j, k]] = 1.0;
                        h_disp[[i, j, k]] = 0.0;
                        continue;
                    }

                    let absorption_argument = alpha0_np_m * k_mag.powf(y - 1.0) * c0 * dt;
                    h_abs[[i, j, k]] = (-absorption_argument).exp();

                    let omega_ref = c0 * k_mag;
                    h_disp[[i, j, k]] =
                        dispersion_factor * alpha0_np_m * k_mag.powf(y - 1.0) * omega_ref * dt;
                }
            }
        }

        Ok(Self {
            h_abs,
            h_disp,
            fft: get_fft_for_grid(nx, ny, nz),
        })
    }

    /// Apply one absorption/dispersion step to the complex pressure field.
    pub fn apply(&self, pressure: &mut Array3<Complex64>) {
        self.fft.forward_complex_inplace(pressure);

        if let (Some(values), Some(abs_values), Some(phase_values)) = (
            pressure.as_slice_mut(),
            self.h_abs.as_slice(),
            self.h_disp.as_slice(),
        ) {
            enumerate_mut_with::<Adaptive, _, _>(values, |index, value| {
                let attenuation = abs_values[index];
                let phase = phase_values[index];
                let dispersion = Complex64::new(phase.cos(), phase.sin());
                *value *= dispersion * attenuation;
            });
        } else {
            let [nx, ny, nz] = pressure.shape();
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let attenuation = self.h_abs[[i, j, k]];
                        let phase = self.h_disp[[i, j, k]];
                        pressure[[i, j, k]] *=
                            Complex64::new(phase.cos(), phase.sin()) * attenuation;
                    }
                }
            }
        }

        self.fft.inverse_complex_inplace(pressure);
    }
}
