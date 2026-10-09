//! Broadband dispersive diffraction operator for the KZK equation.
//!
//! ## Motivation
//!
//! Nonlinear propagation generates harmonics at nf₀. The standard single-k₀
//! diffraction operators (`Parabolic`, `WideAngle`, Padé) all apply the same
//! phase velocity c₀ to every temporal frequency, introducing a phase error
//! of O((n−1)²) for the nth harmonic. For 30% bandwidth the third harmonic
//! accumulates ~26% phase error per Rayleigh distance.
//!
//! ## Transfer function
//!
//! For each temporal frequency bin ω_k:
//!
//! ```text
//! k_n = ω_k / c₀
//! H(k_T, ω_k) = exp(i (kz_k − k_n) Δz)
//! kz_k = √(k_n² − k_T²)        for k_T ≤ k_n (propagating)
//! kz_k = i√(k_T² − k_n²)       for k_T > k_n  (evanescent → decay)
//! ```
//!
//! This is the exact Helmholtz propagator evaluated per temporal harmonic,
//! equivalent to `WideAngle` at each frequency independently.
//!
//! ## Computational cost
//!
//! O(nt/2) × 2D FFT + multiply + IFFT over τ, vs 1 × 2D FFT in single-k₀
//! schemes. Use only when pulse bandwidth > ~20% or 3rd-harmonic accuracy matters.
//!
//! ## References
//!
//! - Zemp RJ, Tavakkoli J, Cobbold RS (2003). "Modeling of nonlinear
//!   ultrasound propagation in tissue from array transducers."
//!   J. Acoust. Soc. Am. 113(1), 139–152. DOI: 10.1121/1.1528926
//! - Huijssen J, Verweij MD (2010). J. Acoust. Soc. Am. 127(1), 33–44.

use apollo::{fft_2d_complex_inplace, ifft_2d_complex_inplace, Complex64 as ApolloComplex64};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::{fft_1d_complex_slice_inplace, ifft_1d_complex_slice_inplace, Complex64};
use leto::Array2;
use leto::Array2 as LetoArray2;
use leto::Array3;
use moirai_parallel::{enumerate_mut_with, for_each_chunk_mut_enumerated_with, Adaptive};

use super::KZKConfig;

/// Broadband exact Helmholtz diffraction operator applied per temporal bin.
pub struct BroadbandDiffractionOperator {
    config: KZKConfig,
    /// Precomputed transverse wavenumber magnitude squared k_T².
    kt2: Array2<f64>,
    /// Scratch buffer for the 2-D spatial FFT at one temporal bin.
    scratch: LetoArray2<ApolloComplex64>,
    /// Folded non-negative angular frequency magnitude per temporal bin.
    omega: Vec<f64>,
}

impl std::fmt::Debug for BroadbandDiffractionOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BroadbandDiffractionOperator")
            .field("config", &self.config)
            .field("kt2_shape", &self.kt2.shape())
            .field("scratch", &self.scratch.shape())
            .field("omega_len", &self.omega.len())
            .finish()
    }
}

impl BroadbandDiffractionOperator {
    /// Create a new broadband dispersive diffraction operator.
    #[must_use]
    pub fn new(config: &KZKConfig) -> Self {
        let nx = config.nx;
        let ny = config.ny;
        let nt = config.nt;
        let dkx = TWO_PI / (nx as f64 * config.dx);
        let dky = TWO_PI / (ny as f64 * config.dx);

        let mut kt2 = Array2::<f64>::zeros((nx, ny));
        for i in 0..nx {
            let kx = if i <= nx / 2 {
                i as f64 * dkx
            } else {
                (i as f64 - nx as f64) * dkx
            };
            let kx2 = kx * kx;
            for j in 0..ny {
                let ky = if j <= ny / 2 {
                    j as f64 * dky
                } else {
                    (j as f64 - ny as f64) * dky
                };
                kt2[[i, j]] = kx2 + ky * ky;
            }
        }

        let mut omega = vec![0.0_f64; nt];
        let domega = TWO_PI / (nt as f64 * config.dt);
        for (k, value) in omega.iter_mut().enumerate() {
            let folded_k = if k <= nt / 2 { k } else { nt - k };
            *value = folded_k as f64 * domega;
        }

        Self {
            config: config.clone(),
            kt2,
            scratch: LetoArray2::<ApolloComplex64>::zeros([nx, ny]),
            omega,
        }
    }

    /// Apply the broadband diffraction update to the full `[nx, ny, nt]` field.
    ///
    /// The pressure array itself is reused as the temporal-spectrum workspace:
    /// all time waveforms are FFT'd in place, each temporal bin is propagated
    /// through a shared 2-D spatial scratch buffer, then all waveforms are
    /// inverse transformed back to retarded time.
    pub fn apply_complex_full(&mut self, field: &mut Array3<Complex64>, step_size: f64) {
        let nt = self.config.nt;
        let ny = self.config.ny;
        let slab_len = ny * nt;
        let field_values = field
            .as_slice_mut()
            .expect("invariant: broadband diffraction field is standard-layout");

        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(field_values, slab_len, |_i, slab| {
            for row in slab.chunks_exact_mut(nt) {
                fft_1d_complex_slice_inplace(row);
            }
        });

        let kt2 = self
            .kt2
            .as_slice()
            .expect("invariant: broadband diffraction kt2 is standard-layout");
        for k in 0..nt {
            let omega_k = self.omega[k];
            if omega_k == 0.0 {
                continue;
            }

            {
                let scratch = self
                    .scratch
                    .as_slice_mut()
                    .expect("invariant: broadband diffraction scratch is standard-layout");
                enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, value| {
                    let mode = field_values[idx * nt + k];
                    *value = ApolloComplex64::new(mode.re, mode.im);
                });
            }

            fft_2d_complex_inplace(&mut self.scratch);

            let kn = omega_k / self.config.c0;
            let kn2 = kn * kn;
            {
                let scratch = self
                    .scratch
                    .as_slice_mut()
                    .expect("invariant: broadband diffraction scratch is standard-layout");
                enumerate_mut_with::<Adaptive, _, _>(scratch, |idx, value| {
                    let mode_kt2 = kt2[idx];
                    if mode_kt2 <= kn2 {
                        let kz = (kn2 - mode_kt2).sqrt();
                        let phase = (kz - kn) * step_size;
                        *value *= ApolloComplex64::from_polar(1.0, phase);
                    } else {
                        let decay = (-(mode_kt2 - kn2).sqrt() * step_size).exp();
                        *value *= ApolloComplex64::new(decay, 0.0);
                    }
                });
            }

            ifft_2d_complex_inplace(&mut self.scratch);

            let scratch = self
                .scratch
                .as_slice()
                .expect("invariant: broadband diffraction scratch is standard-layout");
            for (idx, value) in scratch.iter().enumerate() {
                field_values[idx * nt + k] = Complex64::new(value.re, value.im);
            }
        }

        for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(field_values, slab_len, |_i, slab| {
            for row in slab.chunks_exact_mut(nt) {
                ifft_1d_complex_slice_inplace(row);
            }
        });
    }
}
