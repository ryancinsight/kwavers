//! Imaging conditions for Reverse Time Migration.
//!
//! All six k-Wave/industry-standard conditions are implemented.
//! Triple-nested-for-loop gradient stencils (Poynting, Laplacian) are
//! replaced with Moirai-backed strided view passes.
//!
//! # Imaging conditions
//!
//! | Variant | Formula |
//! |---------|---------|
//! | `ZeroLag` | `I = Σ_t S·R` |
//! | `Normalized` | `I = Σ_t S·R / √(Σ_t S²·Σ_t R²)` |
//! | `Laplacian` | `I = Σ_t ∇²S·R` |
//! | `EnergyNormalized` | `I = Σ_t S·R / Σ_t S²` |
//! | `SourceNormalized` | `I = Σ_t (∂S/∂t)·R` (centred difference) |
//! | `Poynting` | `I = Σ_t ∇S·∇R` (centred gradient, interior only) |
//!
//! ## Poynting vectorisation
//!
//! The centred gradient at interior point `(i,j,k)`:
//! ```text
//! ∂S/∂x ≈ 0.5·(S[i+1,j,k] − S[i-1,j,k])
//! ```
//! The dot product `∇S·∇R = (∂S/∂x)(∂R/∂x) + (∂S/∂y)(∂R/∂y) + (∂S/∂z)(∂R/∂z)`
//! equals `0.25·(S[i+1]−S[i-1])·(R[i+1]−R[i-1]) + …`.
//!
//! Each directional term is a separate Moirai pass on the interior slice.
//!
//! Reference: Zhang & Sun (2009), "Practical issues in reverse time
//! migration", *The Leading Edge* **28**(4), 446–452.

use kwavers_core::error::KwaversResult;
use leto::{
    /* s -- no leto equivalent */,
    Array3,
    Array4,
};

use super::super::super::config::RtmImagingCondition;
use super::super::super::constants::RTM_AMPLITUDE_THRESHOLD;
use super::super::super::constants::RTM_LAPLACIAN_SCALING;
use super::super::types::ReverseTimeMigration;
use super::parallel::for_each_view_mut;

impl ReverseTimeMigration {
    /// Cross-correlate `source_wavefield` and `receiver_wavefield` according
    /// to `self.config.rtm_imaging_condition` and accumulate into
    /// `self.image`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn apply_imaging_condition(
        &mut self,
        source_wavefield: &Array4<f64>,
        receiver_wavefield: &Array4<f64>,
    ) -> KwaversResult<()> {
        let n_time_steps = source_wavefield.shape()[0];

        match self.config.rtm_imaging_condition {
            // ── ZeroLag ───────────────────────────────────────────────────
            RtmImagingCondition::ZeroLag => {
                for t in 0..n_time_steps {
                    let src = source_wavefield.slice(s![t, .., .., ..]);
                    let rcv = receiver_wavefield.slice(s![t, .., .., ..]);

                    for_each_view_mut(self.image.view_mut(), |idx, img| {
                        let s = src[idx];
                        if s.abs() > RTM_AMPLITUDE_THRESHOLD {
                            *img += s * rcv[idx];
                        }
                    });
                }
            }

            // ── Normalized ────────────────────────────────────────────────
            RtmImagingCondition::Normalized => {
                let mut src_energy = Array3::<f64>::zeros(self.image.dim());
                let mut rcv_energy = Array3::<f64>::zeros(self.image.dim());

                for t in 0..n_time_steps {
                    let src = source_wavefield.slice(s![t, .., .., ..]);
                    let rcv = receiver_wavefield.slice(s![t, .., .., ..]);

                    for_each_view_mut(self.image.view_mut(), |idx, img| {
                        *img += src[idx] * rcv[idx];
                    });

                    for_each_view_mut(src_energy.view_mut(), |idx, energy| {
                        let s = src[idx];
                        *energy += s * s;
                    });

                    for_each_view_mut(rcv_energy.view_mut(), |idx, energy| {
                        let r = rcv[idx];
                        *energy += r * r;
                    });
                }

                let src_energy = src_energy.view();
                let rcv_energy = rcv_energy.view();
                for_each_view_mut(self.image.view_mut(), |idx, img| {
                    let norm = (src_energy[idx] * rcv_energy[idx]).sqrt();
                    if norm > RTM_AMPLITUDE_THRESHOLD {
                        *img /= norm;
                    }
                });
            }

            // ── Laplacian ─────────────────────────────────────────────────
            RtmImagingCondition::Laplacian => {
                for t in 0..n_time_steps {
                    let src = source_wavefield.slice(s![t, .., .., ..]).to_owned();
                    let rcv = receiver_wavefield.slice(s![t, .., .., ..]);

                    let src_lap = self.compute_laplacian(&src)?;

                    let src_lap = src_lap.view();
                    for_each_view_mut(self.image.view_mut(), |idx, img| {
                        *img += RTM_LAPLACIAN_SCALING * src_lap[idx] * rcv[idx];
                    });
                }
            }

            // ── EnergyNormalized ──────────────────────────────────────────
            RtmImagingCondition::EnergyNormalized => {
                let mut src_energy = Array3::<f64>::zeros(self.image.dim());

                for t in 0..n_time_steps {
                    let src = source_wavefield.slice(s![t, .., .., ..]);
                    let rcv = receiver_wavefield.slice(s![t, .., .., ..]);

                    for_each_view_mut(self.image.view_mut(), |idx, img| {
                        *img += src[idx] * rcv[idx];
                    });

                    for_each_view_mut(src_energy.view_mut(), |idx, energy| {
                        let s = src[idx];
                        *energy += s * s;
                    });
                }

                let src_energy = src_energy.view();
                for_each_view_mut(self.image.view_mut(), |idx, img| {
                    let energy = src_energy[idx];
                    if energy > RTM_AMPLITUDE_THRESHOLD {
                        *img /= energy;
                    }
                });
            }

            // ── SourceNormalized ──────────────────────────────────────────
            // I(x) = Σ_t (∂S/∂t)·R   via centred differences
            RtmImagingCondition::SourceNormalized => {
                for t in 0..n_time_steps {
                    let rcv = receiver_wavefield.slice(s![t, .., .., ..]);
                    let src_dt = if t == 0 && n_time_steps > 1 {
                        &source_wavefield.slice(s![1, .., .., ..])
                            - &source_wavefield.slice(s![0, .., .., ..])
                    } else if t + 1 == n_time_steps && n_time_steps > 1 {
                        &source_wavefield.slice(s![t, .., .., ..])
                            - &source_wavefield.slice(s![t - 1, .., .., ..])
                    } else if n_time_steps > 1 {
                        0.5 * (&source_wavefield.slice(s![t + 1, .., .., ..])
                            - &source_wavefield.slice(s![t - 1, .., .., ..]))
                    } else {
                        Array3::<f64>::zeros(self.image.dim())
                    };

                    for_each_view_mut(self.image.view_mut(), |idx, img| {
                        *img += src_dt[idx] * rcv[idx];
                    });
                }
            }

            // ── Poynting ──────────────────────────────────────────────────
            // I(x) = Σ_t ∇S·∇R  (centred gradient dot-product, interior only)
            //
            // Each directional term: 0.25·(S[+1]−S[-1])·(R[+1]−R[-1])
            // Three sequential Moirai passes over same-shape strided views.
            RtmImagingCondition::Poynting => {
                let (_, nx, ny, nz) = source_wavefield.dim();
                Self::ensure_3d_interior((nx, ny, nz))?;
                let inn = s![1..nx - 1, 1..ny - 1, 1..nz - 1];

                for t in 0..n_time_steps {
                    let src = source_wavefield.slice(s![t, .., .., ..]);
                    let rcv = receiver_wavefield.slice(s![t, .., .., ..]);

                    // x: 0.25·(S[i+1]−S[i-1])·(R[i+1]−R[i-1])
                    let sxp = src.slice(s![2..nx, 1..ny - 1, 1..nz - 1]);
                    let sxm = src.slice(s![..nx - 2, 1..ny - 1, 1..nz - 1]);
                    let rxp = rcv.slice(s![2..nx, 1..ny - 1, 1..nz - 1]);
                    let rxm = rcv.slice(s![..nx - 2, 1..ny - 1, 1..nz - 1]);
                    for_each_view_mut(self.image.slice_mut(inn), |idx, img| {
                        *img += 0.25 * (sxp[idx] - sxm[idx]) * (rxp[idx] - rxm[idx]);
                    });

                    // y
                    let syp = src.slice(s![1..nx - 1, 2..ny, 1..nz - 1]);
                    let sym = src.slice(s![1..nx - 1, ..ny - 2, 1..nz - 1]);
                    let ryp = rcv.slice(s![1..nx - 1, 2..ny, 1..nz - 1]);
                    let rym = rcv.slice(s![1..nx - 1, ..ny - 2, 1..nz - 1]);
                    for_each_view_mut(self.image.slice_mut(inn), |idx, img| {
                        *img += 0.25 * (syp[idx] - sym[idx]) * (ryp[idx] - rym[idx]);
                    });

                    // z
                    let szp = src.slice(s![1..nx - 1, 1..ny - 1, 2..nz]);
                    let szm = src.slice(s![1..nx - 1, 1..ny - 1, ..nz - 2]);
                    let rzp = rcv.slice(s![1..nx - 1, 1..ny - 1, 2..nz]);
                    let rzm = rcv.slice(s![1..nx - 1, 1..ny - 1, ..nz - 2]);
                    for_each_view_mut(self.image.slice_mut(inn), |idx, img| {
                        *img += 0.25 * (szp[idx] - szm[idx]) * (rzp[idx] - rzm[idx]);
                    });
                }
            }
        }

        Ok(())
    }
}
