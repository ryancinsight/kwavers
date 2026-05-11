//! Imaging conditions for Reverse Time Migration.
//!
//! All six k-Wave/industry-standard conditions are implemented.
//! Triple-nested-for-loop gradient stencils (Poynting, Laplacian) are
//! replaced with `Zip`/`Zip::par_for_each` slice-view passes.
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
//! Each directional term is a separate `Zip` pass on the interior slice,
//! with 5 arrays (1 mut + 4 read), well within ndarray's limit.
//!
//! Reference: Zhang & Sun (2009), "Practical issues in reverse time
//! migration", *The Leading Edge* **28**(4), 446–452.

use crate::core::error::KwaversResult;
use ndarray::{s, Array3, Array4, Zip};

use super::super::super::config::RtmImagingCondition;
use super::super::super::constants::RTM_AMPLITUDE_THRESHOLD;
use super::super::super::constants::RTM_LAPLACIAN_SCALING;
use super::super::types::ReverseTimeMigration;

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

                    Zip::from(&mut self.image)
                        .and(&src)
                        .and(&rcv)
                        .par_for_each(|img, &s, &r| {
                            if s.abs() > RTM_AMPLITUDE_THRESHOLD {
                                *img += s * r;
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

                    Zip::from(&mut self.image)
                        .and(&src)
                        .and(&rcv)
                        .par_for_each(|img, &s, &r| *img += s * r);

                    Zip::from(&mut src_energy)
                        .and(&src)
                        .par_for_each(|e, &s| *e += s * s);

                    Zip::from(&mut rcv_energy)
                        .and(&rcv)
                        .par_for_each(|e, &r| *e += r * r);
                }

                Zip::from(&mut self.image)
                    .and(&src_energy)
                    .and(&rcv_energy)
                    .par_for_each(|img, &se, &re| {
                        let norm = (se * re).sqrt();
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

                    Zip::from(&mut self.image)
                        .and(&src_lap)
                        .and(&rcv)
                        .par_for_each(|img, &lap, &r| {
                            *img += RTM_LAPLACIAN_SCALING * lap * r;
                        });
                }
            }

            // ── EnergyNormalized ──────────────────────────────────────────
            RtmImagingCondition::EnergyNormalized => {
                let mut src_energy = Array3::<f64>::zeros(self.image.dim());

                for t in 0..n_time_steps {
                    let src = source_wavefield.slice(s![t, .., .., ..]);
                    let rcv = receiver_wavefield.slice(s![t, .., .., ..]);

                    Zip::from(&mut self.image)
                        .and(&src)
                        .and(&rcv)
                        .par_for_each(|img, &s, &r| *img += s * r);

                    Zip::from(&mut src_energy)
                        .and(&src)
                        .par_for_each(|e, &s| *e += s * s);
                }

                Zip::from(&mut self.image)
                    .and(&src_energy)
                    .par_for_each(|img, &energy| {
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

                    Zip::from(&mut self.image)
                        .and(&src_dt)
                        .and(&rcv)
                        .par_for_each(|img, &ds, &r| *img += ds * r);
                }
            }

            // ── Poynting ──────────────────────────────────────────────────
            // I(x) = Σ_t ∇S·∇R  (centred gradient dot-product, interior only)
            //
            // Each directional term: 0.25·(S[+1]−S[-1])·(R[+1]−R[-1])
            // Three sequential Zip passes, 5 arrays each (1 mut + 4 read).
            RtmImagingCondition::Poynting => {
                let (_, nx, ny, nz) = source_wavefield.dim();
                let inn = s![1..nx - 1, 1..ny - 1, 1..nz - 1];

                for t in 0..n_time_steps {
                    let src = source_wavefield.slice(s![t, .., .., ..]);
                    let rcv = receiver_wavefield.slice(s![t, .., .., ..]);

                    // x: 0.25·(S[i+1]−S[i-1])·(R[i+1]−R[i-1])
                    Zip::from(self.image.slice_mut(inn))
                        .and(&src.slice(s![2..nx, 1..ny - 1, 1..nz - 1]))
                        .and(&src.slice(s![..nx - 2, 1..ny - 1, 1..nz - 1]))
                        .and(&rcv.slice(s![2..nx, 1..ny - 1, 1..nz - 1]))
                        .and(&rcv.slice(s![..nx - 2, 1..ny - 1, 1..nz - 1]))
                        .par_for_each(|img, &sxp, &sxm, &rxp, &rxm| {
                            *img += 0.25 * (sxp - sxm) * (rxp - rxm);
                        });

                    // y
                    Zip::from(self.image.slice_mut(inn))
                        .and(&src.slice(s![1..nx - 1, 2..ny, 1..nz - 1]))
                        .and(&src.slice(s![1..nx - 1, ..ny - 2, 1..nz - 1]))
                        .and(&rcv.slice(s![1..nx - 1, 2..ny, 1..nz - 1]))
                        .and(&rcv.slice(s![1..nx - 1, ..ny - 2, 1..nz - 1]))
                        .par_for_each(|img, &syp, &sym, &ryp, &rym| {
                            *img += 0.25 * (syp - sym) * (ryp - rym);
                        });

                    // z
                    Zip::from(self.image.slice_mut(inn))
                        .and(&src.slice(s![1..nx - 1, 1..ny - 1, 2..nz]))
                        .and(&src.slice(s![1..nx - 1, 1..ny - 1, ..nz - 2]))
                        .and(&rcv.slice(s![1..nx - 1, 1..ny - 1, 2..nz]))
                        .and(&rcv.slice(s![1..nx - 1, 1..ny - 1, ..nz - 2]))
                        .par_for_each(|img, &szp, &szm, &rzp, &rzm| {
                            *img += 0.25 * (szp - szm) * (rzp - rzm);
                        });
                }
            }
        }

        Ok(())
    }
}
