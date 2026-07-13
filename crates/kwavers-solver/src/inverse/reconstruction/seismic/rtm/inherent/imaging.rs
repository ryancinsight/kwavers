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
use leto::{Array3, Array4};

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
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
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
                    let src = source_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");
                    let rcv = receiver_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");

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
                let mut src_energy = Array3::<f64>::zeros(self.image.shape());
                let mut rcv_energy = Array3::<f64>::zeros(self.image.shape());

                for t in 0..n_time_steps {
                    let src = source_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");
                    let rcv = receiver_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");

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
                    let src = source_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");
                    let rcv = receiver_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");

                    let src_lap = self.compute_laplacian(&src)?;

                    let src_lap = src_lap.view();
                    for_each_view_mut(self.image.view_mut(), |idx, img| {
                        *img += RTM_LAPLACIAN_SCALING * src_lap[idx] * rcv[idx];
                    });
                }
            }

            // ── EnergyNormalized ──────────────────────────────────────────
            RtmImagingCondition::EnergyNormalized => {
                let mut src_energy = Array3::<f64>::zeros(self.image.shape());

                for t in 0..n_time_steps {
                    let src = source_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");
                    let rcv = receiver_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");

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
                    let rcv = receiver_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");
                    // Centred temporal derivative ∂S/∂t: forward diff at t=0,
                    // backward diff at the final step, centred (×0.5) in the interior.
                    let src_dt: Array3<f64> = if n_time_steps > 1 {
                        let (hi, lo, scale) = if t == 0 {
                            (1usize, 0usize, 1.0)
                        } else if t + 1 == n_time_steps {
                            (t, t - 1, 1.0)
                        } else {
                            (t + 1, t - 1, 0.5)
                        };
                        let a = source_wavefield
                            .slice_with::<3>(&s![hi, .., .., ..])
                            .expect("invariant: RTM forward time index in range");
                        let b = source_wavefield
                            .slice_with::<3>(&s![lo, .., .., ..])
                            .expect("invariant: RTM backward time index in range");
                        let mut out = Array3::<f64>::zeros(self.image.shape());
                        leto_ops::sub(&a, &b, &mut out.view_mut())
                            .expect("invariant: RTM temporal-difference shapes match");
                        if scale != 1.0 {
                            for value in out
                                .as_slice_mut()
                                .expect("invariant: RTM difference buffer contiguous")
                            {
                                *value *= scale;
                            }
                        }
                        out
                    } else {
                        Array3::<f64>::zeros(self.image.shape())
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
                let [_, nx, ny, nz] = source_wavefield.shape();
                Self::ensure_3d_interior((nx, ny, nz))?;
                let inn = s![1..nx - 1, 1..ny - 1, 1..nz - 1];

                for t in 0..n_time_steps {
                    let src = source_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");
                    let rcv = receiver_wavefield
                        .slice_with::<3>(&s![t, .., .., ..])
                        .expect("invariant: RTM slice indices in range");

                    // x: 0.25·(S[i+1]−S[i-1])·(R[i+1]−R[i-1])
                    let sxp = src
                        .slice_with::<3>(&s![2..nx, 1..ny - 1, 1..nz - 1])
                        .expect("invariant: RTM slice indices in range");
                    let sxm = src
                        .slice_with::<3>(&s![..nx - 2, 1..ny - 1, 1..nz - 1])
                        .expect("invariant: RTM slice indices in range");
                    let rxp = rcv
                        .slice_with::<3>(&s![2..nx, 1..ny - 1, 1..nz - 1])
                        .expect("invariant: RTM slice indices in range");
                    let rxm = rcv
                        .slice_with::<3>(&s![..nx - 2, 1..ny - 1, 1..nz - 1])
                        .expect("invariant: RTM slice indices in range");
                    for_each_view_mut(
                        self.image
                            .slice_with_mut::<3>(&inn)
                            .expect("invariant: RTM interior slice in range"),
                        |idx, img| {
                            *img += 0.25 * (sxp[idx] - sxm[idx]) * (rxp[idx] - rxm[idx]);
                        },
                    );

                    // y
                    let syp = src
                        .slice_with::<3>(&s![1..nx - 1, 2..ny, 1..nz - 1])
                        .expect("invariant: RTM slice indices in range");
                    let sym = src
                        .slice_with::<3>(&s![1..nx - 1, ..ny - 2, 1..nz - 1])
                        .expect("invariant: RTM slice indices in range");
                    let ryp = rcv
                        .slice_with::<3>(&s![1..nx - 1, 2..ny, 1..nz - 1])
                        .expect("invariant: RTM slice indices in range");
                    let rym = rcv
                        .slice_with::<3>(&s![1..nx - 1, ..ny - 2, 1..nz - 1])
                        .expect("invariant: RTM slice indices in range");
                    for_each_view_mut(
                        self.image
                            .slice_with_mut::<3>(&inn)
                            .expect("invariant: RTM interior slice in range"),
                        |idx, img| {
                            *img += 0.25 * (syp[idx] - sym[idx]) * (ryp[idx] - rym[idx]);
                        },
                    );

                    // z
                    let szp = src
                        .slice_with::<3>(&s![1..nx - 1, 1..ny - 1, 2..nz])
                        .expect("invariant: RTM slice indices in range");
                    let szm = src
                        .slice_with::<3>(&s![1..nx - 1, 1..ny - 1, ..nz - 2])
                        .expect("invariant: RTM slice indices in range");
                    let rzp = rcv
                        .slice_with::<3>(&s![1..nx - 1, 1..ny - 1, 2..nz])
                        .expect("invariant: RTM slice indices in range");
                    let rzm = rcv
                        .slice_with::<3>(&s![1..nx - 1, 1..ny - 1, ..nz - 2])
                        .expect("invariant: RTM slice indices in range");
                    for_each_view_mut(
                        self.image
                            .slice_with_mut::<3>(&inn)
                            .expect("invariant: RTM interior slice in range"),
                        |idx, img| {
                            *img += 0.25 * (szp[idx] - szm[idx]) * (rzp[idx] - rzm[idx]);
                        },
                    );
                }
            }
        }

        Ok(())
    }
}
