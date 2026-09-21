mod coefficient;
mod spectral;
mod update;

use crate::forward::lanes::LaneAxis;
use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use coefficient::compute_nonlinear_density_coefficient;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Fft3dInOutExt;
use spectral::apply_shifted_kappa;
use update::{update_density_fused, update_density_unfused};

// Implementation note on divergence caching:
// `update_density_cartesian` writes ∂u_α/∂α directly into `div_ux`/`div_uy`/`div_uz`
// (eliminating the intermediate `dpx/dpy/dpz` copy that was present pre-Opt-3).
// `apply_absorption_to_pressure` fuses div_u* into `dpx` as a single Zip (Opt-7 + Opt-12),
// then IFFTs to L1 in `dpx` and L2 in `dpy`.  The absorption read is unaffected because it
// always reads from `div_u*`.  Saves 3 × N-element memcpy per step.

impl PSTDSolver {
    /// Standard 3-D Cartesian density update.
    ///
    /// Uses staggered grid shift operators with kappa correction matching the C++ k-wave binary:
    ///   dux/dx = IFFT( ddx_k_shift_neg[x] * kappa[i,j,k] * FFT(ux)[i,j,k] )
    ///
    /// kappa IS applied here — Treeby & Cox (2010) Eq. 17 explicitly includes the k-space
    /// correction factor κ in the density update, same as in the velocity update (Eq. 16).
    ///
    /// ## Optimisations applied
    ///
    /// **IFFT → div_u directly**: the IFFT result is written into `div_ux`/`div_uy`/`div_uz`
    /// rather than first into `dpx`/`dpy`/`dpz` and then copied via `.assign()`.  The
    /// absorption kernel reads from `div_u*` directly and is unaffected.  Saves 3 × N
    /// element memcpy operations per step.
    ///
    /// **Fused PML + density update**: when `self.pml_exp` is populated (CPML boundary,
    /// no Dirichlet bypass), the split-field PML is applied inline:
    /// ```text
    ///   ρ_x^{n+1}[i,j,k] = p`i` · (p`i` · ρ_x^n − Δt · coef · ∂u_x/∂x)
    /// ```
    /// where `p`i` = pml_den_x`i` = exp(-σ_x`i`·Δt/2)`.  This replaces the previous
    /// `apply_pml_to_density()` pre/post calls with a single Zip pass per density component,
    /// saving 2 × N element writes per axis per step (6 passes eliminated for 3D).
    ///
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    #[inline]
    pub(crate) fn update_density_cartesian(&mut self, dt: f64) -> KwaversResult<()> {
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        // ── Opt: IFFT directly into div_u* — eliminates 3 × N memcpy ──────────
        // dux/dx with negative shift + kappa correction (Treeby & Cox 2010, Eq. 17).
        self.fft.forward_r2c_into(&self.fields.ux, &mut self.ux_k);
        apply_shifted_kappa(
            &mut self.grad_k,
            &self.ux_k,
            &self.kappa,
            &self.ddx_k_shift_neg,
            LaneAxis::X,
        );
        // Write IFFT result directly to div_ux; dpx is not used for density.
        self.fft
            .inverse_c2r_into(&mut self.grad_k, &mut self.div_ux);

        // duy/dy with negative shift + kappa (matches k-Wave Eq. 17).
        if has_y {
            self.fft.forward_r2c_into(&self.fields.uy, &mut self.ux_k);
            apply_shifted_kappa(
                &mut self.grad_k,
                &self.ux_k,
                &self.kappa,
                &self.ddy_k_shift_neg,
                LaneAxis::Y,
            );
            self.fft
                .inverse_c2r_into(&mut self.grad_k, &mut self.div_uy);
        } else {
            self.div_uy.fill(0.0);
        }

        // duz/dz with negative shift + kappa (matches k-Wave Eq. 17).
        if has_z {
            self.fft.forward_r2c_into(&self.fields.uz, &mut self.ux_k);
            apply_shifted_kappa(
                &mut self.grad_k,
                &self.ux_k,
                &self.kappa,
                &self.ddz_k_shift_neg,
                LaneAxis::Z,
            );
            self.fft
                .inverse_c2r_into(&mut self.grad_k, &mut self.div_uz);
        } else {
            self.div_uz.fill(0.0);
        }

        // ── Mass-conservation density update ───────────────────────────────────
        // Linear:    coef = rho0
        // Nonlinear: coef = rho0 + 2·(rhox + rhoy + rhoz)   [Westervelt]
        //
        // When pml_exp is available and there is no Dirichlet bypass, the PML is
        // fused into the density update: ρ = pml · (pml · ρ_old − Δt · coef · ∂u/∂α)
        // eliminating the two apply_pml_to_density() calls (pre + post, 6 field
        // passes for 3-D) in favour of one additional factor per element in the
        // already-necessary update loop.
        let use_fused = self.pml_exp.is_some() && self.dirichlet_pml_bypass_x.is_empty();

        if self.config.nonlinearity {
            compute_nonlinear_density_coefficient(
                &mut self.div_u,
                &self.materials.rho0,
                &self.rhox,
                &self.rhoy,
                &self.rhoz,
            );

            if use_fused {
                let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                    KwaversError::InternalError(
                        "pml_exp unexpectedly None in nonlinear density fused path".into(),
                    )
                })?;
                let pml_dx = pml_exp.den_x.as_slice().ok_or_else(|| {
                    KwaversError::InternalError("pml_den_x must be contiguous".into())
                })?;
                update_density_fused(
                    &mut self.rhox,
                    &self.div_ux,
                    &self.div_u,
                    pml_dx,
                    LaneAxis::X,
                    dt,
                );

                if has_y {
                    let pml_dy = pml_exp.den_y.as_slice().ok_or_else(|| {
                        KwaversError::InternalError("pml_den_y must be contiguous".into())
                    })?;
                    update_density_fused(
                        &mut self.rhoy,
                        &self.div_uy,
                        &self.div_u,
                        pml_dy,
                        LaneAxis::Y,
                        dt,
                    );
                }

                if has_z {
                    let pml_dz = pml_exp.den_z.as_slice().ok_or_else(|| {
                        KwaversError::InternalError("pml_den_z must be contiguous".into())
                    })?;
                    update_density_fused(
                        &mut self.rhoz,
                        &self.div_uz,
                        &self.div_u,
                        pml_dz,
                        LaneAxis::Z,
                        dt,
                    );
                }
            } else {
                // Fallback: pre-PML → update → post-PML
                self.apply_pml_to_density()?;

                update_density_unfused(&mut self.rhox, &self.div_ux, &self.div_u, dt);

                if has_y {
                    update_density_unfused(&mut self.rhoy, &self.div_uy, &self.div_u, dt);
                }

                if has_z {
                    update_density_unfused(&mut self.rhoz, &self.div_uz, &self.div_u, dt);
                }

                self.apply_pml_to_density()?;
            }
        } else {
            // Linear case
            if use_fused {
                let pml_exp = self.pml_exp.as_ref().ok_or_else(|| {
                    KwaversError::InternalError(
                        "pml_exp unexpectedly None in linear density fused path".into(),
                    )
                })?;
                let pml_dx = pml_exp.den_x.as_slice().ok_or_else(|| {
                    KwaversError::InternalError("pml_den_x must be contiguous".into())
                })?;
                update_density_fused(
                    &mut self.rhox,
                    &self.div_ux,
                    &self.materials.rho0,
                    pml_dx,
                    LaneAxis::X,
                    dt,
                );

                if has_y {
                    let pml_dy = pml_exp.den_y.as_slice().ok_or_else(|| {
                        KwaversError::InternalError("pml_den_y must be contiguous".into())
                    })?;
                    update_density_fused(
                        &mut self.rhoy,
                        &self.div_uy,
                        &self.materials.rho0,
                        pml_dy,
                        LaneAxis::Y,
                        dt,
                    );
                }

                if has_z {
                    let pml_dz = pml_exp.den_z.as_slice().ok_or_else(|| {
                        KwaversError::InternalError("pml_den_z must be contiguous".into())
                    })?;
                    update_density_fused(
                        &mut self.rhoz,
                        &self.div_uz,
                        &self.materials.rho0,
                        pml_dz,
                        LaneAxis::Z,
                        dt,
                    );
                }
            } else {
                // Fallback: pre-PML → update → post-PML
                self.apply_pml_to_density()?;

                update_density_unfused(&mut self.rhox, &self.div_ux, &self.materials.rho0, dt);

                if has_y {
                    update_density_unfused(&mut self.rhoy, &self.div_uy, &self.materials.rho0, dt);
                }

                if has_z {
                    update_density_unfused(&mut self.rhoz, &self.div_uz, &self.materials.rho0, dt);
                }

                self.apply_pml_to_density()?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{apply_shifted_kappa, update_density_fused};
    use crate::forward::lanes::{axis_index, LaneAxis};
    use kwavers_math::fft::Complex64;
    use leto::{Array1, Array3 as LetoArray3};

    /// One volume below the lane pass's parallel floor and one above it.
    const SHAPES: [[usize; 3]; 2] = [[5, 3, 7], [37, 29, 31]];
    const AXES: [LaneAxis; 3] = [LaneAxis::X, LaneAxis::Y, LaneAxis::Z];

    fn real_field(shape: [usize; 3], seed: f64) -> LetoArray3<f64> {
        let values = (0..shape.iter().product::<usize>())
            .map(|index| (index as f64).mul_add(0.754_8, seed).sin())
            .collect();
        LetoArray3::from_shape_vec(shape, values).expect("values match the shape")
    }

    fn complex_field(shape: [usize; 3], seed: f64) -> LetoArray3<Complex64> {
        let values = (0..shape.iter().product::<usize>())
            .map(|index| {
                let x = (index as f64).mul_add(0.618_0, seed);
                Complex64::new(x.sin(), x.cos())
            })
            .collect();
        LetoArray3::from_shape_vec(shape, values).expect("values match the shape")
    }

    fn table(shape: [usize; 3]) -> Vec<f64> {
        (0..*shape.iter().max().expect("a volume has three extents"))
            .map(|n| (n as f64).mul_add(-0.013, 1.0))
            .collect()
    }

    /// The lane pass computes the same expression, in the same order, as the
    /// per-element loop it replaced, so the fields agree to the bit.
    #[test]
    fn density_update_is_the_per_element_formula_to_the_bit() {
        let dt = 1.0e-7;
        for shape in SHAPES {
            let [nx, ny, nz] = shape;
            let pml = table(shape);
            let divergence = real_field(shape, 1.0);
            let coefficient = real_field(shape, 2.0);
            for axis in AXES {
                let mut density = real_field(shape, 3.0);
                let mut expected = density.clone();
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let p = pml[axis_index(axis, i, j, k)];
                            expected[[i, j, k]] = p
                                * (p * expected[[i, j, k]]
                                    - dt * coefficient[[i, j, k]] * divergence[[i, j, k]]);
                        }
                    }
                }
                update_density_fused(&mut density, &divergence, &coefficient, &pml, axis, dt);
                let same = density
                    .iter()
                    .zip(expected.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(same, "density update diverges at {shape:?}");
            }
        }
    }

    #[test]
    fn shifted_kappa_is_the_per_element_formula_to_the_bit() {
        for shape in SHAPES {
            let [nx, ny, nz] = shape;
            let spectrum = complex_field(shape, 1.0);
            let kappa = real_field(shape, 2.0);
            let shift_values: Vec<Complex64> = table(shape)
                .into_iter()
                .map(|v| Complex64::new(v, -v))
                .collect();
            let shift = Array1::from_vec([shift_values.len()], shift_values)
                .expect("values match the length");
            for axis in AXES {
                let mut gradient = complex_field(shape, 5.0);
                let mut expected = gradient.clone();
                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            expected[[i, j, k]] = (shift[axis_index(axis, i, j, k)]
                                * spectrum[[i, j, k]])
                                * kappa[[i, j, k]];
                        }
                    }
                }
                apply_shifted_kappa(&mut gradient, &spectrum, &kappa, &shift, axis);
                let same = gradient.iter().zip(expected.iter()).all(|(a, b)| {
                    a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits()
                });
                assert!(same, "shifted kappa diverges at {shape:?}");
            }
        }
    }
}
