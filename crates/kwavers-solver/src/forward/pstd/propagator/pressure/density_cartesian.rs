use crate::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use crate::forward::pstd::propagator::lanes::{axis_index, for_each_z_lane, LaneAxis};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{Complex64, Fft3dInOutExt};
use leto::Array1;
use leto::Array3 as LetoArray3;
use moirai_parallel::{enumerate_mut_with, Adaptive};

// Implementation note on divergence caching:
// `update_density_cartesian` writes ∂u_α/∂α directly into `div_ux`/`div_uy`/`div_uz`
// (eliminating the intermediate `dpx/dpy/dpz` copy that was present pre-Opt-3).
// `apply_absorption_to_pressure` fuses div_u* into `dpx` as a single Zip (Opt-7 + Opt-12),
// then IFFTs to L1 in `dpx` and L2 in `dpy`.  The absorption read is unaffected because it
// always reads from `div_u*`.  Saves 3 × N-element memcpy per step.

trait DenseRealField {
    fn shape3(&self) -> [usize; 3];
    fn as_dense_slice(&self) -> Option<&[f64]>;
    fn value(&self, i: usize, j: usize, k: usize) -> f64;
}

impl DenseRealField for LetoArray3<f64> {
    fn shape3(&self) -> [usize; 3] {
        self.shape()
    }

    fn as_dense_slice(&self) -> Option<&[f64]> {
        self.as_slice()
    }

    fn value(&self, i: usize, j: usize, k: usize) -> f64 {
        self[[i, j, k]]
    }
}

fn apply_shifted_kappa(
    grad_k: &mut LetoArray3<Complex64>,
    spectrum: &LetoArray3<Complex64>,
    kappa: &LetoArray3<f64>,
    shift: &Array1<Complex64>,
    axis: LaneAxis,
) {
    assert_eq!(
        grad_k.shape(),
        spectrum.shape(),
        "invariant: PSTD gradient spectrum shape matches velocity spectrum"
    );
    assert_eq!(
        grad_k.shape(),
        kappa.shape(),
        "invariant: PSTD gradient spectrum shape matches kappa"
    );

    let [_nx, ny, nz] = grad_k.shape();
    if let (Some(grad_values), Some(spectrum_values), Some(kappa_values), Some(shift_values)) = (
        grad_k.as_slice_mut(),
        spectrum.as_slice(),
        kappa.as_slice(),
        shift.as_slice(),
    ) {
        let element_bytes = 2 * size_of::<Complex64>() + size_of::<f64>();
        for_each_z_lane(grad_values, [ny, nz], element_bytes, |start, i, j, grad| {
            let spectrum = &spectrum_values[start..start + nz];
            let kappa = &kappa_values[start..start + nz];
            let lane = grad.iter_mut().zip(spectrum).zip(kappa);
            match axis {
                LaneAxis::X | LaneAxis::Y => {
                    let shift = shift_values[axis_index(axis, i, j, 0)];
                    for ((grad, &spectrum), &kappa) in lane {
                        *grad = (shift * spectrum) * kappa;
                    }
                }
                LaneAxis::Z => {
                    for (((grad, &spectrum), &kappa), &shift) in lane.zip(&shift_values[..nz]) {
                        *grad = (shift * spectrum) * kappa;
                    }
                }
            }
        });
        return;
    }

    let [nx, ny, nz] = grad_k.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                grad_k[[i, j, k]] =
                    (shift[axis_index(axis, i, j, k)] * spectrum[[i, j, k]]) * kappa[[i, j, k]];
            }
        }
    }
}

fn compute_nonlinear_density_coefficient(
    coefficient: &mut LetoArray3<f64>,
    rho0: &LetoArray3<f64>,
    rhox: &LetoArray3<f64>,
    rhoy: &LetoArray3<f64>,
    rhoz: &LetoArray3<f64>,
) {
    assert_eq!(
        coefficient.shape(),
        rho0.shape(),
        "invariant: PSTD nonlinear coefficient shape matches rho0"
    );
    assert_eq!(
        coefficient.shape(),
        rhox.shape(),
        "invariant: PSTD nonlinear coefficient shape matches rhox"
    );
    assert_eq!(
        coefficient.shape(),
        rhoy.shape(),
        "invariant: PSTD nonlinear coefficient shape matches rhoy"
    );
    assert_eq!(
        coefficient.shape(),
        rhoz.shape(),
        "invariant: PSTD nonlinear coefficient shape matches rhoz"
    );

    if let (
        Some(coef_values),
        Some(rho0_values),
        Some(rx_values),
        Some(ry_values),
        Some(rz_values),
    ) = (
        coefficient.as_slice_mut(),
        rho0.as_slice(),
        rhox.as_slice(),
        rhoy.as_slice(),
        rhoz.as_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(coef_values, |index, coefficient| {
            *coefficient = 2.0f64.mul_add(
                rx_values[index] + ry_values[index] + rz_values[index],
                rho0_values[index],
            );
        });
        return;
    }

    let [nx, ny, nz] = coefficient.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                coefficient[[i, j, k]] = 2.0f64.mul_add(
                    rhox[[i, j, k]] + rhoy[[i, j, k]] + rhoz[[i, j, k]],
                    rho0[[i, j, k]],
                );
            }
        }
    }
}

fn update_density_fused(
    density: &mut LetoArray3<f64>,
    divergence: &LetoArray3<f64>,
    coefficient: &impl DenseRealField,
    pml: &[f64],
    axis: LaneAxis,
    dt: f64,
) {
    assert_eq!(
        density.shape(),
        divergence.shape(),
        "invariant: PSTD density shape matches divergence"
    );
    assert_eq!(
        density.shape(),
        coefficient.shape3(),
        "invariant: PSTD density shape matches update coefficient"
    );

    let [_nx, ny, nz] = density.shape();
    if let (Some(density_values), Some(div_values), Some(coef_values)) = (
        density.as_slice_mut(),
        divergence.as_slice(),
        coefficient.as_dense_slice(),
    ) {
        for_each_z_lane(
            density_values,
            [ny, nz],
            3 * size_of::<f64>(),
            |start, i, j, density| {
                let coefficient = &coef_values[start..start + nz];
                let divergence = &div_values[start..start + nz];
                let lane = density.iter_mut().zip(coefficient).zip(divergence);
                match axis {
                    LaneAxis::X | LaneAxis::Y => {
                        let p = pml[axis_index(axis, i, j, 0)];
                        for ((density, &coefficient), &divergence) in lane {
                            *density = p * (p * *density - dt * coefficient * divergence);
                        }
                    }
                    LaneAxis::Z => {
                        for (((density, &coefficient), &divergence), &p) in lane.zip(&pml[..nz]) {
                            *density = p * (p * *density - dt * coefficient * divergence);
                        }
                    }
                }
            },
        );
        return;
    }

    let [nx, ny, nz] = density.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let p = pml[axis_index(axis, i, j, k)];
                density[[i, j, k]] = p
                    * (p * density[[i, j, k]]
                        - dt * coefficient.value(i, j, k) * divergence[[i, j, k]]);
            }
        }
    }
}

fn update_density_unfused(
    density: &mut LetoArray3<f64>,
    divergence: &LetoArray3<f64>,
    coefficient: &impl DenseRealField,
    dt: f64,
) {
    assert_eq!(
        density.shape(),
        divergence.shape(),
        "invariant: PSTD density shape matches divergence"
    );
    assert_eq!(
        density.shape(),
        coefficient.shape3(),
        "invariant: PSTD density shape matches update coefficient"
    );

    if let (Some(density_values), Some(div_values), Some(coef_values)) = (
        density.as_slice_mut(),
        divergence.as_slice(),
        coefficient.as_dense_slice(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(density_values, |index, density| {
            *density -= dt * coef_values[index] * div_values[index];
        });
        return;
    }

    let [nx, ny, nz] = density.shape();
    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                density[[i, j, k]] -= dt * coefficient.value(i, j, k) * divergence[[i, j, k]];
            }
        }
    }
}

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
    use super::{apply_shifted_kappa, axis_index, update_density_fused, LaneAxis};
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
