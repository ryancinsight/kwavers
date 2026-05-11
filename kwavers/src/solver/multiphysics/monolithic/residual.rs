use super::coupler::MonolithicCoupler;
use super::utils::laplacian_3d;
use crate::core::error::KwaversResult;
use crate::domain::field::UnifiedFieldType;
use ndarray::{s, Array3};

impl MonolithicCoupler {
    /// Compute residual F(u) = u − u_prev − Δt·R(u)
    ///
    /// Evaluates the implicit residual for the coupled acoustic–optical–thermal
    /// system.  Each physics field occupies a contiguous block of `nx` rows in
    /// the flattened `Array3<f64>` (total shape: `n_fields*nx × ny × nz`).
    ///
    /// **Rate terms R(u) by field type:**
    ///
    /// | Field | R(u) |
    /// |-------|------|
    /// | Pressure | c²·∇²p + α_ac·μ_a·I (photoacoustic source) |
    /// | LightFluence | D·∇²I − μ_a·I |
    /// | Temperature | κ·∇²T + μ_a·I/(ρ·cₚ) + α_ac·p²/(ρ·c·ρ·cₚ) |
    /// | Other | 0 (identity: F = u − u_prev) |
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_residual(
        &self,
        u: &Array3<f64>,
        u_prev: &Array3<f64>,
        dt: f64,
        grid_dims: (usize, usize, usize),
        field_order: &[UnifiedFieldType],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = grid_dims;
        let _n_fields = field_order.len();
        let (dx, dy, dz) = self.grid_spacing;

        // Start with F(u) = u − u_prev
        let mut residual = u - u_prev;

        // Build quick-lookup slices for cross-coupling
        let field_slice = |arr: &Array3<f64>, idx: usize| -> Array3<f64> {
            arr.slice(s![idx * nx..(idx + 1) * nx, .., ..]).to_owned()
        };

        // Index map: field_type -> block index   (None if field not present)
        let idx_of =
            |ft: UnifiedFieldType| -> Option<usize> { field_order.iter().position(|&k| k == ft) };

        // Pre-extract fields needed for cross-coupling (clones are O(nx·ny·nz))
        let pressure = idx_of(UnifiedFieldType::Pressure).map(|i| field_slice(u, i));
        let light = idx_of(UnifiedFieldType::LightFluence).map(|i| field_slice(u, i));

        let coeff = &self.physics_coefficients;

        // Laplacian scratch buffers are allocated locally per iteration since
        // `compute_residual` takes `&self` (not `&mut self`), preventing use of
        // a persistent struct-level scratch field without unsafe aliasing.
        for (block, &ft) in field_order.iter().enumerate() {
            let row_start = block * nx;
            let field_block = field_slice(u, block);

            // Compute the rate contribution R for this field
            let rate: Array3<f64> = match ft {
                // ── Acoustic pressure ──────────────────────────────────────
                // R_p = c²·∇²p + Γ·μ_a·I  (photoacoustic source)
                //
                // Reference: Oraevsky & Karabutov (2003) "Optoacoustic tomography"
                // in Biomedical Photonics Handbook, CRC Press.
                UnifiedFieldType::Pressure => {
                    let c2 = coeff.sound_speed * coeff.sound_speed;
                    let mut r = laplacian_3d(&field_block, grid_dims, dx, dy, dz);
                    r.par_mapv_inplace(|v| v * c2);

                    // Photoacoustic source: p₀ = Γ · μₐ · I
                    // Grüneisen ≈ 0.12 for water at 37 °C (not 1.0).
                    if let Some(ref light_f) = light {
                        let gamma_mu_a = coeff.gruneisen * coeff.optical_absorption;
                        r.zip_mut_with(light_f, |r_val, &i_val| {
                            *r_val += gamma_mu_a * i_val;
                        });
                    }
                    r
                }

                // ── Optical fluence (diffusion approximation) ──────────────
                // R_I = D·∇²I − μ_a·I
                UnifiedFieldType::LightFluence => {
                    let d = coeff.optical_diffusion();
                    let mut r = laplacian_3d(&field_block, grid_dims, dx, dy, dz);
                    r.par_mapv_inplace(|v| v * d);
                    r.zip_mut_with(&field_block, |r_val, &i_val| {
                        *r_val -= coeff.optical_absorption * i_val;
                    });
                    r
                }

                // ── Temperature ────────────────────────────────────────────
                // R_T = κ·∇²T + Q_opt/(ρ·cₚ) + Q_ac/(ρ·cₚ)
                // Q_opt = μ_a · I    (optical absorption heating)
                // Q_ac  = α · p²/(ρ·c)  (acoustic absorption heating)
                UnifiedFieldType::Temperature => {
                    let kappa = coeff.thermal_diffusivity();
                    let inv_rho_cp = 1.0 / (coeff.density * coeff.specific_heat);
                    let mut r = laplacian_3d(&field_block, grid_dims, dx, dy, dz);
                    r.par_mapv_inplace(|v| v * kappa);

                    // Optical absorption heating
                    if let Some(ref light_f) = light {
                        r.zip_mut_with(light_f, |r_val, &i_val| {
                            *r_val += coeff.optical_absorption * i_val * inv_rho_cp;
                        });
                    }

                    // Acoustic absorption heating
                    if let Some(ref pres) = pressure {
                        let inv_rho_c = 1.0 / (coeff.density * coeff.sound_speed);
                        r.zip_mut_with(pres, |r_val, &p_val| {
                            let intensity = p_val * p_val * inv_rho_c;
                            *r_val += coeff.acoustic_absorption * intensity * inv_rho_cp;
                        });
                    }
                    r
                }

                // ── All other fields: no physics rate ──────────────────────
                _ => Array3::zeros((nx, ny, nz)),
            };

            // F_block = (u − u_prev) − dt·R   (already have u − u_prev in residual)
            let mut res_block = residual.slice_mut(s![row_start..row_start + nx, .., ..]);
            res_block.zip_mut_with(&rate, |f_val, &r_val| {
                *f_val -= dt * r_val;
            });
        }

        Ok(residual)
    }

    /// Jacobian-vector product: J·v ≈ [F(u+εv) − F(u)] / ε
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn jacobian_vector_product(
        &self,
        v: &Array3<f64>,
        u: &Array3<f64>,
        u_prev: &Array3<f64>,
        dt: f64,
        dims: (usize, usize, usize),
        field_order: &[UnifiedFieldType],
    ) -> KwaversResult<Array3<f64>> {
        // Finite difference approximation of directional derivative
        let eps = 1e-8 * (1.0 + super::utils::norm(u));
        let u_plus = &(u + &(v * eps));

        let f_u = self.compute_residual(u, u_prev, dt, dims, field_order)?;
        let f_u_plus = self.compute_residual(u_plus, u_prev, dt, dims, field_order)?;

        let jv = (&f_u_plus - &f_u) * (1.0 / eps);
        Ok(jv)
    }

    /// Line search: find step size α that reduces residual
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    #[allow(clippy::too_many_arguments)]
    pub(super) fn line_search(
        &self,
        u: &Array3<f64>,
        du: &Array3<f64>,
        f: &Array3<f64>,
        u_prev: &Array3<f64>,
        dt: f64,
        dims: (usize, usize, usize),
        field_order: &[UnifiedFieldType],
    ) -> KwaversResult<f64> {
        let f_norm = super::utils::norm(f);

        // Try decreasing step sizes: 1, 1/2, 1/4, 1/8, 1/16
        for k in 0i32..5 {
            let alpha = 2.0_f64.powi(-k);
            let u_new = &(u + &(du * alpha));
            let f_new = self.compute_residual(u_new, u_prev, dt, dims, field_order)?;
            let f_new_norm = super::utils::norm(&f_new);

            // Sufficient decrease criterion: ||F(u+α·du)|| < 0.9·||F(u)||
            if f_new_norm < 0.9 * f_norm {
                return Ok(alpha);
            }
        }

        // If no acceptable step found, use smallest tested
        Ok(2.0_f64.powi(-5))
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::NewtonKrylovConfig;
    use super::super::coupler::MonolithicCoupler;
    use super::super::utils::norm;
    use super::*;
    use crate::solver::integration::nonlinear::GMRESConfig;
    use ndarray::Array3;

    #[test]
    fn test_compute_residual_zero_fields() {
        // With all-zero fields, residual should be all-zero
        let coupler = MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
        let field_order = vec![UnifiedFieldType::Pressure, UnifiedFieldType::Temperature];
        let dims = (4, 3, 2);
        let n = field_order.len() * dims.0;
        let u = Array3::zeros((n, dims.1, dims.2));
        let u_prev = u.clone();

        let res = coupler
            .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
            .unwrap();
        let norm = norm(&res);
        assert!(
            norm < 1e-15,
            "Residual of zero state should be zero, got {norm}"
        );
    }

    /// Halving the Grüneisen parameter halves the photoacoustic source contribution
    /// in the Pressure block residual.
    ///
    /// The photoacoustic source is R_p += Γ · μₐ · I (Oraevsky & Karabutov 2003).
    /// With zero pressure (no Laplacian contribution), the entire residual at a
    /// lit voxel is Γ · μₐ · I, so halving Γ must halve the residual there.
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    #[test]
    fn test_photoacoustic_source_scales_with_gruneisen() {
        let make_coupler = |gruneisen: f64| {
            let mut c =
                MonolithicCoupler::new(NewtonKrylovConfig::default(), GMRESConfig::default());
            c.physics_coefficients.gruneisen = gruneisen;
            c.grid_spacing = (1e-3, 1e-3, 1e-3);
            c
        };

        let dims = (4, 4, 4);
        let nx = dims.0;
        // Stacked layout: row 0..nx = Pressure block, row nx..2*nx = LightFluence block
        // (field_order is sorted; Pressure < LightFluence alphabetically? Check enum order.)
        // Use explicit field_order matching enum discriminant order.
        let field_order = vec![UnifiedFieldType::Pressure, UnifiedFieldType::LightFluence];
        let n_blocks = field_order.len();

        // All-zero pressure block; unit fluence at interior node in LightFluence block
        let mut u = Array3::zeros((n_blocks * nx, dims.1, dims.2));
        // LightFluence block starts at row nx
        u[[nx + 1, 1, 1]] = 1.0; // interior fluence voxel
        let u_prev = u.clone();

        let c1 = make_coupler(0.12);
        let r1 = c1
            .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
            .unwrap();

        let c2 = make_coupler(0.06);
        let r2 = c2
            .compute_residual(&u, &u_prev, 1e-6, dims, &field_order)
            .unwrap();

        // Pressure block residual at (1,1,1): p=0, Laplacian=0, so R = Γ·μₐ·I
        let v1 = r1[[1, 1, 1]]; // Pressure block row 1
        let v2 = r2[[1, 1, 1]];
        assert!(
            v1.abs() > 1e-20,
            "Pressure residual at lit voxel must be non-zero, got {v1}"
        );
        let ratio = v1 / v2;
        assert!(
            (ratio - 2.0).abs() < 1e-10,
            "Residual ratio (γ=0.12)/(γ=0.06) must be exactly 2.0, got {ratio}"
        );
    }
}
