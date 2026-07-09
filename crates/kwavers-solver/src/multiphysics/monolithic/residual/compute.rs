use super::super::coupler::MonolithicCoupler;
use super::super::spatial_operator::laplacian_3d_into;
use super::super::state_vector::field_block_view;
use crate::workspace::inplace_ops::scale_inplace;
use kwavers_core::error::KwaversResult;
use kwavers_field::UnifiedFieldType;
use leto::Array3;

impl MonolithicCoupler {
    /// Compute residual F(u) = u − u_prev − Δt·R(u).
    ///
    /// Evaluates the implicit residual for the coupled acoustic-optical-thermal
    /// system. Each physics field occupies a contiguous block of `nx` rows in
    /// the flattened `Array3<f64>` with total shape
    /// `(n_fields * nx, ny, nz)`.
    ///
    /// **Rate terms R(u) by field type:**
    ///
    /// | Field | R(u) |
    /// |-------|------|
    /// | Pressure | c²·∇²p + Γ·μ_a·I |
    /// | LightFluence | D·∇²I − μ_a·I |
    /// | Temperature | κ·∇²T + μ_a·I/(ρ·c_p) + α_ac·p²/(ρ·c·ρ·c_p) |
    /// | Other | 0 |
    ///
    /// The assembly borrows field blocks from the stacked state and reuses one
    /// rate scratch buffer across rate-producing blocks. The flattened state
    /// remains the single source of truth during Newton residual evaluation.
    ///
    /// # Errors
    /// - Propagates validation errors from lower-level field/rate kernels.
    pub(in crate::multiphysics::monolithic) fn compute_residual(
        &self,
        u: &Array3<f64>,
        u_prev: &Array3<f64>,
        dt: f64,
        grid_dims: (usize, usize, usize),
        field_order: &[UnifiedFieldType],
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = grid_dims;
        let (dx, dy, dz) = self.grid_spacing;

        let mut residual = u - u_prev;

        let idx_of =
            |ft: UnifiedFieldType| -> Option<usize> { field_order.iter().position(|&k| k == ft) };

        let pressure = idx_of(UnifiedFieldType::Pressure).map(|i| field_block_view(u, nx, i));
        let light = idx_of(UnifiedFieldType::LightFluence).map(|i| field_block_view(u, nx, i));

        let coeff = &self.physics_coefficients;
        let mut rate = Array3::zeros((nx, ny, nz));
        for (block, &ft) in field_order.iter().enumerate() {
            let row_start = block * nx;
            let field_block = field_block_view(u, nx, block);

            match ft {
                UnifiedFieldType::Pressure => {
                    let c2 = coeff.sound_speed * coeff.sound_speed;
                    laplacian_3d_into(&field_block, grid_dims, dx, dy, dz, &mut rate);
                    scale_inplace(&mut rate, c2);

                    if let Some(light_f) = light.as_ref() {
                        let gamma_mu_a = coeff.gruneisen * coeff.optical_absorption;
                        for (r_val, i_val) in rate.iter_mut().zip(light_f.iter()) {
            {
                            *r_val += gamma_mu_a * i_val;
                        };
        };
                    }
                }
                UnifiedFieldType::LightFluence => {
                    let d = coeff.optical_diffusion();
                    laplacian_3d_into(&field_block, grid_dims, dx, dy, dz, &mut rate);
                    scale_inplace(&mut rate, d);
                    for (r_val, i_val) in rate.iter_mut().zip(&field_block.iter()) {
            {
                        *r_val -= coeff.optical_absorption * i_val;
                    };
        };
                }
                UnifiedFieldType::Temperature => {
                    let kappa = coeff.thermal_diffusivity();
                    let inv_rho_cp = 1.0 / (coeff.density * coeff.specific_heat);
                    laplacian_3d_into(&field_block, grid_dims, dx, dy, dz, &mut rate);
                    scale_inplace(&mut rate, kappa);

                    if let Some(light_f) = light.as_ref() {
                        for (r_val, i_val) in rate.iter_mut().zip(light_f.iter()) {
            {
                            *r_val += coeff.optical_absorption * i_val * inv_rho_cp;
                        };
        };
                    }

                    if let Some(pres) = pressure.as_ref() {
                        let inv_rho_c = 1.0 / (coeff.density * coeff.sound_speed);
                        for (r_val, p_val) in rate.iter_mut().zip(pres.iter()) {
            {
                            let intensity = p_val * p_val * inv_rho_c;
                            *r_val += coeff.acoustic_absorption * intensity * inv_rho_cp;
                        };
        };
                    }
                }
                _ => continue,
            };

            let mut res_block = residual.slice_mut(s![row_start..row_start + nx, .., ..]);
            for (f_val, r_val) in res_block.iter_mut().zip(&rate.iter()) {
            {
                *f_val -= dt * r_val;
            };
        };
        }

        Ok(residual)
    }
}
