    fn update_velocity_staggered(&mut self, dt: f64) -> KwaversResult<()> {
        let (nx, ny, nz) = self.fields.p.dim();

        // 1. Compute gradients
        let mut dp_dx = if nx > 1 {
            Some(self.staggered_operator.apply_forward_x(self.fields.p.view())?)
        } else {
            None
        };

        let mut dp_dy = if ny > 1 {
            Some(self.staggered_operator.apply_forward_y(self.fields.p.view())?)
        } else {
            None
        };

        let mut dp_dz = if nz > 1 {
            Some(self.staggered_operator.apply_forward_z(self.fields.p.view())?)
        } else {
            None
        };

        // 2. Apply CPML corrections
        if let Some(ref mut cpml) = self.cpml_boundary {
            if let Some(ref mut grad) = dp_dx {
                cpml.update_and_apply_p_gradient_correction(grad, 0);
            }
            if let Some(ref mut grad) = dp_dy {
                cpml.update_and_apply_p_gradient_correction(grad, 1);
            }
            if let Some(ref mut grad) = dp_dz {
                cpml.update_and_apply_p_gradient_correction(grad, 2);
            }
        }

        // 3. Update velocity components
        if let Some(grad) = dp_dx {
            for i in 0..nx - 1 {
                for j in 0..ny {
                    for k in 0..nz {
                        let rho = 0.5 * (self.materials.rho0[[i, j, k]] + self.materials.rho0[[i + 1, j, k]]);
                        if rho > 1e-9 {
                            self.fields.ux[[i, j, k]] -= dt / rho * grad[[i, j, k]];
                        }
                    }
                }
            }
            self.fields.ux.slice_mut(s![nx - 1, .., ..]).fill(0.0);
        }

        if let Some(grad) = dp_dy {
            for i in 0..nx {
                for j in 0..ny - 1 {
                    for k in 0..nz {
                        let rho = 0.5 * (self.materials.rho0[[i, j, k]] + self.materials.rho0[[i, j + 1, k]]);
                        if rho > 1e-9 {
                            self.fields.uy[[i, j, k]] -= dt / rho * grad[[i, j, k]];
                        }
                    }
                }
            }
            self.fields.uy.slice_mut(s![.., ny - 1, ..]).fill(0.0);
        }

        if let Some(grad) = dp_dz {
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz - 1 {
                        let rho = 0.5 * (self.materials.rho0[[i, j, k]] + self.materials.rho0[[i, j, k + 1]]);
                        if rho > 1e-9 {
                            self.fields.uz[[i, j, k]] -= dt / rho * grad[[i, j, k]];
                        }
                    }
                }
            }
            self.fields.uz.slice_mut(s![.., .., nz - 1]).fill(0.0);
        }

        Ok(())
    }
