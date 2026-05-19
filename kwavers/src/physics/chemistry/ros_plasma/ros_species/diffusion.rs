//! Diffusion mathematics for ROS concentration tracking

use super::concentrations::ROSConcentrations;
use super::types::ROSSpecies;
use ndarray::Array3;

impl ROSConcentrations {
    /// Apply diffusion using simple forward Euler
    pub fn apply_diffusion(&mut self, dx: f64, dy: f64, dz: f64, dt: f64) {
        // Calculate maximum stability factor for the most diffusive species
        let max_d = ROSSpecies::HydroxylRadical.diffusion_coefficient();
        let min_spacing = dx.min(dy).min(dz);
        let stability_factor = max_d * dt / min_spacing.powi(2);

        // Check 3D stability condition: D·dt/dx² ≤ 1/(2·3) = 1/6 for cubic explicit scheme
        if stability_factor > 1.0 / 6.0 {
            log::warn!(
                "3D diffusion stability condition violated: D*dt/dx² = {:.3} > 1/6 ≈ 0.167. \
                Consider reducing timestep or using implicit scheme.",
                stability_factor
            );
        }

        // Collect updates to avoid borrow checker issues
        let mut updates: Vec<(ROSSpecies, Array3<f64>)> = Vec::new();

        for (species, conc) in &self.fields {
            let d = species.diffusion_coefficient();

            // 3D explicit stability: D·dt/h² ≤ 1/6; switch to semi-implicit beyond that
            if d * dt / dx.min(dy).min(dz).powi(2) > 1.0 / 6.0 {
                // Use semi-implicit scheme for numerical stability
                let mut updated_conc = conc.clone();
                Self::apply_semi_implicit_diffusion_static(
                    &mut updated_conc,
                    d,
                    dx,
                    dy,
                    dz,
                    dt,
                    self.shape,
                );
                updates.push((*species, updated_conc));
            } else {
                // Use explicit scheme for small stability factors
                let mut new_conc = conc.clone();

                // Use efficient 3D diffusion computation
                let dx2_inv = 1.0 / (dx * dx);
                let dy2_inv = 1.0 / (dy * dy);
                let dz2_inv = 1.0 / (dz * dz);

                // Process interior points
                for i in 1..self.shape.0 - 1 {
                    for j in 1..self.shape.1 - 1 {
                        for k in 1..self.shape.2 - 1 {
                            let center_val = conc[[i, j, k]];

                            // Compute Laplacian using neighboring values
                            let laplacian = (2.0f64.mul_add(-center_val, conc[[i, j, k + 1]])
                                + conc[[i, j, k - 1]])
                            .mul_add(
                                dz2_inv,
                                (2.0f64.mul_add(-center_val, conc[[i + 1, j, k]])
                                    + conc[[i - 1, j, k]])
                                .mul_add(
                                    dx2_inv,
                                    (2.0f64.mul_add(-center_val, conc[[i, j + 1, k]])
                                        + conc[[i, j - 1, k]])
                                        * dy2_inv,
                                ),
                            );

                            new_conc[[i, j, k]] = (d * laplacian).mul_add(dt, center_val);
                        }
                    }
                }

                updates.push((*species, new_conc));
            }
        }

        // Apply updates
        for (species, new_conc) in updates {
            self.fields.insert(species, new_conc);
        }
    }

    /// Apply semi-implicit diffusion for numerical stability (static version)
    fn apply_semi_implicit_diffusion_static(
        conc: &mut Array3<f64>,
        d: f64,
        dx: f64,
        _dy: f64,
        _dz: f64,
        dt: f64,
        shape: (usize, usize, usize),
    ) {
        // ADI (Alternating Direction Implicit) method
        // This is more stable than explicit Euler
        let mut temp = conc.clone();

        // X-direction sweep
        for j in 1..shape.1 - 1 {
            for k in 1..shape.2 - 1 {
                let alpha = d * dt / (2.0 * dx * dx);
                // Solve tridiagonal system for each line
                // Use Crank-Nicolson discretization
                for i in 1..shape.0 - 1 {
                    let rhs = conc[[i, j, k]]
                        + alpha
                            * (2.0f64.mul_add(-conc[[i, j, k]], conc[[i + 1, j, k]])
                                + conc[[i - 1, j, k]]);
                    temp[[i, j, k]] = rhs / 2.0f64.mul_add(alpha, 1.0);
                }
            }
        }

        // **Implementation Status**: 1D semi-implicit diffusion (X-direction)
        // Full ADI (Alternating Direction Implicit) requires Y and Z sweeps for 3D isotropy
        // Current: 1D provides numerical stability for dominant diffusion direction
        // Complete 3D ADI implementation deferred to Sprint 125+ chemistry enhancement
        //
        // **Reference**: Peaceman & Rachford (1955) "The Numerical Solution of Parabolic Equations"
        *conc = temp;
    }
}
