//! Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations
//!
//! This module implements the standard Yee algorithm for solving Maxwell's equations
//! on a staggered grid.

use crate::domain::field::EMFields;
use crate::domain::grid::Grid;
use crate::physics::electromagnetic::equations::{
    EMDimension, EMMaterialDistribution, ElectromagneticWaveEquation,
};
use ndarray::{ArrayD, IxDyn};

/// FDTD Solver for Maxwell's Equations
pub struct FDTD {
    /// Computational grid
    pub grid: Grid,
    /// Material properties distribution
    pub materials: EMMaterialDistribution,
    /// Electromagnetic fields
    pub fields: EMFields,
    /// Current simulation time (s)
    pub time: f64,
    /// Time step (s)
    pub dt: f64,
}

impl FDTD {
    /// Create a new FDTD solver
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `materials` - Material properties distribution (must match grid dimensions)
    /// * `dt` - Time step
    pub fn new(
        grid: Grid,
        materials: EMMaterialDistribution,
        dt: f64,
    ) -> Result<Self, String> {
        // Validate material shape matches grid
        let mat_shape = materials.shape();
        let grid_shape = [grid.nx, grid.ny, grid.nz];

        if mat_shape != grid_shape {
            return Err(format!(
                "Material shape {:?} does not match grid shape {:?}",
                mat_shape, grid_shape
            ));
        }

        // Initialize fields (E and H)
        // Shape is [nx, ny, nz, 3] for 3D vector fields
        let field_shape = IxDyn(&[grid.nx, grid.ny, grid.nz, 3]);
        let electric = ArrayD::zeros(field_shape.clone());
        let magnetic = ArrayD::zeros(field_shape);
        let fields = EMFields::new(electric, magnetic);

        Ok(Self {
            grid,
            materials,
            fields,
            time: 0.0,
            dt,
        })
    }
}

impl ElectromagneticWaveEquation for FDTD {
    fn em_dimension(&self) -> EMDimension {
        match self.grid.dimensionality {
            1 => EMDimension::One,
            2 => EMDimension::Two,
            3 => EMDimension::Three,
            _ => EMDimension::Three, // Default to 3D
        }
    }

    fn material_properties(&self) -> &EMMaterialDistribution {
        &self.materials
    }

    fn em_fields(&self) -> &EMFields {
        &self.fields
    }

    fn step_maxwell(&mut self, dt: f64) -> Result<(), String> {
        self.dt = dt;

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let dx = self.grid.dx;
        let dy = self.grid.dy;
        let dz = self.grid.dz;

        let eps0 = 8.854e-12;
        let mu0 = 4.0 * std::f64::consts::PI * 1e-7;

        // Update H field
        // H^{n+1/2} = H^{n-1/2} - (dt/mu) * curl E^n
        for i in 0..nx - 1 {
            for j in 0..ny - 1 {
                for k in 0..nz - 1 {
                    let mu_r = self.materials.permeability[[i, j, k]];
                    let mu = mu_r * mu0;
                    let coeff = dt / mu;

                    // (curl E)_x = dEz/dy - dEy/dz
                    let d_ez_dy = (self.fields.electric[[i, j + 1, k, 2]]
                        - self.fields.electric[[i, j, k, 2]])
                        / dy;
                    let d_ey_dz = (self.fields.electric[[i, j, k + 1, 1]]
                        - self.fields.electric[[i, j, k, 1]])
                        / dz;
                    self.fields.magnetic[[i, j, k, 0]] -= coeff * (d_ez_dy - d_ey_dz);

                    // (curl E)_y = dEx/dz - dEz/dx
                    let d_ex_dz = (self.fields.electric[[i, j, k + 1, 0]]
                        - self.fields.electric[[i, j, k, 0]])
                        / dz;
                    let d_ez_dx = (self.fields.electric[[i + 1, j, k, 2]]
                        - self.fields.electric[[i, j, k, 2]])
                        / dx;
                    self.fields.magnetic[[i, j, k, 1]] -= coeff * (d_ex_dz - d_ez_dx);

                    // (curl E)_z = dEy/dx - dEx/dy
                    let d_ey_dx = (self.fields.electric[[i + 1, j, k, 1]]
                        - self.fields.electric[[i, j, k, 1]])
                        / dx;
                    let d_ex_dy = (self.fields.electric[[i, j + 1, k, 0]]
                        - self.fields.electric[[i, j, k, 0]])
                        / dy;
                    self.fields.magnetic[[i, j, k, 2]] -= coeff * (d_ey_dx - d_ex_dy);
                }
            }
        }

        // Update E field
        // E^{n+1} = E^n + (dt/eps) * curl H^{n+1/2} - (dt*sigma/eps) * E^n
        for i in 1..nx {
            for j in 1..ny {
                for k in 1..nz {
                    let eps_r = self.materials.permittivity[[i, j, k]];
                    let eps = eps_r * eps0;
                    let sigma = self.materials.conductivity[[i, j, k]];

                    let factor = sigma * dt / (2.0 * eps);
                    let c1 = (1.0 - factor) / (1.0 + factor);
                    let c2 = (dt / eps) / (1.0 + factor);

                    // (curl H)_x = dHz/dy - dHy/dz
                    let d_hz_dy = (self.fields.magnetic[[i, j, k, 2]]
                        - self.fields.magnetic[[i, j - 1, k, 2]])
                        / dy;
                    let d_hy_dz = (self.fields.magnetic[[i, j, k, 1]]
                        - self.fields.magnetic[[i, j, k - 1, 1]])
                        / dz;
                    self.fields.electric[[i, j, k, 0]] =
                        c1 * self.fields.electric[[i, j, k, 0]] + c2 * (d_hz_dy - d_hy_dz);

                    // (curl H)_y = dHx/dz - dHz/dx
                    let d_hx_dz = (self.fields.magnetic[[i, j, k, 0]]
                        - self.fields.magnetic[[i, j, k - 1, 0]])
                        / dz;
                    let d_hz_dx = (self.fields.magnetic[[i, j, k, 2]]
                        - self.fields.magnetic[[i - 1, j, k, 2]])
                        / dx;
                    self.fields.electric[[i, j, k, 1]] =
                        c1 * self.fields.electric[[i, j, k, 1]] + c2 * (d_hx_dz - d_hz_dx);

                    // (curl H)_z = dHy/dx - dHx/dy
                    let d_hy_dx = (self.fields.magnetic[[i, j, k, 1]]
                        - self.fields.magnetic[[i - 1, j, k, 1]])
                        / dx;
                    let d_hx_dy = (self.fields.magnetic[[i, j, k, 0]]
                        - self.fields.magnetic[[i, j - 1, k, 0]])
                        / dy;
                    self.fields.electric[[i, j, k, 2]] =
                        c1 * self.fields.electric[[i, j, k, 2]] + c2 * (d_hy_dx - d_hx_dy);
                }
            }
        }

        self.time += dt;
        Ok(())
    }

    fn apply_em_boundary_conditions(&mut self, _fields: &mut EMFields) {
        // Minimal implementation: fields at boundaries are not updated by the loop,
        // so they effectively stay at their initial value (zero) -> PEC boundary condition.
        // For a more complete solver, we would implement ABCs or PML here.
    }

    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String> {
        // Check for NaN or Infinite values
        if fields.electric.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err("Electric field contains NaN or Inf".to_string());
        }
        if fields.magnetic.iter().any(|&x| x.is_nan() || x.is_infinite()) {
            return Err("Magnetic field contains NaN or Inf".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;

    #[test]
    fn test_fdtd_step() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let materials = EMMaterialDistribution::vacuum(&[10, 10, 10]);
        let mut fdtd = FDTD::new(grid, materials, 1e-12).unwrap();

        // Inject a source (manually for test)
        // E_z at center = 1.0
        fdtd.fields.electric[[5, 5, 5, 2]] = 1.0;

        // Step
        fdtd.step_maxwell(1e-12).unwrap();

        // Check that fields have evolved (H should be non-zero near source)
        // H is updated first using E, so neighbors of E should have H.
        // E at (5,5,5) affects H around it.
        // Hx depends on dEz/dy. So Hx at (5,4,5) and (5,5,5) might be affected.
        // Hy depends on -dEz/dx.

        // Just check that some magnetic field is non-zero
        let max_h = fdtd.fields.magnetic.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(max_h > 0.0, "Magnetic field should be generated by curl E");

        // Step again
        fdtd.step_maxwell(1e-12).unwrap();

        // Check that E has evolved (curl H affects E)
        // E should propagate
    }
}
