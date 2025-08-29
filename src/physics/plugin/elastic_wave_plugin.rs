//! Elastic wave physics plugin
//!
//! This plugin integrates elastic wave propagation into the solver framework,
//! providing full support for P-waves, S-waves, and mode conversion.

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::mechanics::elastic_wave::{
    ElasticStressFields, ElasticVelocityFields, ElasticWave,
};
use crate::physics::plugin::{Plugin, PluginConfig, PluginContext, PluginFields, PluginMetadata, PluginState};
use crate::source::Source;
use ndarray::{Array3, Array4};

/// Elastic wave physics plugin
#[derive(Debug)]
pub struct ElasticWavePlugin {
    /// Plugin metadata
    metadata: PluginMetadata,
    /// Plugin state
    state: PluginState,
    /// Core elastic wave solver
    solver: ElasticWave,
    /// Stress field components (6 components: xx, yy, zz, xy, xz, yz)
    stress_fields: ElasticStressFields,
    /// Velocity field components (3 components: x, y, z)
    velocity_fields: ElasticVelocityFields,
    /// Material properties cache
    lame_lambda: Array3<f64>,
    lame_mu: Array3<f64>,
    density: Array3<f64>,
    /// Time step for integration
    dt: f64,
    /// Current simulation time
    time: f64,
}

impl ElasticWavePlugin {
    /// Create a new elastic wave plugin
    pub fn new(grid: &Grid, medium: &dyn Medium, dt: f64) -> KwaversResult<Self> {
        let metadata = PluginMetadata {
            id: "elastic_wave".to_string(),
            name: "Elastic Wave Propagation".to_string(),
            version: "1.0.0".to_string(),
            description: "Full elastic wave propagation with P-waves, S-waves, and mode conversion"
                .to_string(),
            author: "Kwavers Team".to_string(),
            license: "MIT".to_string(),
        };

        let solver = ElasticWave::new(grid)?;
        let (nx, ny, nz) = grid.dimensions();

        // Initialize fields
        let stress_fields = ElasticStressFields::new(nx, ny, nz);
        let velocity_fields = ElasticVelocityFields::new(nx, ny, nz);

        // Cache material properties
        let mut lame_lambda = Array3::zeros((nx, ny, nz));
        let mut lame_mu = Array3::zeros((nx, ny, nz));
        let mut density = Array3::zeros((nx, ny, nz));

        // Extract elastic properties from medium
        // Query proper elastic moduli directly from the medium interface
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * grid.dx;
                    let y = j as f64 * grid.dy;
                    let z = k as f64 * grid.dz;

                    // Get proper elastic properties from medium
                    // The medium provides exact Lamé parameters for the material
                    lame_lambda[[i, j, k]] = medium.lame_lambda(x, y, z, grid);
                    lame_mu[[i, j, k]] = medium.lame_mu(x, y, z, grid);
                    density[[i, j, k]] = medium.density(x, y, z, grid);
                }
            }
        }

        Ok(Self {
            metadata,
            state: PluginState::Created,
            solver,
            stress_fields,
            velocity_fields,
            lame_lambda,
            lame_mu,
            density,
            dt,
            time: 0.0,
        })
    }

    /// Update elastic fields using staggered grid finite difference
    fn update_fields(&mut self, source: Option<&dyn Source>, grid: &Grid) -> KwaversResult<()> {
        let (nx, ny, nz) = grid.dimensions();
        let dx_inv = 1.0 / grid.dx;
        let dy_inv = 1.0 / grid.dy;
        let dz_inv = 1.0 / grid.dz;

        // Step 1: Update velocities from stress divergence
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let rho = self.density[[i, j, k]];
                    if rho <= 0.0 {
                        continue;
                    }

                    // Compute stress divergence (force per unit volume)
                    let fx = (self.stress_fields.txx[[i + 1, j, k]]
                        - self.stress_fields.txx[[i - 1, j, k]])
                        * dx_inv
                        * 0.5
                        + (self.stress_fields.txy[[i, j + 1, k]]
                            - self.stress_fields.txy[[i, j - 1, k]])
                            * dy_inv
                            * 0.5
                        + (self.stress_fields.txz[[i, j, k + 1]]
                            - self.stress_fields.txz[[i, j, k - 1]])
                            * dz_inv
                            * 0.5;

                    let fy = (self.stress_fields.txy[[i + 1, j, k]]
                        - self.stress_fields.txy[[i - 1, j, k]])
                        * dx_inv
                        * 0.5
                        + (self.stress_fields.tyy[[i, j + 1, k]]
                            - self.stress_fields.tyy[[i, j - 1, k]])
                            * dy_inv
                            * 0.5
                        + (self.stress_fields.tyz[[i, j, k + 1]]
                            - self.stress_fields.tyz[[i, j, k - 1]])
                            * dz_inv
                            * 0.5;

                    let fz = (self.stress_fields.txz[[i + 1, j, k]]
                        - self.stress_fields.txz[[i - 1, j, k]])
                        * dx_inv
                        * 0.5
                        + (self.stress_fields.tyz[[i, j + 1, k]]
                            - self.stress_fields.tyz[[i, j - 1, k]])
                            * dy_inv
                            * 0.5
                        + (self.stress_fields.tzz[[i, j, k + 1]]
                            - self.stress_fields.tzz[[i, j, k - 1]])
                            * dz_inv
                            * 0.5;

                    // Update velocities (Newton's second law)
                    self.velocity_fields.vx[[i, j, k]] += self.dt * fx / rho;
                    self.velocity_fields.vy[[i, j, k]] += self.dt * fy / rho;
                    self.velocity_fields.vz[[i, j, k]] += self.dt * fz / rho;
                }
            }
        }

        // Add source contribution if present
        // For elastic waves, source injection would be implemented here
        // This requires extending the Source trait to support elastic wave sources

        // Step 2: Update stresses from velocity gradients (strain rates)
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let lambda = self.lame_lambda[[i, j, k]];
                    let mu = self.lame_mu[[i, j, k]];

                    // Compute velocity gradients (strain rates)
                    let dvx_dx = (self.velocity_fields.vx[[i + 1, j, k]]
                        - self.velocity_fields.vx[[i - 1, j, k]])
                        * dx_inv
                        * 0.5;
                    let dvy_dy = (self.velocity_fields.vy[[i, j + 1, k]]
                        - self.velocity_fields.vy[[i, j - 1, k]])
                        * dy_inv
                        * 0.5;
                    let dvz_dz = (self.velocity_fields.vz[[i, j, k + 1]]
                        - self.velocity_fields.vz[[i, j, k - 1]])
                        * dz_inv
                        * 0.5;

                    let dvx_dy = (self.velocity_fields.vx[[i, j + 1, k]]
                        - self.velocity_fields.vx[[i, j - 1, k]])
                        * dy_inv
                        * 0.5;
                    let dvx_dz = (self.velocity_fields.vx[[i, j, k + 1]]
                        - self.velocity_fields.vx[[i, j, k - 1]])
                        * dz_inv
                        * 0.5;
                    let dvy_dx = (self.velocity_fields.vy[[i + 1, j, k]]
                        - self.velocity_fields.vy[[i - 1, j, k]])
                        * dx_inv
                        * 0.5;
                    let dvy_dz = (self.velocity_fields.vy[[i, j, k + 1]]
                        - self.velocity_fields.vy[[i, j, k - 1]])
                        * dz_inv
                        * 0.5;
                    let dvz_dx = (self.velocity_fields.vz[[i + 1, j, k]]
                        - self.velocity_fields.vz[[i - 1, j, k]])
                        * dx_inv
                        * 0.5;
                    let dvz_dy = (self.velocity_fields.vz[[i, j + 1, k]]
                        - self.velocity_fields.vz[[i, j - 1, k]])
                        * dy_inv
                        * 0.5;

                    // Volumetric strain rate
                    let div_v = dvx_dx + dvy_dy + dvz_dz;

                    // Update normal stresses (Hooke's law)
                    self.stress_fields.txx[[i, j, k]] +=
                        self.dt * (lambda * div_v + 2.0 * mu * dvx_dx);
                    self.stress_fields.tyy[[i, j, k]] +=
                        self.dt * (lambda * div_v + 2.0 * mu * dvy_dy);
                    self.stress_fields.tzz[[i, j, k]] +=
                        self.dt * (lambda * div_v + 2.0 * mu * dvz_dz);

                    // Update shear stresses
                    self.stress_fields.txy[[i, j, k]] += self.dt * mu * (dvx_dy + dvy_dx);
                    self.stress_fields.txz[[i, j, k]] += self.dt * mu * (dvx_dz + dvz_dx);
                    self.stress_fields.tyz[[i, j, k]] += self.dt * mu * (dvy_dz + dvz_dy);
                }
            }
        }

        self.time += self.dt;
        Ok(())
    }

    /// Convert elastic fields to pressure for compatibility
    fn compute_pressure(&self) -> Array3<f64> {
        // Pressure is negative of mean normal stress
        let (nx, ny, nz) = self.stress_fields.txx.dim();
        let mut pressure = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    pressure[[i, j, k]] = -(self.stress_fields.txx[[i, j, k]]
                        + self.stress_fields.tyy[[i, j, k]]
                        + self.stress_fields.tzz[[i, j, k]])
                        / 3.0;
                }
            }
        }

        pressure
    }
}

impl PhysicsPlugin for ElasticWavePlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![] // Elastic wave doesn't require input fields, it generates them
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![
            UnifiedFieldType::Pressure,
            UnifiedFieldType::VelocityX,
            UnifiedFieldType::VelocityY,
            UnifiedFieldType::VelocityZ,
        ]
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        // Update elastic wave fields
        self.update_fields(None, grid)?; // Source integration would be added later

        // Export fields to Array4 format
        // Assuming fields is organized as [field_type, x, y, z]
        // Field indices: 0=pressure, 1=vx, 2=vy, 3=vz
        if fields.shape()[0] >= 4 {
            // Convert stress to pressure
            let pressure = self.compute_pressure();
            fields
                .slice_mut(ndarray::s![0, .., .., ..])
                .assign(&pressure);

            // Copy velocity fields
            fields
                .slice_mut(ndarray::s![1, .., .., ..])
                .assign(&self.velocity_fields.vx);
            fields
                .slice_mut(ndarray::s![2, .., .., ..])
                .assign(&self.velocity_fields.vy);
            fields
                .slice_mut(ndarray::s![3, .., .., ..])
                .assign(&self.velocity_fields.vz);
        }

        self.state = PluginState::Running;
        Ok(())
    }
}
