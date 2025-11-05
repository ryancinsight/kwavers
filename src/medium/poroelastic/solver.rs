//! Poroelastic Wave Solver
//!
//! Implements finite difference time domain (FDTD) solver for poroelastic wave equations
//! based on Biot's theory of poroelasticity.

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::{Array3, Array4, s, Axis};
use std::f64::consts::PI;

/// Configuration for poroelastic solver
#[derive(Debug, Clone)]
pub struct PoroelasticConfig {
    /// Simulation time step (s)
    pub dt: f64,
    /// Total simulation time (s)
    pub simulation_time: f64,
    /// Perfectly matched layer thickness
    pub pml_thickness: usize,
    /// Save data every N time steps
    pub save_every: usize,
    /// Use GPU acceleration
    pub gpu_accelerated: bool,
    /// Enable dispersion correction
    pub dispersion_correction: bool,
}

impl Default for PoroelasticConfig {
    fn default() -> Self {
        Self {
            dt: 1e-7,        // 0.1 μs
            simulation_time: 1e-4, // 100 μs
            pml_thickness: 10,
            save_every: 10,
            gpu_accelerated: false,
            dispersion_correction: true,
        }
    }
}

/// Poroelastic wave solver
pub struct PoroelasticSolver {
    /// Computational grid
    grid: Grid,
    /// Solver configuration
    config: PoroelasticConfig,
    /// Solid displacement fields (ux, uy, uz)
    solid_displacement: Array4<f32>,
    /// Fluid displacement fields (wx, wy, wz)
    fluid_displacement: Array4<f32>,
    /// Solid stress fields (σxx, σyy, σzz, σxy, σxz, σyz)
    solid_stress: Array4<f32>,
    /// Fluid pressure field
    fluid_pressure: Array3<f32>,
    /// Previous time step fields for time integration
    solid_displacement_prev: Array4<f32>,
    fluid_displacement_prev: Array4<f32>,
    /// Perfectly matched layer
    pml: Option<PMLLayer>,
}

impl PoroelasticSolver {
    /// Create new poroelastic solver
    pub fn new(
        grid: &Grid,
        config: PoroelasticConfig,
        properties: &crate::medium::poroelastic::PoroelasticProperties,
    ) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();

        // Initialize field arrays
        let solid_displacement = Array4::<f32>::zeros((3, nx, ny, nz)); // 3 components
        let fluid_displacement = Array4::<f32>::zeros((3, nx, ny, nz));
        let solid_stress = Array4::<f32>::zeros((6, nx, ny, nz)); // 6 components
        let fluid_pressure = Array3::<f32>::zeros((nx, ny, nz));

        let solid_displacement_prev = solid_displacement.clone();
        let fluid_displacement_prev = fluid_displacement.clone();

        // Create PML if requested
        let pml = if config.pml_thickness > 0 {
            Some(PMLLayer::new(grid, config.pml_thickness, properties)?)
        } else {
            None
        };

        Ok(Self {
            grid: grid.clone(),
            config,
            solid_displacement,
            fluid_displacement,
            solid_stress,
            fluid_pressure,
            solid_displacement_prev,
            fluid_displacement_prev,
            pml,
        })
    }

    /// Run poroelastic simulation
    pub fn simulate(&mut self) -> KwaversResult<PoroelasticSimulationResult> {
        println!("Starting poroelastic simulation...");
        println!("Grid: {}x{}x{}", self.grid.nx, self.grid.ny, self.grid.nz);
        println!("Time step: {:.2e} s, Total time: {:.2e} s",
                 self.config.dt, self.config.simulation_time);

        let n_steps = (self.config.simulation_time / self.config.dt) as usize;
        let mut saved_fields = Vec::new();

        // Main time-stepping loop
        for step in 0..n_steps {
            // Update fields
            self.time_step()?;

            // Save data periodically
            if step % self.config.save_every == 0 {
                let fields = PoroelasticFields {
                    time: step as f64 * self.config.dt,
                    solid_displacement: self.solid_displacement.clone(),
                    fluid_displacement: self.fluid_displacement.clone(),
                    solid_stress: self.solid_stress.clone(),
                    fluid_pressure: self.fluid_pressure.clone(),
                };
                saved_fields.push(fields);

                println!("Step {}/{}: t = {:.2e} s", step, n_steps, fields.time);
            }

            // Update previous time step fields
            self.solid_displacement_prev.assign(&self.solid_displacement);
            self.fluid_displacement_prev.assign(&self.fluid_displacement);
        }

        println!("Simulation completed successfully");

        Ok(PoroelasticSimulationResult {
            config: self.config.clone(),
            saved_fields,
            final_energy: self.compute_total_energy(),
            convergence_metric: self.compute_convergence_metric(),
        })
    }

    /// Single time step of poroelastic simulation
    fn time_step(&mut self) -> KwaversResult<()> {
        // 1. Update solid stress from solid displacement
        self.update_solid_stress()?;

        // 2. Update fluid pressure from fluid displacement
        self.update_fluid_pressure()?;

        // 3. Update solid displacement from stresses
        self.update_solid_displacement()?;

        // 4. Update fluid displacement from pressure and coupling
        self.update_fluid_displacement()?;

        // 5. Apply boundary conditions (PML)
        if let Some(pml) = &mut self.pml {
            pml.apply(&mut self.solid_displacement, &mut self.fluid_displacement)?;
        }

        Ok(())
    }

    /// Update solid stress tensor
    fn update_solid_stress(&mut self) -> KwaversResult<()> {
        let (nx, ny, nz) = self.grid.dimensions();

        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Compute strain components
                    let eps_xx = (self.solid_displacement[[0, i+1, j, k]] - self.solid_displacement[[0, i-1, j, k]]) / (2.0 * self.grid.dx as f32);
                    let eps_yy = (self.solid_displacement[[1, i, j+1, k]] - self.solid_displacement[[1, i, j-1, k]]) / (2.0 * self.grid.dy as f32);
                    let eps_zz = (self.solid_displacement[[2, i, j, k+1]] - self.solid_displacement[[2, i, j, k-1]]) / (2.0 * self.grid.dz as f32);

                    let eps_xy = 0.5 * (
                        (self.solid_displacement[[0, i, j+1, k]] - self.solid_displacement[[0, i, j-1, k]]) / (2.0 * self.grid.dy as f32) +
                        (self.solid_displacement[[1, i+1, j, k]] - self.solid_displacement[[1, i-1, j, k]]) / (2.0 * self.grid.dx as f32)
                    );

                    // Simplified isotropic solid stress (would need material properties)
                    let lambda = 1e9; // Lame's first parameter
                    let mu = 5e8;     // Shear modulus

                    self.solid_stress[[0, i, j, k]] = (lambda * (eps_xx + eps_yy + eps_zz) + 2.0 * mu * eps_xx) as f32; // σ_xx
                    self.solid_stress[[1, i, j, k]] = (lambda * (eps_xx + eps_yy + eps_zz) + 2.0 * mu * eps_yy) as f32; // σ_yy
                    self.solid_stress[[2, i, j, k]] = (lambda * (eps_xx + eps_yy + eps_zz) + 2.0 * mu * eps_zz) as f32; // σ_zz
                    self.solid_stress[[3, i, j, k]] = (mu * eps_xy) as f32; // σ_xy
                }
            }
        }

        Ok(())
    }

    /// Update fluid pressure
    fn update_fluid_pressure(&mut self) -> KwaversResult<()> {
        let (nx, ny, nz) = self.grid.dimensions();

        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Compute fluid dilatation
                    let div_w = (
                        (self.fluid_displacement[[0, i+1, j, k]] - self.fluid_displacement[[0, i-1, j, k]]) / (2.0 * self.grid.dx as f32) +
                        (self.fluid_displacement[[1, i, j+1, k]] - self.fluid_displacement[[1, i, j-1, k]]) / (2.0 * self.grid.dy as f32) +
                        (self.fluid_displacement[[2, i, j, k+1]] - self.fluid_displacement[[2, i, j, k-1]]) / (2.0 * self.grid.dz as f32)
                    );

                    // Simplified fluid pressure (would need material properties)
                    let m = 2e10; // Biot modulus
                    let alpha = 0.9; // Biot coefficient

                    self.fluid_pressure[[i, j, k]] = (m * alpha * div_w) as f32;
                }
            }
        }

        Ok(())
    }

    /// Update solid displacement
    fn update_solid_displacement(&mut self) -> KwaversResult<()> {
        let (nx, ny, nz) = self.grid.dimensions();
        let dt = self.config.dt as f32;

        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Solid momentum equation
                    for comp in 0..3 {
                        let stress_div = self.compute_stress_divergence(i, j, k, comp);

                        // Simplified: ρ ∂²u/∂t² = ∇·σ
                        // Using central difference for second derivative
                        let accel = stress_div / 1000.0; // Simplified density

                        let current_u = self.solid_displacement[[comp, i, j, k]];
                        let prev_u = self.solid_displacement_prev[[comp, i, j, k]];

                        // Verlet integration
                        self.solid_displacement[[comp, i, j, k]] = 2.0 * current_u - prev_u + accel * dt * dt;
                    }
                }
            }
        }

        Ok(())
    }

    /// Update fluid displacement
    fn update_fluid_displacement(&mut self) -> KwaversResult<()> {
        let (nx, ny, nz) = self.grid.dimensions();
        let dt = self.config.dt as f32;

        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    // Fluid momentum equation
                    for comp in 0..3 {
                        let pressure_grad = self.compute_pressure_gradient(i, j, k, comp);

                        // Simplified fluid dynamics (would need full Biot equations)
                        let accel = -pressure_grad / 1000.0; // Simplified

                        let current_w = self.fluid_displacement[[comp, i, j, k]];
                        let prev_w = self.fluid_displacement_prev[[comp, i, j, k]];

                        // Verlet integration
                        self.fluid_displacement[[comp, i, j, k]] = 2.0 * current_w - prev_w + accel * dt * dt;
                    }
                }
            }
        }

        Ok(())
    }

    /// Compute divergence of stress tensor
    fn compute_stress_divergence(&self, i: usize, j: usize, k: usize, comp: usize) -> f32 {
        let dx = self.grid.dx as f32;
        let dy = self.grid.dy as f32;
        let dz = self.grid.dz as f32;

        match comp {
            0 => { // x-component
                (self.solid_stress[[0, i+1, j, k]] - self.solid_stress[[0, i-1, j, k]]) / (2.0 * dx) +
                (self.solid_stress[[3, i, j+1, k]] - self.solid_stress[[3, i, j-1, k]]) / (2.0 * dy) +
                (self.solid_stress[[4, i, j, k+1]] - self.solid_stress[[4, i, j, k-1]]) / (2.0 * dz)
            }
            1 => { // y-component
                (self.solid_stress[[3, i+1, j, k]] - self.solid_stress[[3, i-1, j, k]]) / (2.0 * dx) +
                (self.solid_stress[[1, i, j+1, k]] - self.solid_stress[[1, i, j-1, k]]) / (2.0 * dy) +
                (self.solid_stress[[5, i, j, k+1]] - self.solid_stress[[5, i, j, k-1]]) / (2.0 * dz)
            }
            2 => { // z-component
                (self.solid_stress[[4, i+1, j, k]] - self.solid_stress[[4, i-1, j, k]]) / (2.0 * dx) +
                (self.solid_stress[[5, i, j+1, k]] - self.solid_stress[[5, i, j-1, k]]) / (2.0 * dy) +
                (self.solid_stress[[2, i, j, k+1]] - self.solid_stress[[2, i, j, k-1]]) / (2.0 * dz)
            }
            _ => 0.0,
        }
    }

    /// Compute gradient of fluid pressure
    fn compute_pressure_gradient(&self, i: usize, j: usize, k: usize, comp: usize) -> f32 {
        let dx = self.grid.dx as f32;
        let dy = self.grid.dy as f32;
        let dz = self.grid.dz as f32;

        match comp {
            0 => (self.fluid_pressure[[i+1, j, k]] - self.fluid_pressure[[i-1, j, k]]) / (2.0 * dx),
            1 => (self.fluid_pressure[[i, j+1, k]] - self.fluid_pressure[[i, j-1, k]]) / (2.0 * dy),
            2 => (self.fluid_pressure[[i, j, k+1]] - self.fluid_pressure[[i, j, k-1]]) / (2.0 * dz),
            _ => 0.0,
        }
    }

    /// Compute total energy in the system
    fn compute_total_energy(&self) -> f64 {
        // Simplified energy computation
        let kinetic_energy: f32 = self.solid_displacement.iter()
            .chain(self.fluid_displacement.iter())
            .map(|&x| x * x)
            .sum();

        let potential_energy: f32 = self.solid_stress.iter()
            .map(|&x| x * x)
            .sum();

        (kinetic_energy + potential_energy) as f64
    }

    /// Compute convergence metric
    fn compute_convergence_metric(&self) -> f64 {
        // Simplified convergence check based on energy stability
        let energy = self.compute_total_energy();
        if energy > 0.0 {
            1.0 / (1.0 + energy.ln().abs())
        } else {
            1.0
        }
    }
}

/// Perfectly matched layer for poroelastic waves
pub struct PMLLayer {
    thickness: usize,
    damping_profile: Array3<f32>,
}

impl PMLLayer {
    pub fn new(
        grid: &Grid,
        thickness: usize,
        properties: &crate::medium::poroelastic::PoroelasticProperties,
    ) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();
        let mut damping_profile = Array3::<f32>::zeros((nx, ny, nz));

        // Create damping profile for PML
        let max_damping = 1000.0; // Tunable parameter

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let dist_from_boundary = [
                        i.min(nx - 1 - i),
                        j.min(ny - 1 - j),
                        k.min(nz - 1 - k),
                    ].iter().min().unwrap();

                    if *dist_from_boundary < thickness {
                        let normalized_dist = *dist_from_boundary as f32 / thickness as f32;
                        damping_profile[[i, j, k]] = max_damping * (1.0 - normalized_dist).powi(2);
                    }
                }
            }
        }

        Ok(Self {
            thickness,
            damping_profile,
        })
    }

    pub fn apply(
        &self,
        solid_disp: &mut Array4<f32>,
        fluid_disp: &mut Array4<f32>,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = solid_disp.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let damping = self.damping_profile[[i, j, k]];
                    if damping > 0.0 {
                        // Apply damping to all displacement components
                        for comp in 0..3 {
                            solid_disp[[comp, i, j, k]] *= (1.0 - damping).exp();
                            fluid_disp[[comp, i, j, k]] *= (1.0 - damping).exp();
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Simulation result data
#[derive(Debug)]
pub struct PoroelasticSimulationResult {
    pub config: PoroelasticConfig,
    pub saved_fields: Vec<PoroelasticFields>,
    pub final_energy: f64,
    pub convergence_metric: f64,
}

/// Field data at a specific time
#[derive(Debug, Clone)]
pub struct PoroelasticFields {
    pub time: f64,
    pub solid_displacement: Array4<f32>,
    pub fluid_displacement: Array4<f32>,
    pub solid_stress: Array4<f32>,
    pub fluid_pressure: Array3<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::medium::poroelastic::PoroelasticProperties;

    #[test]
    fn test_poroelastic_solver_creation() {
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();
        let config = PoroelasticConfig::default();
        let properties = PoroelasticProperties::liver();

        let solver = PoroelasticSolver::new(&grid, config, &properties);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_pml_creation() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let properties = PoroelasticProperties::liver();

        let pml = PMLLayer::new(&grid, 5, &properties);
        assert!(pml.is_ok());

        let pml_layer = pml.unwrap();
        assert_eq!(pml_layer.damping_profile.dim(), (32, 32, 32));
    }

    #[test]
    fn test_simulation_basic() {
        let grid = Grid::new(8, 8, 8, 0.002, 0.002, 0.002).unwrap();
        let mut config = PoroelasticConfig::default();
        config.simulation_time = 1e-5; // Very short simulation for testing
        config.save_every = 1;

        let properties = PoroelasticProperties::liver();
        let mut solver = PoroelasticSolver::new(&grid, config, &properties).unwrap();

        let result = solver.simulate();
        assert!(result.is_ok());

        let sim_result = result.unwrap();
        assert!(!sim_result.saved_fields.is_empty());
        assert!(sim_result.final_energy >= 0.0);
    }
}

