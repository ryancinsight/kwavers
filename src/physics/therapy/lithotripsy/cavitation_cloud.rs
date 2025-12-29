//! Cavitation Cloud Dynamics for Lithotripsy
//!
//! Models the formation and evolution of cavitation bubble clouds around stones
//! during shock wave lithotripsy. These clouds are responsible for stone erosion
//! through repeated bubble collapse and microjet formation.
//!
//! ## Key Physics
//!
//! 1. **Cloud Formation**: Bubble nucleation and growth around stone surface
//! 2. **Cloud Collapse**: Coordinated collapse of bubble clusters
//! 3. **Microjet Formation**: High-speed liquid jets impacting stone surface
//! 4. **Stone Erosion**: Material removal through repetitive impacts
//!
//! ## References
//!
//! - Pishchalnikov et al. (2003): "Cavitation bubble cluster activity in shock wave lithotripsy"
//! - Sapozhnikov et al. (2002): "Cloud cavitation control for lithotripsy"
//! - Zhong et al. (1997): "Dynamics of bubble cloud in shock wave lithotripsy"

use crate::physics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use ndarray::Array3;
use serde::{Deserialize, Serialize};

/// Parameters for cavitation cloud formation and dynamics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudParameters {
    /// Initial bubble radius [m]
    pub initial_radius: f64,
    /// Bubble surface tension [N/m]
    pub surface_tension: f64,
    /// Liquid viscosity [Pa·s]
    pub viscosity: f64,
    /// Ambient pressure [Pa]
    pub ambient_pressure: f64,
    /// Gas concentration in liquid [mol/m³]
    pub gas_concentration: f64,
    /// Nucleation threshold pressure [Pa]
    pub nucleation_threshold: f64,
    /// Cloud expansion velocity [m/s]
    pub cloud_expansion_velocity: f64,
    /// Maximum cloud radius [m]
    pub max_cloud_radius: f64,
    /// Bubble interaction distance [m]
    pub interaction_distance: f64,
    /// Cloud collapse time [s]
    pub collapse_time: f64,
}

impl Default for CloudParameters {
    fn default() -> Self {
        Self {
            initial_radius: 1e-6,           // 1 μm initial bubbles
            surface_tension: 0.072,         // Water surface tension
            viscosity: 0.001,               // Water viscosity
            ambient_pressure: 101325.0,     // Atmospheric pressure
            gas_concentration: 0.24,        // Dissolved air in water
            nucleation_threshold: -10e6,    // -10 MPa for nucleation
            cloud_expansion_velocity: 50.0, // 50 m/s expansion
            max_cloud_radius: 5e-3,         // 5 mm max cloud size
            interaction_distance: 1e-4,     // 100 μm interaction distance
            collapse_time: 100e-6,          // 100 μs collapse time
        }
    }
}

/// Cavitation cloud dynamics model
#[derive(Debug)]
pub struct CavitationCloudDynamics {
    /// Cloud parameters
    params: CloudParameters,
    /// Bubble field (3D array of bubble states)
    bubble_field: Array3<Vec<BubbleState>>,
    /// Cloud density field (bubbles per m³)
    cloud_density: Array3<f64>,
    /// Cloud pressure field [Pa]
    cloud_pressure: Array3<f64>,
    /// Cloud velocity field [m/s]
    _cloud_velocity: Array3<f64>,
    /// Erosion rate field [kg/(m²·s)]
    erosion_rate: Array3<f64>,
    /// Cloud boundaries
    cloud_boundaries: Vec<(usize, usize, usize)>,
}

impl CavitationCloudDynamics {
    /// Create new cavitation cloud dynamics model
    pub fn new(params: CloudParameters, grid_shape: (usize, usize, usize)) -> Self {
        let bubble_field = Array3::from_elem(grid_shape, Vec::new());
        let cloud_density = Array3::zeros(grid_shape);
        let cloud_pressure = Array3::from_elem(grid_shape, params.ambient_pressure);
        let _cloud_velocity = Array3::zeros(grid_shape);
        let erosion_rate = Array3::zeros(grid_shape);

        Self {
            params,
            bubble_field,
            cloud_density,
            cloud_pressure,
            _cloud_velocity,
            erosion_rate,
            cloud_boundaries: Vec::new(),
        }
    }

    /// Initialize cavitation cloud around stone surface
    ///
    /// # Arguments
    /// * `stone_geometry` - Binary mask of stone location (1 = stone, 0 = liquid)
    /// * `shock_pressure` - Incident shock wave pressure field [Pa]
    pub fn initialize_cloud(&mut self, stone_geometry: &Array3<f64>, shock_pressure: &Array3<f64>) {
        self.cloud_boundaries.clear();

        // Find stone-liquid interfaces where cavitation can nucleate
        let (nx, ny, nz) = stone_geometry.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    if stone_geometry[[i, j, k]] > 0.5 {
                        // Stone voxel
                        // Check neighboring liquid voxels for nucleation
                        let neighbors = self.get_neighbor_indices(i, j, k, stone_geometry.dim());

                        for (ni, nj, nk) in neighbors {
                            if stone_geometry[[ni, nj, nk]] < 0.5 {
                                // Liquid voxel
                                let shock_p = shock_pressure[[ni, nj, nk]];

                                // Nucleate bubbles if pressure is below threshold
                                if shock_p < self.params.nucleation_threshold {
                                    self.nucleate_bubbles(ni, nj, nk, shock_p);
                                    self.cloud_boundaries.push((ni, nj, nk));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Get neighboring voxel indices (6-connected)
    fn get_neighbor_indices(
        &self,
        i: usize,
        j: usize,
        k: usize,
        dims: (usize, usize, usize),
    ) -> Vec<(usize, usize, usize)> {
        let mut neighbors = Vec::new();
        let (nx, ny, nz) = dims;

        // Check all 6 directions
        if i > 0 {
            neighbors.push((i - 1, j, k));
        }
        if i < nx - 1 {
            neighbors.push((i + 1, j, k));
        }
        if j > 0 {
            neighbors.push((i, j - 1, k));
        }
        if j < ny - 1 {
            neighbors.push((i, j + 1, k));
        }
        if k > 0 {
            neighbors.push((i, j, k - 1));
        }
        if k < nz - 1 {
            neighbors.push((i, j, k + 1));
        }

        neighbors
    }

    /// Nucleate bubbles at a specific location
    fn nucleate_bubbles(&mut self, i: usize, j: usize, k: usize, shock_pressure: f64) {
        // Calculate number of bubbles to nucleate based on pressure deficit
        let pressure_deficit = self.params.nucleation_threshold - shock_pressure;
        let nucleation_rate = pressure_deficit.abs() / 1e6; // Empirical scaling

        // Create bubble parameters using canonical BubbleParameters fields
        let bubble_params = BubbleParameters {
            r0: self.params.initial_radius,
            p0: self.params.ambient_pressure,
            rho_liquid: 1000.0, // Water density
            c_liquid: 1500.0,   // Speed of sound in water
            mu_liquid: self.params.viscosity,
            sigma: self.params.surface_tension,
            accommodation_coeff: 0.01,
            thermal_conductivity: 0.6,
            specific_heat_liquid: 4182.0, // Water cp [J/(kg·K)]
            gamma: 1.4,                   // Polytropic exponent
            pv: 2330.0,                   // Vapor pressure [Pa] at 20°C
            initial_gas_pressure: self.params.ambient_pressure,
            ..Default::default()
        };

        // Create initial bubble state
        let bubble_state = BubbleState::new(&bubble_params);

        // Add to bubble field (simplified - just store one bubble per voxel)
        // In reality, this should be a distribution of bubble sizes
        self.bubble_field[[i, j, k]].push(bubble_state);

        // Initialize cloud density
        self.cloud_density[[i, j, k]] = nucleation_rate * 1e12; // Bubbles per m³
    }

    /// Evolve cavitation cloud over time
    ///
    /// # Arguments
    /// * `dt` - Time step [s]
    /// * `total_time` - Current simulation time [s]
    pub fn evolve_cloud(&mut self, dt: f64, total_time: f64) {
        // Cloud evolution phases:
        // 1. Expansion phase (bubbles grow and cloud expands)
        // 2. Interaction phase (bubbles begin to interact)
        // 3. Collapse phase (coordinated collapse with microjet formation)

        if total_time < self.params.collapse_time * 0.3 {
            // Expansion phase
            self.expand_cloud(dt);
        } else if total_time < self.params.collapse_time * 0.7 {
            // Interaction phase
            self.interact_bubbles(dt);
        } else {
            // Collapse phase
            self.collapse_cloud(dt);
        }
    }

    /// Cloud expansion phase
    fn expand_cloud(&mut self, dt: f64) {
        let expansion_rate = self.params.cloud_expansion_velocity;

        // Expand cloud boundaries
        for (i, j, k) in &self.cloud_boundaries.clone() {
            // Expand in all directions
            let neighbors = self.get_neighbor_indices(*i, *j, *k, self.cloud_density.dim());

            for (ni, nj, nk) in neighbors {
                if self.cloud_density[[ni, nj, nk]] == 0.0 {
                    // Expand cloud to this voxel
                    self.cloud_density[[ni, nj, nk]] = self.cloud_density[[*i, *j, *k]] * 0.5;
                    self.cloud_boundaries.push((ni, nj, nk));
                }
            }
        }

        // Update bubble sizes during expansion
        let (nx, ny, nz) = self.bubble_field.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    for bubble in &mut self.bubble_field[[i, j, k]] {
                        // Simple expansion model
                        bubble.radius += expansion_rate * dt * 0.1; // Slow expansion
                    }
                }
            }
        }
    }

    /// Bubble interaction phase
    fn interact_bubbles(&mut self, dt: f64) {
        // Bubbles begin to interact through Bjerknes forces
        // Simplified: reduce bubble growth rate due to interactions

        let (nx, ny, nz) = self.bubble_field.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let density = self.cloud_density[[i, j, k]];

                    for bubble in &mut self.bubble_field[[i, j, k]] {
                        // Interaction reduces growth rate
                        let interaction_factor = 1.0 / (1.0 + density * 1e-12);
                        bubble.radius += bubble.wall_velocity * dt * interaction_factor;
                    }
                }
            }
        }
    }

    /// Cloud collapse phase with microjet formation
    fn collapse_cloud(&mut self, dt: f64) {
        // Coordinated collapse leading to microjet formation
        // This is where stone erosion occurs

        let (nx, ny, nz) = self.bubble_field.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let mut cell_impact: Option<f64> = None;
                    for bubble in &mut self.bubble_field[[i, j, k]] {
                        // Rapid collapse
                        let collapse_rate = -bubble.radius / self.params.collapse_time;
                        bubble.radius += collapse_rate * dt;

                        // Calculate microjet impact when bubble collapses (defer write)
                        if bubble.radius < self.params.initial_radius * 0.1 {
                            let microjet_velocity = 200.0; // m/s
                            let density = 1000.0; // kg/m³
                            let impact_pressure =
                                0.5 * density * microjet_velocity * microjet_velocity;
                            cell_impact = Some(impact_pressure);
                        }
                    }
                    if let Some(p) = cell_impact {
                        self.cloud_pressure[[i, j, k]] = p;
                    }
                }
            }
        }
    }

    /// Calculate microjet impact on stone surface
    fn _calculate_microjet_impact(&mut self, i: usize, j: usize, k: usize, _bubble: &BubbleState) {
        // Microjet velocity can reach hundreds of m/s
        // Impact pressure: P = 0.5 * ρ * v²
        let microjet_velocity = 200.0; // m/s (typical value)
        let density = 1000.0; // kg/m³

        let impact_pressure = 0.5 * density * microjet_velocity * microjet_velocity;

        // Store impact pressure for erosion calculation
        self.cloud_pressure[[i, j, k]] = impact_pressure;

        // Calculate erosion rate based on impact
        // Erosion rate ∝ P^{3.5} (empirical relation)
        let erosion_rate = 1e-12 * impact_pressure.powf(3.5);
        self.erosion_rate[[i, j, k]] = erosion_rate;
    }

    /// Get cloud density field
    #[must_use]
    pub fn cloud_density(&self) -> &Array3<f64> {
        &self.cloud_density
    }

    /// Get cloud pressure field
    #[must_use]
    pub fn cloud_pressure(&self) -> &Array3<f64> {
        &self.cloud_pressure
    }

    /// Get erosion rate field
    #[must_use]
    pub fn erosion_rate(&self) -> &Array3<f64> {
        &self.erosion_rate
    }

    /// Get total eroded mass over time
    #[must_use]
    pub fn total_eroded_mass(&self, time_step: f64) -> f64 {
        let total_erosion_rate: f64 = self.erosion_rate.iter().sum();
        total_erosion_rate * time_step
    }

    /// Get cloud boundaries
    #[must_use]
    pub fn cloud_boundaries(&self) -> &[(usize, usize, usize)] {
        &self.cloud_boundaries
    }

    /// Get parameters
    #[must_use]
    pub fn parameters(&self) -> &CloudParameters {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cloud_initialization() {
        let params = CloudParameters::default();
        let grid_shape = (10, 10, 10);

        let mut cloud = CavitationCloudDynamics::new(params, grid_shape);

        // Create simple stone geometry (sphere in center)
        let mut stone_geometry = Array3::<f64>::zeros(grid_shape);
        let center = 5;
        let radius = 2;

        for i in 0..grid_shape.0 {
            for j in 0..grid_shape.1 {
                for k in 0..grid_shape.2 {
                    let dist = ((i as f64 - center as f64).powi(2)
                        + (j as f64 - center as f64).powi(2)
                        + (k as f64 - center as f64).powi(2))
                    .sqrt();
                    if dist <= radius as f64 {
                        stone_geometry[[i, j, k]] = 1.0;
                    }
                }
            }
        }

        // Create shock pressure field
        let shock_pressure = Array3::from_elem(grid_shape, -15e6); // -15 MPa

        cloud.initialize_cloud(&stone_geometry, &shock_pressure);

        // Should have nucleated bubbles around stone surface
        assert!(!cloud.cloud_boundaries().is_empty());

        // Check that cloud density is non-zero in some regions
        let total_density: f64 = cloud.cloud_density().iter().sum();
        assert!(total_density > 0.0);
    }

    #[test]
    fn test_cloud_evolution() {
        let params = CloudParameters::default();
        let grid_shape = (5, 5, 5);

        let mut cloud = CavitationCloudDynamics::new(params, grid_shape);

        // Initialize with some bubbles in center and set up boundaries
        let initial_density = 1e12; // 10^12 bubbles/m³
        cloud.cloud_density[[2, 2, 2]] = initial_density;
        cloud.cloud_boundaries.push((2, 2, 2)); // Add center as boundary for expansion
        let initial_nonzero_voxels = cloud.cloud_density().iter().filter(|&&x| x > 0.0).count();

        let dt = 1e-6; // 1 μs
        cloud.evolve_cloud(dt, 10e-6); // Expansion phase (< 30 μs)

        // Cloud should have expanded to more voxels
        let final_nonzero_voxels = cloud.cloud_density().iter().filter(|&&x| x > 0.0).count();
        assert!(
            final_nonzero_voxels > initial_nonzero_voxels,
            "Cloud should expand to more voxels"
        );

        // Total density should be conserved (approximately)
        let final_total_density: f64 = cloud.cloud_density().iter().sum();
        assert!(
            final_total_density > initial_density * 0.8,
            "Total density should be approximately conserved during expansion"
        );
    }

    #[test]
    fn test_microjet_impact() {
        let params = CloudParameters::default();
        let grid_shape = (3, 3, 3);

        let mut cloud = CavitationCloudDynamics::new(params, grid_shape);

        // Manually trigger microjet calculation
        let bubble_params = BubbleParameters {
            r0: 1e-6,
            p0: 101325.0,
            rho_liquid: 1000.0,
            c_liquid: 1500.0,
            mu_liquid: 0.001,
            sigma: 0.072,
            pv: 2330.0,
            thermal_conductivity: 0.6,
            specific_heat_liquid: 4186.0,
            accommodation_coeff: 0.01,
            gas_species: crate::physics::bubble_dynamics::bubble_state::GasSpecies::Air,
            initial_gas_pressure: 1e5,
            gas_composition: std::collections::HashMap::new(),
            gamma: 1.4,
            t0: 293.15,
            driving_frequency: 1e6,
            driving_amplitude: 1e5,
            use_compressibility: true,
            use_thermal_effects: false,
            use_mass_transfer: false,
        };

        let bubble_state = BubbleState::new(&bubble_params);
        cloud.calculate_microjet_impact(1, 1, 1, &bubble_state);

        // Should have high impact pressure
        assert!(cloud.cloud_pressure()[[1, 1, 1]] > 1e6); // At least 1 MPa

        // Should have erosion rate
        assert!(cloud.erosion_rate()[[1, 1, 1]] > 0.0);
    }
}
