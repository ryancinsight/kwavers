//! Complete Lithotripsy Simulation Suite
//!
//! Integrates all lithotripsy physics components into a unified simulation
//! framework capable of modeling stone fragmentation, cavitation effects,
//! and tissue bioeffects with real-time safety monitoring.
//!
//! ## Simulation Components
//!
//! 1. **Stone Model**: Material properties, fracture mechanics, damage accumulation
//! 2. **Shock Wave Physics**: Generation, propagation, nonlinear effects
//! 3. **Cavitation Cloud**: Bubble nucleation, cloud dynamics, erosion
//! 4. **Bioeffects**: Tissue damage assessment, safety monitoring
//! 5. **Treatment Planning**: Optimal parameter selection, dose calculation
//!
//! ## Clinical Applications
//!
//! - **Lithotripsy**: Kidney and gall stone fragmentation
//! - **Shared Components**: Reusable for sonodynamic therapy and histotripsy
//!
//! ## References
//!
//! - Coleman et al. (2011): "The physics and physiology of shock wave lithotripsy"
//! - Cleveland et al. (2000): "The physics of shock wave lithotripsy"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::plugin::kzk_solver::KzkSolverPlugin;
use ndarray::Array3;

use super::{
    bioeffects::{BioeffectsModel, BioeffectsParameters, SafetyAssessment},
    cavitation_cloud::{CavitationCloudDynamics, CloudParameters},
    shock_wave::{ShockWaveGenerator, ShockWaveParameters, ShockWavePropagation},
    stone_fracture::{StoneFractureModel, StoneMaterial},
};

/// Lithotripsy simulation parameters
#[derive(Debug, Clone)]
pub struct LithotripsyParameters {
    /// Stone material properties
    pub stone_material: StoneMaterial,
    /// Shock wave generation parameters
    pub shock_parameters: ShockWaveParameters,
    /// Cavitation cloud parameters
    pub cloud_parameters: CloudParameters,
    /// Bioeffects assessment parameters
    pub bioeffects_parameters: BioeffectsParameters,
    /// Treatment frequency [Hz]
    pub treatment_frequency: f64,
    /// Number of shock waves to deliver
    pub num_shock_waves: usize,
    /// Time between shock waves [s]
    pub interpulse_delay: f64,
    /// Stone geometry (3D binary mask)
    pub stone_geometry: Array3<f64>,
}

impl Default for LithotripsyParameters {
    fn default() -> Self {
        Self {
            stone_material: StoneMaterial::calcium_oxalate_monohydrate(),
            shock_parameters: ShockWaveParameters::default(),
            cloud_parameters: CloudParameters::default(),
            bioeffects_parameters: BioeffectsParameters::default(),
            treatment_frequency: 1.0,                    // 1 Hz
            num_shock_waves: 1000,                       // Typical treatment
            interpulse_delay: 1.0,                       // 1 second between pulses
            stone_geometry: Array3::zeros((32, 32, 32)), // Will be set later
        }
    }
}

/// Simulation results
#[derive(Debug)]
pub struct SimulationResults {
    /// Final stone damage field
    pub final_stone_damage: Array3<f64>,
    /// Fragment size distribution [m]
    pub fragment_sizes: Vec<f64>,
    /// Total eroded mass [kg]
    pub total_eroded_mass: f64,
    /// Treatment time [s]
    pub treatment_time: f64,
    /// Number of shock waves delivered
    pub shock_waves_delivered: usize,
    /// Final safety assessment
    pub final_safety_assessment: SafetyAssessment,
    /// Time series of stone volume reduction
    pub stone_volume_history: Vec<f64>,
    /// Time series of eroded mass
    pub erosion_history: Vec<f64>,
    /// Cavitation dose history
    pub cavitation_history: Vec<f64>,
}

/// Complete lithotripsy simulator
#[derive(Debug)]
pub struct LithotripsySimulator {
    /// Simulation parameters
    params: LithotripsyParameters,
    /// Computational grid
    grid: Grid,
    /// Stone fracture model
    stone_model: StoneFractureModel,
    /// Shock wave generator
    shock_generator: ShockWaveGenerator,
    /// Shock wave propagator
    shock_propagator: ShockWavePropagation,
    /// Cavitation cloud dynamics
    cavitation_cloud: CavitationCloudDynamics,
    /// Bioeffects model
    bioeffects_model: BioeffectsModel,
    /// KZK solver for nonlinear propagation
    _kzk_solver: KzkSolverPlugin,
    /// Simulation state
    simulation_time: f64,
    /// Results accumulator
    results: SimulationResults,
}

impl LithotripsySimulator {
    /// Create new lithotripsy simulator
    pub fn new(params: LithotripsyParameters, grid: Grid) -> KwaversResult<Self> {
        // Validate stone geometry matches grid
        if params.stone_geometry.dim() != grid.dimensions() {
            return Err(crate::error::KwaversError::Validation(
                crate::error::ValidationError::FieldValidation {
                    field: "stone_geometry".to_string(),
                    value: format!("{:?}", params.stone_geometry.dim()),
                    constraint: format!("Must match grid dimensions {:?}", grid.dimensions())
                        .to_string(),
                },
            ));
        }

        let stone_model = StoneFractureModel::new(params.stone_material.clone(), grid.dimensions());
        let shock_generator = ShockWaveGenerator::new(params.shock_parameters.clone(), &grid)?;
        let shock_propagator = ShockWavePropagation::new(0.1, &grid)?;
        let cavitation_cloud =
            CavitationCloudDynamics::new(params.cloud_parameters.clone(), grid.dimensions());
        let bioeffects_model =
            BioeffectsModel::new(grid.dimensions(), params.bioeffects_parameters.clone());
        let kzk_solver = KzkSolverPlugin::new();

        let results = SimulationResults {
            final_stone_damage: Array3::zeros(grid.dimensions()),
            fragment_sizes: Vec::new(),
            total_eroded_mass: 0.0,
            treatment_time: 0.0,
            shock_waves_delivered: 0,
            final_safety_assessment: SafetyAssessment {
                overall_safe: true,
                max_mechanical_index: 0.0,
                max_thermal_index: 0.0,
                max_cavitation_dose: 0.0,
                max_damage_probability: 0.0,
                violations: Vec::new(),
            },
            stone_volume_history: Vec::new(),
            erosion_history: Vec::new(),
            cavitation_history: Vec::new(),
        };

        Ok(Self {
            params,
            grid,
            stone_model,
            shock_generator,
            shock_propagator,
            cavitation_cloud,
            bioeffects_model,
            _kzk_solver: kzk_solver,
            simulation_time: 0.0,
            results,
        })
    }

    /// Run complete lithotripsy simulation
    pub fn run_simulation(&mut self) -> KwaversResult<&SimulationResults> {
        println!(
            "Starting lithotripsy simulation with {} shock waves",
            self.params.num_shock_waves
        );

        // Record initial stone volume
        let initial_volume = self.calculate_stone_volume();
        self.results.stone_volume_history.push(initial_volume);

        for pulse_num in 1..=self.params.num_shock_waves {
            // Deliver one shock wave
            self.deliver_shock_wave()?;

            // Update accounting immediately upon delivery
            self.results.shock_waves_delivered = pulse_num;
            self.simulation_time += self.params.interpulse_delay;

            // Check safety after each pulse
            let safety = self.bioeffects_model.check_safety();
            if !safety.overall_safe {
                println!("Safety limit exceeded after {} pulses", pulse_num);
                break;
            }

            // Record progress
            if pulse_num % 100 == 0 {
                let current_volume = self.calculate_stone_volume();
                let eroded_mass = self
                    .cavitation_cloud
                    .total_eroded_mass(self.params.interpulse_delay);

                println!(
                    "Pulse {}: Stone volume = {:.2e} m³, Eroded mass = {:.2e} kg",
                    pulse_num, current_volume, eroded_mass
                );

                self.results.stone_volume_history.push(current_volume);
                self.results.erosion_history.push(eroded_mass);
            }
        }

        // Finalize results
        self.finalize_results();

        println!(
            "Simulation complete: {} pulses delivered, {:.2e} kg eroded",
            self.results.shock_waves_delivered, self.results.total_eroded_mass
        );

        Ok(&self.results)
    }

    /// Deliver a single shock wave
    fn deliver_shock_wave(&mut self) -> KwaversResult<()> {
        // Generate shock wave at source
        let source_pressure = self
            .shock_generator
            .generate_shock_field(&self.grid, self.params.shock_parameters.center_frequency);

        // Propagate shock wave to stone region
        let propagated_pressure = self.shock_propagator.propagate_shock_wave(
            &source_pressure,
            self.params.shock_parameters.center_frequency,
        )?;

        // Apply shock wave to stone
        self.apply_shock_to_stone(&propagated_pressure)?;

        // Trigger cavitation effects
        self.simulate_cavitation_effects(&propagated_pressure)?;

        // Update bioeffects assessment
        let intensity_field = self.calculate_acoustic_intensity(&propagated_pressure);
        let cavitation_field = self.cavitation_cloud.cloud_density();

        self.bioeffects_model.update_assessment(
            &propagated_pressure,
            &intensity_field,
            cavitation_field,
            self.params.shock_parameters.center_frequency,
            self.params.interpulse_delay,
        );

        Ok(())
    }

    /// Apply shock wave loading to stone fracture model
    fn apply_shock_to_stone(&mut self, pressure_field: &Array3<f64>) -> KwaversResult<()> {
        // Convert acoustic pressure to stress in stone
        // Simplified: assume direct coupling (in reality, there's acoustic impedance mismatch)
        let stress_field = pressure_field.clone();

        // Calculate strain rate from shock rise time
        let strain_rate = 1.0 / self.params.shock_parameters.rise_time;

        // Apply damage accumulation
        self.stone_model.apply_stress_loading(
            &stress_field,
            self.params.interpulse_delay,
            strain_rate,
        );

        Ok(())
    }

    /// Simulate cavitation effects from shock wave
    fn simulate_cavitation_effects(&mut self, shock_pressure: &Array3<f64>) -> KwaversResult<()> {
        // Initialize cavitation cloud around stone
        self.cavitation_cloud
            .initialize_cloud(&self.params.stone_geometry, shock_pressure);

        // Evolve cloud over the pulse duration
        let cloud_time_steps = 100;
        let dt = self.params.shock_parameters.pulse_duration / cloud_time_steps as f64;

        for step in 0..cloud_time_steps {
            let time = step as f64 * dt;
            self.cavitation_cloud.evolve_cloud(dt, time);
        }

        Ok(())
    }

    /// Calculate acoustic intensity field
    fn calculate_acoustic_intensity(&self, pressure_field: &Array3<f64>) -> Array3<f64> {
        // I = (1/(2ρc)) * p² (time-averaged intensity)
        let rho = 1000.0; // kg/m³
        let c = 1500.0; // m/s

        pressure_field.mapv(|p| p * p / (2.0 * rho * c))
    }

    /// Calculate current stone volume
    fn calculate_stone_volume(&self) -> f64 {
        let voxel_volume = self.grid.dx * self.grid.dy * self.grid.dz;
        let intact_voxels = self
            .params
            .stone_geometry
            .iter()
            .zip(self.stone_model.damage_field().iter())
            .filter(|(&geom, &damage)| geom > 0.5 && damage < 1.0)
            .count();

        intact_voxels as f64 * voxel_volume
    }

    /// Finalize simulation results
    fn finalize_results(&mut self) {
        // Copy final stone damage
        self.results.final_stone_damage = self.stone_model.damage_field().clone();

        // Get fragment sizes
        self.results.fragment_sizes = self.stone_model.fragment_sizes().to_vec();

        // Calculate total eroded mass
        self.results.total_eroded_mass = self
            .cavitation_cloud
            .total_eroded_mass(self.simulation_time);

        // Set treatment time
        self.results.treatment_time = self.simulation_time;

        // Get final safety assessment
        self.results.final_safety_assessment = self.bioeffects_model.check_safety().clone();

        // Add final values to history
        let final_volume = self.calculate_stone_volume();
        self.results.stone_volume_history.push(final_volume);
        self.results
            .erosion_history
            .push(self.results.total_eroded_mass);
    }

    /// Get current simulation state
    #[must_use]
    pub fn current_state(&self) -> LithotripsyState {
        LithotripsyState {
            simulation_time: self.simulation_time,
            shock_waves_delivered: self.results.shock_waves_delivered,
            stone_volume: self.calculate_stone_volume(),
            eroded_mass: self
                .cavitation_cloud
                .total_eroded_mass(self.simulation_time),
            safety_assessment: self
                .bioeffects_model
                .current_assessment()
                .check_safety_limits(),
        }
    }

    /// Get simulation parameters
    #[must_use]
    pub fn parameters(&self) -> &LithotripsyParameters {
        &self.params
    }

    /// Get stone model
    #[must_use]
    pub fn stone_model(&self) -> &StoneFractureModel {
        &self.stone_model
    }

    /// Get cavitation cloud
    #[must_use]
    pub fn cavitation_cloud(&self) -> &CavitationCloudDynamics {
        &self.cavitation_cloud
    }

    /// Get bioeffects model
    #[must_use]
    pub fn bioeffects_model(&self) -> &BioeffectsModel {
        &self.bioeffects_model
    }
}

/// Current simulation state snapshot
#[derive(Debug, Clone)]
pub struct LithotripsyState {
    /// Current simulation time [s]
    pub simulation_time: f64,
    /// Number of shock waves delivered
    pub shock_waves_delivered: usize,
    /// Current stone volume [m³]
    pub stone_volume: f64,
    /// Total eroded mass [kg]
    pub eroded_mass: f64,
    /// Current safety assessment
    pub safety_assessment: SafetyAssessment,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_lithotripsy_simulator_creation() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).unwrap();

        let params = LithotripsyParameters {
            stone_geometry: Array3::zeros(grid.dimensions()),
            ..Default::default()
        };

        let simulator = LithotripsySimulator::new(params, grid);
        assert!(simulator.is_ok());
    }

    #[test]
    fn test_stone_volume_calculation() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();

        let mut params = LithotripsyParameters {
            stone_geometry: Array3::zeros(grid.dimensions()),
            ..Default::default()
        };

        // Create a 2x2x2 stone in the center
        for i in 4..6 {
            for j in 4..6 {
                for k in 4..6 {
                    params.stone_geometry[[i, j, k]] = 1.0;
                }
            }
        }

        let simulator = LithotripsySimulator::new(params, grid).unwrap();

        // Stone volume should be 8 * voxel_volume = 8e-9 m³
        let expected_volume = 8.0 * 1e-9;
        let actual_volume = simulator.calculate_stone_volume();

        assert!((actual_volume - expected_volume).abs() < 1e-12);
    }

    #[test]
    fn test_simulation_state() {
        let grid = Grid::new(8, 8, 8, 1e-3, 1e-3, 1e-3).unwrap();

        let params = LithotripsyParameters {
            num_shock_waves: 10, // Short simulation for testing
            stone_geometry: Array3::zeros(grid.dimensions()),
            ..Default::default()
        };

        let mut simulator = LithotripsySimulator::new(params, grid).unwrap();

        // Run short simulation
        let results = simulator.run_simulation().unwrap();

        // Should have delivered some shock waves
        assert!(results.shock_waves_delivered > 0);

        // Should have some treatment time
        assert!(results.treatment_time > 0.0);

        // Safety should be assessed
        assert!(
            results.final_safety_assessment.overall_safe
                || !results.final_safety_assessment.violations.is_empty()
        );
    }
}
