use crate::core::constants::fundamental::{
    ATMOSPHERIC_PRESSURE, BOLTZMANN, ELECTRON_MASS, ELEMENTARY_CHARGE,
};
use crate::core::constants::thermodynamic::BUBBLE_REFERENCE_TEMPERATURE_K;
use crate::core::error::KwaversResult;
use crate::physics::bubble_dynamics::bubble_state::{BubbleParameters, BubbleState};
use crate::physics::bubble_dynamics::keller_miksis::KellerMiksisModel;
use ndarray::Array3;

use super::super::emission_calculator::SonoluminescenceEmission;
use super::thermodynamics::update_thermodynamics;
use crate::core::constants::numerical::TWO_PI;
use crate::physics::optics::sonoluminescence::emission::spectrum::EmissionParameters;

/// Integrated bubble dynamics and sonoluminescence emission
///
/// This struct encapsulates the sonoluminescence emission calculations and
/// field management. Bubble dynamics parameters and models are passed as
/// arguments to simulation methods, following the dependency injection pattern
/// to maintain clean layer separation.
///
/// **Architecture Note**: Bubble dynamics models are NOT stored in this struct.
/// Instead, they are passed as parameters to `simulate_step()`. This maintains
/// the 9-layer architecture where optics layer depends on physics layer, not vice versa.
#[derive(Debug)]
pub struct IntegratedSonoluminescence {
    /// Emission calculator
    pub emission: SonoluminescenceEmission,
    /// Acoustic pressure field driving bubble oscillations (Pa)
    pub acoustic_pressure: Array3<f64>,
    /// Temperature field from bubble dynamics (K)
    pub temperature_field: Array3<f64>,
    /// Pressure field from bubble dynamics (Pa)
    pub pressure_field: Array3<f64>,
    /// Bubble radius field (m)
    pub radius_field: Array3<f64>,
    /// Bubble wall velocity field (m/s)
    pub wall_velocity_field: Array3<f64>,
    /// Particle velocity field for Cherenkov calculations (m/s)
    pub particle_velocity_field: Array3<f64>,
    /// Charge density field for Cherenkov calculations (C/m³)
    pub charge_density_field: Array3<f64>,
    /// Compression ratio field ρ/ρ₀
    pub compression_field: Array3<f64>,
}

impl IntegratedSonoluminescence {
    /// Create new integrated sonoluminescence calculator
    ///
    /// **Parameters**:
    /// - `grid_shape`: (nx, ny, nz) spatial grid dimensions
    /// - `bubble_params`: BubbleParameters used to initialize field values
    /// - `emission_params`: EmissionParameters for light emission calculation
    ///
    /// **Note**: The `bubble_params` argument is used only to initialize the
    /// radius field to the equilibrium radius (r0). The bubble dynamics model
    /// itself is NOT stored. Pass it to `simulate_step()` instead.
    #[must_use]
    pub fn new(
        grid_shape: (usize, usize, usize),
        bubble_params: BubbleParameters,
        emission_params: EmissionParameters,
    ) -> Self {
        let emission = SonoluminescenceEmission::new(grid_shape, emission_params);

        Self {
            emission,
            acoustic_pressure: Array3::zeros(grid_shape),
            // Initial bubble interior: thermal equilibrium at the Brenner (2002)
            // 300 K baseline used by the Cherenkov & emission models in this
            // submodule; ambient pressure sourced from SSOT.
            temperature_field: Array3::from_elem(grid_shape, BUBBLE_REFERENCE_TEMPERATURE_K),
            pressure_field: Array3::from_elem(grid_shape, ATMOSPHERIC_PRESSURE),
            radius_field: Array3::from_elem(grid_shape, bubble_params.r0),
            wall_velocity_field: Array3::zeros(grid_shape),
            particle_velocity_field: Array3::zeros(grid_shape),
            charge_density_field: Array3::zeros(grid_shape),
            compression_field: Array3::from_elem(grid_shape, 1.0),
        }
    }

    /// Set the acoustic pressure field driving bubble oscillations
    pub fn set_acoustic_pressure(&mut self, pressure: Array3<f64>) {
        self.acoustic_pressure = pressure;
    }

    /// Simulate bubble dynamics and calculate sonoluminescence emission
    ///
    /// This method integrates the Keller-Miksis equation for bubble dynamics
    /// with the sonoluminescence emission models to provide physically accurate
    /// light emission calculations.
    ///
    /// **Architecture Pattern**: Bubble dynamics models are passed as parameters
    /// (dependency injection) rather than stored in `self`. This maintains clean
    /// layer separation: optics layer uses physics layer models without owning them.
    ///
    /// Reference: Brenner et al. (2002), "Single-bubble sonoluminescence"
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn simulate_step(
        &mut self,
        dt: f64,
        time: f64,
        bubble_params: &BubbleParameters,
        bubble_model: &KellerMiksisModel,
    ) -> KwaversResult<()> {
        let omega = TWO_PI * bubble_params.driving_frequency;

        let (nx, ny, nz) = self.temperature_field.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let p_amp = self.acoustic_pressure[[i, j, k]];

                    let mut state = BubbleState::new(bubble_params);
                    state.radius = self.radius_field[[i, j, k]];
                    state.wall_velocity = self.wall_velocity_field[[i, j, k]];
                    state.temperature = self.temperature_field[[i, j, k]];
                    state.pressure_internal = self.pressure_field[[i, j, k]];

                    // RK4 integration: dY/dt = F(Y, t), Y = [R, V]
                    // k1
                    let dp_dt_k1 = p_amp * omega * (omega * time).cos();
                    let k1_v =
                        bubble_model.calculate_acceleration(&mut state, p_amp, dp_dt_k1, time)?;
                    let k1_r = state.wall_velocity;

                    // k2
                    let t_k2 = 0.5f64.mul_add(dt, time);
                    let dp_dt_k2 = p_amp * omega * (omega * t_k2).cos();
                    let mut state_k2 = state.clone();
                    state_k2.radius += 0.5 * dt * k1_r;
                    state_k2.wall_velocity += 0.5 * dt * k1_v;
                    update_thermodynamics(&mut state_k2, bubble_params);
                    let k2_v = bubble_model.calculate_acceleration(
                        &mut state_k2,
                        p_amp,
                        dp_dt_k2,
                        t_k2,
                    )?;
                    let k2_r = state_k2.wall_velocity;

                    // k3
                    let dp_dt_k3 = dp_dt_k2;
                    let mut state_k3 = state.clone();
                    state_k3.radius += 0.5 * dt * k2_r;
                    state_k3.wall_velocity += 0.5 * dt * k2_v;
                    update_thermodynamics(&mut state_k3, bubble_params);
                    let k3_v = bubble_model.calculate_acceleration(
                        &mut state_k3,
                        p_amp,
                        dp_dt_k3,
                        t_k2,
                    )?;
                    let k3_r = state_k3.wall_velocity;

                    // k4
                    let t_k4 = time + dt;
                    let dp_dt_k4 = p_amp * omega * (omega * t_k4).cos();
                    let mut state_k4 = state.clone();
                    state_k4.radius += dt * k3_r;
                    state_k4.wall_velocity += dt * k3_v;
                    update_thermodynamics(&mut state_k4, bubble_params);
                    let k4_v = bubble_model.calculate_acceleration(
                        &mut state_k4,
                        p_amp,
                        dp_dt_k4,
                        t_k4,
                    )?;
                    let k4_r = state_k4.wall_velocity;

                    // Final RK4 update
                    let new_radius = (dt / 6.0).mul_add(
                        2.0f64.mul_add(k3_r, 2.0f64.mul_add(k2_r, k1_r)) + k4_r,
                        state.radius,
                    );
                    let new_velocity = (dt / 6.0).mul_add(
                        2.0f64.mul_add(k3_v, 2.0f64.mul_add(k2_v, k1_v)) + k4_v,
                        state.wall_velocity,
                    );

                    state.radius = new_radius;
                    state.wall_velocity = new_velocity;
                    update_thermodynamics(&mut state, bubble_params);

                    // Auxiliary field calculations
                    let compression_ratio = (bubble_params.r0 / new_radius).powi(3);

                    // Particle velocity (thermal electrons + wall motion)
                    let thermal_velocity_sq = 3.0 * BOLTZMANN * state.temperature / ELECTRON_MASS;
                    let collapse_velocity = new_velocity.abs();
                    let particle_velocity =
                        (thermal_velocity_sq + collapse_velocity * collapse_velocity).sqrt();

                    // Charge density (Saha ionization)
                    let ionization_fraction = self.emission.bremsstrahlung.saha_ionization(
                        state.temperature,
                        state.pressure_internal,
                        self.emission.params.ionization_energy,
                    );
                    let number_density = state.pressure_internal / (BOLTZMANN * state.temperature);
                    let electron_density = ionization_fraction * number_density;
                    let charge_density = electron_density * ELEMENTARY_CHARGE;

                    // Store updated state
                    self.radius_field[[i, j, k]] = new_radius;
                    self.wall_velocity_field[[i, j, k]] = new_velocity;
                    self.temperature_field[[i, j, k]] = state.temperature;
                    self.pressure_field[[i, j, k]] = state.pressure_internal;
                    self.particle_velocity_field[[i, j, k]] = particle_velocity;
                    self.charge_density_field[[i, j, k]] = charge_density;
                    self.compression_field[[i, j, k]] = compression_ratio;
                }
            }
        }
        Ok(())
    }
}
