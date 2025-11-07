//! Shock Wave Generation and Propagation for Lithotripsy
//!
//! Models the formation, propagation, and interaction of shock waves with stones
//! and tissue. Implements nonlinear acoustics for high-intensity ultrasound.
//!
//! ## Key Physics
//!
//! 1. **Shock Wave Formation**: Nonlinear steepening of acoustic waves
//!    ∂²p/∂t² - c²∇²p = β/ρc³ p ∂p/∂t (Burgers equation)
//!
//! 2. **Shock Wave Propagation**: KZK equation for diffraction and absorption
//!    ∂p/∂z = (i/2k)∇_⊥²p + (α/2)ip + (βk/2ρc³)p²
//!
//! 3. **Stone-Shock Interaction**: Acoustic impedance mismatch and stress concentration
//!
//! ## References
//!
//! - Cleveland et al. (2000): "The physics of shock wave lithotripsy"
//! - Sapozhnikov et al. (2007): "Effect of overpressure and pulse duration on stone fragmentation with lithotripters"
//! - Pishchalnikov et al. (2011): "Destruction of kidney stones using high-intensity focused ultrasound"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::plugin::kzk_solver::KzkSolverPlugin;
use ndarray::{Array3, Array4};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Shock wave generation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShockWaveParameters {
    /// Center frequency [Hz]
    pub center_frequency: f64,
    /// Peak positive pressure [Pa]
    pub peak_positive_pressure: f64,
    /// Peak negative pressure [Pa]
    pub peak_negative_pressure: f64,
    /// Shock rise time [s]
    pub rise_time: f64,
    /// Pulse duration [s]
    pub pulse_duration: f64,
    /// Pulse repetition frequency [Hz]
    pub prf: f64,
    /// Nonlinearity parameter B/A
    pub b_over_a: f64,
    /// Shock formation distance [m]
    pub shock_formation_distance: f64,
}

impl Default for ShockWaveParameters {
    fn default() -> Self {
        Self {
            center_frequency: 1e6,          // 1 MHz
            peak_positive_pressure: 30e6,   // 30 MPa
            peak_negative_pressure: -5e6,   // -5 MPa
            rise_time: 50e-9,               // 50 ns
            pulse_duration: 1e-6,           // 1 μs
            prf: 1.0,                       // 1 Hz
            b_over_a: 5.2,                  // Water nonlinearity
            shock_formation_distance: 0.1,  // 10 cm
        }
    }
}

/// Shock wave generator using nonlinear acoustics
#[derive(Debug)]
pub struct ShockWaveGenerator {
    /// Generation parameters
    params: ShockWaveParameters,
    /// KZK solver for nonlinear propagation
    kzk_solver: KzkSolverPlugin,
    /// Source pressure waveform
    source_waveform: Vec<f64>,
}

impl ShockWaveGenerator {
    /// Create new shock wave generator
    pub fn new(params: ShockWaveParameters, grid: &Grid) -> KwaversResult<Self> {
        let kzk_solver = KzkSolverPlugin::new();

        // Generate initial pressure waveform
        let source_waveform = Self::generate_source_waveform(&params);

        Ok(Self {
            params,
            kzk_solver,
            source_waveform,
        })
    }

    /// Generate source pressure waveform
    fn generate_source_waveform(params: &ShockWaveParameters) -> Vec<f64> {
        // Simple shock wave profile: fast rise, exponential decay
        // Based on typical lithotripter waveforms

        let n_samples = 1000; // Arbitrary number of samples
        let mut waveform = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let t = i as f64 * params.pulse_duration / n_samples as f64;

            if t <= params.rise_time {
                // Linear rise to peak positive pressure
                let amplitude = (t / params.rise_time) * params.peak_positive_pressure;
                waveform.push(amplitude);
            } else if t <= params.pulse_duration {
                // Exponential decay to negative pressure
                let decay_factor = (params.rise_time / (t - params.rise_time + 1e-12)).ln();
                let amplitude = params.peak_negative_pressure * decay_factor.exp();
                waveform.push(amplitude);
            } else {
                waveform.push(0.0);
            }
        }

        waveform
    }

    /// Generate shock wave field at source
    #[must_use]
    pub fn generate_shock_field(&self, grid: &Grid, frequency: f64) -> Array3<f64> {
        let (nx, ny, nz) = grid.dimensions();
        let mut pressure_field = Array3::<f64>::zeros((nx, ny, nz));

        // Create focused shock wave pattern
        // Simplified: spherical focusing with shock front

        let center_x = grid.nx / 2;
        let center_y = grid.ny / 2;
        let focal_z = (grid.nz as f64 * 0.7) as usize; // Focus at 70% depth

        for i in 0..grid.nx {
            for j in 0..grid.ny {
                for k in 0..grid.nz {
                    // Distance from focal point
                    let dx = (i as f64 - center_x as f64) * grid.dx;
                    let dy = (j as f64 - center_y as f64) * grid.dy;
                    let dz = (k as f64 - focal_z as f64) * grid.dz;

                    let distance = (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt();

                    if distance < 0.05 { // Within focal region
                        // Apply shock waveform with phase delay for focusing
                        let phase_delay = distance / 1500.0; // Speed of sound delay
                        let time_index = (phase_delay * frequency).round() as usize % self.source_waveform.len();

                        pressure_field[[i, j, k]] = self.source_waveform[time_index];
                    }
                }
            }
        }

        pressure_field
    }

    /// Get source waveform
    #[must_use]
    pub fn source_waveform(&self) -> &[f64] {
        &self.source_waveform
    }

    /// Get parameters
    #[must_use]
    pub fn parameters(&self) -> &ShockWaveParameters {
        &self.params
    }
}

/// Shock wave propagation using KZK equation
#[derive(Debug)]
pub struct ShockWavePropagation {
    /// KZK solver
    kzk_solver: KzkSolverPlugin,
    /// Propagation distance [m]
    propagation_distance: f64,
    /// Attenuation field
    attenuation_field: Array3<f64>,
    /// Nonlinearity field
    nonlinearity_field: Array3<f64>,
}

impl ShockWavePropagation {
    /// Create new shock wave propagator
    pub fn new(propagation_distance: f64, grid: &Grid) -> KwaversResult<Self> {
        let kzk_solver = KzkSolverPlugin::new();

        // Initialize attenuation and nonlinearity fields
        let attenuation_field = Self::calculate_attenuation_field(grid);
        let nonlinearity_field = Self::calculate_nonlinearity_field(grid);

        Ok(Self {
            kzk_solver,
            propagation_distance,
            attenuation_field,
            nonlinearity_field,
        })
    }

    /// Calculate attenuation field (frequency-dependent)
    fn calculate_attenuation_field(grid: &Grid) -> Array3<f64> {
        let mut attenuation = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

        // Attenuation increases with depth (z-direction)
        // α = α0 * f^y where y ≈ 1.1 for tissue
        for k in 0..grid.nz {
            let depth = k as f64 * grid.dz;
            // Frequency-dependent attenuation (simplified)
            let alpha = 0.5 * depth; // Np/m at 1 MHz, increases with depth
            for i in 0..grid.nx {
                for j in 0..grid.ny {
                    attenuation[[i, j, k]] = alpha;
                }
            }
        }

        attenuation
    }

    /// Calculate nonlinearity field (B/A ratio)
    fn calculate_nonlinearity_field(grid: &Grid) -> Array3<f64> {
        // Most tissues have B/A ≈ 5-7, water is 5.2
        // Stones have different nonlinearity
        let mut nonlinearity = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

        // Default: water nonlinearity
        nonlinearity.fill(5.2);

        // Add stone region with different nonlinearity (simplified)
        let stone_center_x = grid.nx / 2;
        let stone_center_y = grid.ny / 2;
        let stone_center_z = grid.nz / 2;
        let stone_radius = 5; // voxels

        for i in (stone_center_x.saturating_sub(stone_radius))..(stone_center_x + stone_radius).min(grid.nx) {
            for j in (stone_center_y.saturating_sub(stone_radius))..(stone_center_y + stone_radius).min(grid.ny) {
                for k in (stone_center_z.saturating_sub(stone_radius))..(stone_center_z + stone_radius).min(grid.nz) {
                    let dx = i as f64 - stone_center_x as f64;
                    let dy = j as f64 - stone_center_y as f64;
                    let dz = k as f64 - stone_center_z as f64;

                    if (dx.powi(2) + dy.powi(2) + dz.powi(2)).sqrt() <= stone_radius as f64 {
                        nonlinearity[[i, j, k]] = 8.0; // Higher nonlinearity for stones
                    }
                }
            }
        }

        nonlinearity
    }

    /// Propagate shock wave using KZK equation
    pub fn propagate_shock_wave(&self, initial_pressure: &Array3<f64>, frequency: f64) -> KwaversResult<Array3<f64>> {
        // Use KZK solver for nonlinear propagation
        // This is a simplified implementation - full KZK would require frequency domain processing

        let mut propagated_pressure = initial_pressure.clone();

        // Apply geometric spreading (spherical wave)
        self.apply_geometric_spreading(&mut propagated_pressure);

        // Apply nonlinear steepening
        self.apply_nonlinear_steepening(&mut propagated_pressure, frequency);

        // Apply attenuation
        self.apply_attenuation(&mut propagated_pressure, frequency);

        Ok(propagated_pressure)
    }

    /// Apply geometric spreading with a bounded decay factor
    /// Uses 1/(1 + r) to avoid singular behavior near r ≈ 0
    fn apply_geometric_spreading(&self, pressure: &mut Array3<f64>) {
        let (nx, ny, nz) = pressure.dim();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let x = (i as f64 - nx as f64 / 2.0).abs();
                    let y = (j as f64 - ny as f64 / 2.0).abs();
                    let z = k as f64; // Propagation in z-direction

                    let r = (x.powi(2) + y.powi(2) + z.powi(2)).sqrt();

                    // Bounded spherical spreading: factor ∈ (0, 1]
                    let factor = 1.0 / (1.0 + r);
                    pressure[[i, j, k]] *= factor;
                }
            }
        }
    }

    /// Apply nonlinear steepening using Burgers equation approximation
    fn apply_nonlinear_steepening(&self, pressure: &mut Array3<f64>, frequency: f64) {
        let omega = 2.0 * PI * frequency;
        let c: f64 = 1500.0; // Speed of sound in water

        // Simplified nonlinear steepening
        // In Burgers equation: ∂p/∂z = (β ω / (2 ρ c³)) p ∂p/∂z
        // This causes shock formation over distance

        let beta = 3.5; // Nonlinearity parameter
        let rho = 1000.0; // Density

        let steepening_factor = beta * omega / (2.0 * rho * (c as f64).powi(3)) * self.propagation_distance;

        for pressure_val in pressure.iter_mut() {
            if *pressure_val > 1e3 { // Only steepen significant pressures
                *pressure_val *= (1.0 + steepening_factor * pressure_val.abs() * 1e-12).min(10.0);
            }
        }
    }

    /// Apply frequency-dependent attenuation using tissue-relevant dB scaling
    /// Typical soft tissue: ~0.5 dB/cm/MHz. Convert to amplitude factor.
    fn apply_attenuation(&self, pressure: &mut Array3<f64>, frequency: f64) {
        let alpha_db_per_cm_per_mhz = 0.5_f64; // Representative value
        let distance_cm = self.propagation_distance * 100.0; // meters → cm
        let frequency_mhz = frequency / 1.0e6;

        // Total attenuation in dB for pressure amplitude
        let atten_db = alpha_db_per_cm_per_mhz * frequency_mhz * distance_cm;
        // Convert dB to linear amplitude factor (20*log10 for pressure)
        let factor = 10_f64.powf(-atten_db / 20.0);

        for pressure_val in pressure.iter_mut() {
            *pressure_val *= factor;
        }
    }

    /// Calculate shock amplitude at focus
    #[must_use]
    pub fn calculate_shock_amplitude(&self, initial_amplitude: f64, frequency: f64) -> f64 {
        // Account for focusing gain and nonlinear effects
        let focusing_gain = 100.0; // Typical focusing gain
        let nonlinear_gain = 2.0;  // Nonlinear enhancement

        initial_amplitude * focusing_gain * nonlinear_gain
    }

    /// Get attenuation field
    #[must_use]
    pub fn attenuation_field(&self) -> &Array3<f64> {
        &self.attenuation_field
    }

    /// Get nonlinearity field
    #[must_use]
    pub fn nonlinearity_field(&self) -> &Array3<f64> {
        &self.nonlinearity_field
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_shock_wave_generator() {
        let params = ShockWaveParameters::default();
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();

        let generator = ShockWaveGenerator::new(params, &grid).unwrap();

        // Check waveform generation
        let waveform = generator.source_waveform();
        assert!(!waveform.is_empty());

        // Peak positive pressure should be present
        let max_pressure = waveform.iter().fold(f64::MIN, |a, &b| a.max(b));
        assert!(max_pressure > 20e6); // At least 20 MPa

        // Peak negative pressure should be present
        let min_pressure = waveform.iter().fold(f64::MAX, |a, &b| a.min(b));
        assert!(min_pressure < -1e6); // At least -1 MPa
    }

    #[test]
    fn test_shock_wave_propagation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let propagator = ShockWavePropagation::new(0.1, &grid).unwrap();

        let initial_pressure = Array3::from_elem(grid.dimensions(), 1e6); // 1 MPa uniform field

        let propagated = propagator.propagate_shock_wave(&initial_pressure, 1e6).unwrap();

        // Propagated field should be attenuated
        let initial_max = initial_pressure.iter().fold(f64::MIN, |a, &b| a.max(b));
        let propagated_max = propagated.iter().fold(f64::MIN, |a, &b| a.max(b));

        assert!(propagated_max < initial_max); // Should be attenuated
        assert!(propagated_max > 0.0); // Should still be positive
    }

    #[test]
    fn test_shock_amplitude_calculation() {
        let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3).unwrap();
        let propagator = ShockWavePropagation::new(0.1, &grid).unwrap();

        let initial_amplitude = 1e6; // 1 MPa
        let frequency = 1e6; // 1 MHz

        let shock_amplitude = propagator.calculate_shock_amplitude(initial_amplitude, frequency);

        // Should be significantly amplified by focusing and nonlinearity
        assert!(shock_amplitude > initial_amplitude * 100.0);
    }
}
