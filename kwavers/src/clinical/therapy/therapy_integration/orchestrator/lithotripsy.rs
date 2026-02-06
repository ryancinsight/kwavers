//! Lithotripsy Execution for Kidney Stone Fragmentation
//!
//! This module provides execution logic for lithotripsy therapy, including shock wave
//! delivery and stone fragmentation monitoring. It integrates with the lithotripsy
//! simulator to track treatment progress and stone breakup.
//!
//! ## Lithotripsy
//!
//! Extracorporeal shock wave lithotripsy (ESWL) uses focused shock waves to fragment
//! kidney stones into smaller pieces that can pass naturally. Key aspects:
//!
//! - Shock wave generation and focusing
//! - Stone fragmentation mechanics
//! - Treatment monitoring and dosimetry
//! - Bioeffects management
//!
//! ## References
//!
//! - Chaussy et al. (1980): "Extracorporeally induced destruction of kidney stones"
//! - ISO 16869:2015: "Lithotripters - Characteristics"
//! - Bailey et al. (2005): "Cavitation detection during lithotripsy"

use crate::clinical::therapy::lithotripsy::LithotripsySimulator;
use crate::core::error::KwaversResult;

use super::super::state::AcousticField;

/// Execute lithotripsy simulation step
///
/// Advances the lithotripsy simulator by one time step, delivering shock waves
/// and updating stone fragmentation state. Tracks treatment progress based on
/// the number of shock waves delivered relative to the planned total.
///
/// # Arguments
///
/// - `lithotripsy_simulator`: Lithotripsy simulator with stone geometry and parameters
/// - `acoustic_field`: Current acoustic field (for coupling with stone response)
/// - `dt`: Time step (s)
///
/// # Returns
///
/// Treatment progress (0-1) indicating fraction of planned shock waves delivered
///
/// # Treatment Protocol
///
/// Typical clinical protocol:
/// - 1-2 Hz shock wave repetition rate
/// - 2000-4000 total shock waves per session
/// - 30-60 minute treatment duration
/// - Real-time imaging guidance
///
/// # Future Enhancement
///
/// Will integrate acoustic field coupling:
/// - Stone material response to shock waves
/// - Cavitation cloud formation at stone surface
/// - Fragment trajectory prediction
/// - Bioeffects monitoring (tissue damage, hematoma)
///
/// # References
///
/// - Evan et al. (1998): "Shock wave lithotripsy-induced renal injury"
/// - Cleveland & McAteer (2012): "Physics of shock-wave lithotripsy"
pub fn execute_lithotripsy_step(
    lithotripsy_simulator: &mut LithotripsySimulator,
    _acoustic_field: &AcousticField,
    dt: f64,
) -> KwaversResult<f64> {
    // Advance lithotripsy simulation
    lithotripsy_simulator.advance(dt)?;

    // Calculate treatment progress
    let state = lithotripsy_simulator.current_state();
    let params = lithotripsy_simulator.parameters();
    let progress = state.shock_waves_delivered as f64 / params.num_shock_waves as f64;

    Ok(progress)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clinical::therapy::lithotripsy::stone_fracture::StoneMaterial;
    use crate::clinical::therapy::lithotripsy::LithotripsyParameters;
    use crate::domain::grid::Grid;
    use ndarray::Array3;

    #[test]
    fn test_lithotripsy_step_execution() {
        // Create test grid
        let grid = Grid::new(16, 16, 16, 0.001, 0.001, 0.001).unwrap();

        // Create lithotripsy parameters
        let stone_geometry = Array3::zeros((16, 16, 16));
        let params = LithotripsyParameters {
            stone_material: StoneMaterial::calcium_oxalate_monohydrate(),
            shock_parameters: Default::default(),
            cloud_parameters: Default::default(),
            bioeffects_parameters: Default::default(),
            treatment_frequency: 1.0, // 1 Hz
            num_shock_waves: 100,
            interpulse_delay: 1.0, // 1 second between shocks
            stone_geometry,
        };

        // Create lithotripsy simulator
        let mut simulator = LithotripsySimulator::new(params, grid.clone()).unwrap();

        // Create acoustic field
        let acoustic_field = super::super::super::state::AcousticField {
            pressure: Array3::from_elem((16, 16, 16), 1e7), // 10 MPa shock wave
            velocity_x: Array3::zeros((16, 16, 16)),
            velocity_y: Array3::zeros((16, 16, 16)),
            velocity_z: Array3::zeros((16, 16, 16)),
        };

        // Execute first step
        let progress1 = execute_lithotripsy_step(&mut simulator, &acoustic_field, 1.0).unwrap();
        assert!(progress1 > 0.0);
        assert!(progress1 <= 1.0);

        // Execute second step
        let progress2 = execute_lithotripsy_step(&mut simulator, &acoustic_field, 1.0).unwrap();
        assert!(progress2 > progress1); // Progress should increase
        assert!(progress2 <= 1.0);
    }

    #[test]
    fn test_lithotripsy_progress_tracking() {
        // Create test grid
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();

        // Create lithotripsy parameters with small number of shocks for testing
        let stone_geometry = Array3::zeros((8, 8, 8));
        let params = LithotripsyParameters {
            stone_material: StoneMaterial::calcium_oxalate_monohydrate(),
            shock_parameters: Default::default(),
            cloud_parameters: Default::default(),
            bioeffects_parameters: Default::default(),
            treatment_frequency: 10.0, // 10 Hz (fast for testing)
            num_shock_waves: 10,       // Only 10 shocks
            interpulse_delay: 0.1,
            stone_geometry,
        };

        let mut simulator = LithotripsySimulator::new(params, grid.clone()).unwrap();

        let acoustic_field = super::super::super::state::AcousticField {
            pressure: Array3::from_elem((8, 8, 8), 5e6),
            velocity_x: Array3::zeros((8, 8, 8)),
            velocity_y: Array3::zeros((8, 8, 8)),
            velocity_z: Array3::zeros((8, 8, 8)),
        };

        // Execute multiple steps and verify progress increases
        let mut last_progress = 0.0;
        for _ in 0..5 {
            let progress = execute_lithotripsy_step(&mut simulator, &acoustic_field, 0.1).unwrap();
            assert!(progress >= last_progress); // Progress should never decrease
            assert!(progress <= 1.0); // Progress should not exceed 100%
            last_progress = progress;
        }
    }

    #[test]
    fn test_lithotripsy_completion() {
        // Create test grid
        let grid = Grid::new(8, 8, 8, 0.001, 0.001, 0.001).unwrap();

        // Create lithotripsy parameters with very few shocks
        let stone_geometry = Array3::zeros((8, 8, 8));
        let params = LithotripsyParameters {
            stone_material: StoneMaterial::calcium_oxalate_monohydrate(),
            shock_parameters: Default::default(),
            cloud_parameters: Default::default(),
            bioeffects_parameters: Default::default(),
            treatment_frequency: 100.0, // 100 Hz (very fast for testing)
            num_shock_waves: 5,
            interpulse_delay: 0.01,
            stone_geometry,
        };

        let mut simulator = LithotripsySimulator::new(params, grid.clone()).unwrap();

        let acoustic_field = super::super::super::state::AcousticField {
            pressure: Array3::from_elem((8, 8, 8), 1e7),
            velocity_x: Array3::zeros((8, 8, 8)),
            velocity_y: Array3::zeros((8, 8, 8)),
            velocity_z: Array3::zeros((8, 8, 8)),
        };

        // Execute enough steps to deliver all shocks
        for _ in 0..10 {
            let _ = execute_lithotripsy_step(&mut simulator, &acoustic_field, 0.01);
        }

        // Final progress should be close to or at 100%
        let final_progress =
            execute_lithotripsy_step(&mut simulator, &acoustic_field, 0.01).unwrap();
        assert!(final_progress >= 0.8); // Should be at least 80% complete
    }
}
