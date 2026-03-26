//! Core cavitation mechanics functionality
//!
//! This module provides fundamental cavitation detection and modeling
//! based on acoustic pressure thresholds and bubble dynamics.

pub mod growth;
pub mod model;
pub mod state;
pub mod thresholds;

pub use growth::rectified_diffusion_rate;
pub use model::{CavitationCore, CavitationModel};
pub use state::{CavitationDose, CavitationState};
pub use thresholds::{
    blake_threshold, flynn_criterion, flynn_threshold, mechanical_index, neppiras_threshold,
    ThresholdModel,
};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    #[test]
    fn test_threshold_calculations() {
        let p0 = 101325.0;
        let pv = 2339.0;
        let sigma = 0.0728;
        let r0 = 1e-6;

        let blake = blake_threshold(sigma, r0, p0, pv);
        assert!(blake < p0 + pv);

        let neppiras = neppiras_threshold(p0, pv, sigma, r0);
        assert!(neppiras > 0.0);

        let mi = mechanical_index(-1.0e6, 1.0e6);
        assert_relative_eq!(mi, 1.0);
    }

    #[test]
    fn test_cavitation_model_update() {
        let mut model = CavitationModel::new((5, 5, 5));
        model.threshold_model = ThresholdModel::MechanicalIndex;

        let mut pressure = Array3::from_elem((5, 5, 5), 101325.0);
        pressure[[2, 2, 2]] = -1.0e6; // Beyond typical threshold

        model.update_states(&pressure, 1.0e6, 1e-6, 1e-6);

        assert!(model.states[[2, 2, 2]].is_cavitating);
        assert!(!model.states[[0, 0, 0]].is_cavitating);
        assert_relative_eq!(model.states[[2, 2, 2]].mechanical_index, 1.0);
    }
}
