//! Clinical adapter for whole-breast multi-row ring UST FWI.
//!
//! The solver module owns numerical inversion. This adapter is the clinical
//! reconstruction boundary: it names the breast UST use case, delegates to the
//! solver, and returns clinical audit metadata without duplicating physics or
//! optimization logic.

mod dataset;
mod diagnostics;
mod direct_field;
mod phantom_hdf5;
mod phantom_mat5;
mod phantom_types;
mod reduction;

pub use dataset::{
    generate_breast_ust_pstd_frequency_dataset, snap_multi_row_ring_array_to_grid,
    BreastUstPstdDataset, BreastUstPstdDatasetConfig, BREAST_UST_PSTD_DATASET_MODEL,
};
pub use diagnostics::{
    acquisition_identifiability, diagnose_breast_ust_observation_pair,
    forward_operator_equivalence_diagnostics,
    forward_operator_equivalence_diagnostics_with_receiver_policy, passive_receiver_mask,
    reconstruction_metrics, scaled_observation_residual_metrics, sine_frequency_bin_coefficient,
    source_channel_residual_diagnostics, source_excitation_diagnostics, source_receiver_mask,
    table1_parity, BreastUstAcquisitionIdentifiability,
    BreastUstForwardOperatorEquivalenceDiagnostics, BreastUstForwardOperatorModelDiagnostics,
    BreastUstForwardOperatorPrediction, BreastUstObservationPairDiagnostics,
    BreastUstReceiverChannelPolicy, BreastUstReconstructionMetrics,
    BreastUstScaledObservationResidualMetrics, BreastUstSourceChannelResidualDiagnostics,
    BreastUstSourceExcitationDiagnostics, BreastUstSourceExcitationFrequencyDiagnostics,
    BreastUstSourceScalingPolicy, BreastUstTable1Parity,
};
pub use direct_field::{
    diagnose_breast_ust_homogeneous_direct_field, BreastUstDirectFieldDiagnostics,
    BreastUstHomogeneousDirectFieldDiagnostics,
    BREAST_UST_HOMOGENEOUS_DIRECT_FIELD_DIAGNOSTIC_MODEL,
};
pub use phantom_hdf5::{
    load_ali_2025_breast_phantom_from_hdf5, load_ali_2025_breast_phantom_from_hdf5_with_config,
    BreastUstAliPhantomHdf5Config, BREAST_UST_ALI_2025_PHANTOM_MODEL,
};
pub use phantom_mat5::{
    load_ali_2025_breast_phantom_from_mat5, load_ali_2025_breast_phantom_from_mat5_with_config,
    BreastUstAliPhantomMat5Config, BreastUstMriBreastSide, BREAST_UST_ALI_2025_MAT5_PHANTOM_MODEL,
};
pub use phantom_types::{
    BreastUstAliPhantom, BreastUstPhantomStorageOrder, BreastUstSoundSpeedUnit,
};
pub use reduction::{
    derive_reduced_breast_ust_array_geometry, prepare_reduced_breast_ust_phantom,
    BreastUstReducedArrayGeometry, BreastUstReducedPhantom,
};

use crate::core::error::KwaversError;
use crate::core::error::KwaversResult;
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use crate::solver::inverse::fwi::frequency_domain::{
    self, Config as FrequencyDomainFwiConfig, FrequencyObservation,
};
use ndarray::Array3;
use std::io::Read;
use std::path::Path;

pub const BREAST_UST_FWI_MODEL: &str = "clinical_breast_ust_multi_row_ring_fwi";

/// External phantom file format selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreastUstAliPhantomFileFormat {
    /// Infer the format from file signatures.
    Auto,
    /// HDF5 or MAT-v7.3/HDF5 sound-speed container.
    Hdf5,
    /// MATLAB Level-5 MRI container from the published GitHub release.
    Mat5,
}

/// Unified phantom ingest configuration.
#[derive(Clone, Debug)]
pub struct BreastUstAliPhantomLoadConfig {
    pub format: BreastUstAliPhantomFileFormat,
    pub hdf5: BreastUstAliPhantomHdf5Config,
    pub mat5: BreastUstAliPhantomMat5Config,
}

impl Default for BreastUstAliPhantomLoadConfig {
    fn default() -> Self {
        Self {
            format: BreastUstAliPhantomFileFormat::Auto,
            hdf5: BreastUstAliPhantomHdf5Config::default(),
            mat5: BreastUstAliPhantomMat5Config::default(),
        }
    }
}

/// Load an Ali et al. phantom from either HDF5/MAT-v7.3 or MATLAB-5.
///
/// # Errors
/// Returns an error when the selected or detected external container violates
/// the corresponding clinical ingest contract.
pub fn load_ali_2025_breast_phantom_with_config<P: AsRef<Path>>(
    path: P,
    config: BreastUstAliPhantomLoadConfig,
) -> KwaversResult<BreastUstAliPhantom> {
    let path_ref = path.as_ref();
    match config.format {
        BreastUstAliPhantomFileFormat::Hdf5 => {
            load_ali_2025_breast_phantom_from_hdf5_with_config(path_ref, config.hdf5)
        }
        BreastUstAliPhantomFileFormat::Mat5 => {
            load_ali_2025_breast_phantom_from_mat5_with_config(path_ref, config.mat5)
        }
        BreastUstAliPhantomFileFormat::Auto => match detect_phantom_format(path_ref)? {
            BreastUstAliPhantomFileFormat::Hdf5 => {
                load_ali_2025_breast_phantom_from_hdf5_with_config(path_ref, config.hdf5)
            }
            BreastUstAliPhantomFileFormat::Mat5 => {
                load_ali_2025_breast_phantom_from_mat5_with_config(path_ref, config.mat5)
            }
            BreastUstAliPhantomFileFormat::Auto => unreachable!(),
        },
    }
}

fn detect_phantom_format(path: &Path) -> KwaversResult<BreastUstAliPhantomFileFormat> {
    let mut file = std::fs::File::open(path)?;
    let mut header = [0u8; 128];
    let read = file.read(&mut header)?;
    if read >= 8 && &header[..8] == b"\x89HDF\r\n\x1a\n" {
        return Ok(BreastUstAliPhantomFileFormat::Hdf5);
    }
    if read >= 128 && &header[..19] == b"MATLAB 5.0 MAT-file" {
        return Ok(BreastUstAliPhantomFileFormat::Mat5);
    }
    Err(KwaversError::InvalidInput(format!(
        "unsupported Ali 2025 phantom container: {}",
        path.display()
    )))
}

/// Clinical reconstruction result for breast UST sound speed.
#[derive(Clone, Debug)]
pub struct BreastUstFwiImage {
    /// Reconstructed whole-breast sound speed [m/s].
    pub sound_speed_m_s: Array3<f64>,
    /// Objective value after the initial model and each accepted solver update.
    pub objective_history: Vec<f64>,
    /// Frequencies used by the reconstruction.
    pub frequencies_used: usize,
    /// Cylindrical-wave transmissions used per frequency.
    pub transmissions_used: usize,
    /// Receivers sampled per cylindrical-wave transmission.
    pub receivers_used: usize,
    /// Clinical model identifier.
    pub model_family: &'static str,
    /// Solver model identifier.
    pub solver_model_family: &'static str,
}

/// Reconstruct a whole-breast sound-speed volume from multi-row ring data.
///
/// # Errors
/// Returns an error when observations, geometry, initial model, or config values
/// violate the solver contract.
pub fn reconstruct_breast_ust_sound_speed_volume(
    observations: &[FrequencyObservation],
    array: &MultiRowRingArray,
    initial_sound_speed_m_s: &Array3<f64>,
    config: &FrequencyDomainFwiConfig,
) -> KwaversResult<BreastUstFwiImage> {
    let solver_result =
        frequency_domain::invert(observations, array, initial_sound_speed_m_s, config)?;

    Ok(BreastUstFwiImage {
        sound_speed_m_s: solver_result.sound_speed_m_s,
        objective_history: solver_result.objective_history,
        frequencies_used: solver_result.frequencies_used,
        transmissions_used: solver_result.transmissions_used,
        receivers_used: solver_result.receivers_used,
        model_family: BREAST_UST_FWI_MODEL,
        solver_model_family: solver_result.model_family,
    })
}

#[cfg(test)]
mod tests {
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use super::*;
    use crate::solver::inverse::fwi::frequency_domain::{
        simulate_frequency_observation, SingleScatterBornOperator,
    };
    use std::sync::Arc;

    #[test]
    fn clinical_adapter_delegates_to_solver_and_preserves_metadata() {
        let array = MultiRowRingArray::new(5, 2, 0.07, 0.01).expect("array");
        let config = FrequencyDomainFwiConfig {
            reference_sound_speed_m_s: SOUND_SPEED_WATER_SIM,
            spacing_m: 0.005,
            iterations: 2,
            initial_step_s_per_m: 2.0e-6,
            min_sound_speed_m_s: 1450.0,
            max_sound_speed_m_s: 1560.0,
            estimate_source_scaling: false,
            tikhonov_weight: 0.0,
            forward_operator: Arc::new(SingleScatterBornOperator),
        };
        let mut truth = Array3::from_elem((2, 2, 2), SOUND_SPEED_WATER_SIM);
        truth[[1, 1, 1]] = 1525.0;
        let observed =
            simulate_frequency_observation(&truth, &array, 230_000.0, &config).expect("observed");
        let observations = [FrequencyObservation::new(
            230_000.0,
            observed.slice(ndarray::s![0..3, ..]).to_owned(),
        )];
        let initial = Array3::from_elem((2, 2, 2), SOUND_SPEED_WATER_SIM);

        let result =
            reconstruct_breast_ust_sound_speed_volume(&observations, &array, &initial, &config)
                .expect("clinical reconstruction");

        assert_eq!(result.model_family, BREAST_UST_FWI_MODEL);
        assert_eq!(result.frequencies_used, 1);
        assert_eq!(result.transmissions_used, 3);
        assert_eq!(result.receivers_used, array.element_count());
        assert!(!result.objective_history.is_empty());
    }
}
