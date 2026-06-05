//! MATLAB Level-5 MRI phantom ingest for Ali et al. 2025 breast UST FWI.
//!
//! The published release asset `BreastPhantomFromMRI.mat` is not HDF5. It is a
//! MATLAB 5.0 file containing a compressed `breast_mri` volume. This module is
//! the clinical anti-corruption layer that decodes that file and applies the
//! paper's MRI-to-sound-speed mapping on a requested uniform grid.

mod convert;
mod parse;

#[cfg(test)]
mod tests;

use crate::reconstruction::breast_ust_fwi::phantom_mat5::convert::{
    mri_to_sound_speed, BreastMriSoundSpeedMapConfig,
};
use crate::reconstruction::breast_ust_fwi::phantom_mat5::parse::{
    read_mat5_numeric_volume, Mat5NumericVolume,
};
use crate::reconstruction::breast_ust_fwi::phantom_types::{
    BreastUstAliPhantom, BreastUstPhantomStorageOrder,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use std::path::Path;

pub const BREAST_UST_ALI_2025_MAT5_PHANTOM_MODEL: &str =
    "clinical_breast_ust_ali_2025_mat5_mri_phantom";

const DEFAULT_MRI_VARIABLE_NAME: &str = "breast_mri";
const PAPER_GRID_SPACING_M: f64 = 0.4e-3;

/// Breast side selected by the published MRI-to-sound-speed transform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreastUstMriBreastSide {
    /// Published left-breast transform: theta=100°, phi=90°, c_min=1400 m/s.
    Left,
    /// Published right-breast transform: theta=80°, phi=80°, c_min=1350 m/s.
    Right,
}

impl BreastUstMriBreastSide {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Left => "left",
            Self::Right => "right",
        }
    }
}

/// MATLAB-5 MRI phantom ingest configuration.
#[derive(Clone, Debug)]
pub struct BreastUstAliPhantomMat5Config {
    /// Variable name inside the MAT-file.
    pub mri_variable_name: String,
    /// Output grid shape `(nx, ny, nz)` used for the sound-speed map.
    pub output_shape: [usize; 3],
    /// Uniform output grid spacing [m].
    pub grid_spacing_m: f64,
    /// Breast side transform parameters from the published MATLAB function.
    pub breast_side: BreastUstMriBreastSide,
    /// MRI intensity threshold used before per-slice hole filling.
    pub tissue_threshold: f64,
}

impl Default for BreastUstAliPhantomMat5Config {
    fn default() -> Self {
        Self {
            mri_variable_name: DEFAULT_MRI_VARIABLE_NAME.to_owned(),
            output_shape: [192, 192, 96],
            grid_spacing_m: PAPER_GRID_SPACING_M,
            breast_side: BreastUstMriBreastSide::Right,
            tissue_threshold: 40.0,
        }
    }
}

/// Load the published MATLAB-5 MRI phantom using default reduced-grid rules.
///
/// # Errors
/// Returns an error if the file is not MATLAB Level-5, the MRI variable is not
/// a real numeric 3-D array, or the MRI-to-sound-speed domain is degenerate.
pub fn load_ali_2025_breast_phantom_from_mat5<P: AsRef<Path>>(
    path: P,
) -> KwaversResult<BreastUstAliPhantom> {
    load_ali_2025_breast_phantom_from_mat5_with_config(
        path,
        BreastUstAliPhantomMat5Config::default(),
    )
}

/// Load the published MATLAB-5 MRI phantom on a requested uniform grid.
///
/// # Theorem
/// The transform matches the published `BreastPhantomVolumetricMRI` pipeline at
/// the domain level: centered metric coordinates are rotated into MRI space,
/// MRI intensity is cubic-interpolated, tissue is thresholded with per-slice
/// hole filling, and tissue intensities are affinely mapped into the paper's
/// side-specific sound-speed interval while exterior voxels become water.
pub fn load_ali_2025_breast_phantom_from_mat5_with_config<P: AsRef<Path>>(
    path: P,
    config: BreastUstAliPhantomMat5Config,
) -> KwaversResult<BreastUstAliPhantom> {
    validate_config(&config)?;
    let path_ref = path.as_ref();
    let Mat5NumericVolume { dims, values, name } =
        read_mat5_numeric_volume(path_ref, &config.mri_variable_name)?;
    let sound_speed_m_s = mri_to_sound_speed(
        dims,
        &values,
        BreastMriSoundSpeedMapConfig {
            output_shape: config.output_shape,
            grid_spacing_m: config.grid_spacing_m,
            breast_side: config.breast_side,
            tissue_threshold: config.tissue_threshold,
        },
    )?;

    Ok(BreastUstAliPhantom {
        sound_speed_m_s,
        spacing_m: config.grid_spacing_m,
        dataset_path: name,
        source_path: path_ref.to_path_buf(),
        storage_order: BreastUstPhantomStorageOrder::FortranContiguous,
        model_family: BREAST_UST_ALI_2025_MAT5_PHANTOM_MODEL,
    })
}

fn validate_config(config: &BreastUstAliPhantomMat5Config) -> KwaversResult<()> {
    if config.mri_variable_name.trim().is_empty() {
        return Err(KwaversError::InvalidInput(
            "mri_variable_name must not be empty".to_owned(),
        ));
    }
    if config.output_shape.contains(&0) {
        return Err(KwaversError::InvalidInput(format!(
            "mat5 output_shape entries must be positive, got {:?}",
            config.output_shape
        )));
    }
    if !config.grid_spacing_m.is_finite() || config.grid_spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "mat5 grid_spacing_m must be positive and finite, got {}",
            config.grid_spacing_m
        )));
    }
    if !config.tissue_threshold.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "mat5 tissue_threshold must be finite, got {}",
            config.tissue_threshold
        )));
    }
    Ok(())
}
