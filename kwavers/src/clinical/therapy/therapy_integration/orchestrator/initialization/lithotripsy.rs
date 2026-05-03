//! Lithotripsy simulator initialization and stone geometry helpers.

use crate::clinical::therapy::lithotripsy::stone_fracture::StoneMaterial;
use crate::clinical::therapy::lithotripsy::{LithotripsyParameters, LithotripsySimulator};
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
#[cfg(feature = "nifti")]
use crate::domain::imaging::medical::{CTImageLoader, MedicalImageLoader};
use log::{info, warn};
use ndarray::Array3;

use super::super::super::config::TherapySessionConfig;

/// Initialize lithotripsy simulator
///
/// Creates a lithotripsy simulator for kidney stone fragmentation.
///
/// # References
///
/// - Chaussy et al. (1980): "Extracorporeally induced destruction of kidney stones"
/// - ISO 16869:2015: "Lithotripters - Characteristics"
pub fn init_lithotripsy_simulator(
    config: &TherapySessionConfig,
    grid: &Grid,
) -> KwaversResult<LithotripsySimulator> {
    let stone_geometry = create_stone_geometry(config, grid);
    let lithotripsy_params = LithotripsyParameters {
        stone_material: StoneMaterial::calcium_oxalate_monohydrate(),
        shock_parameters: Default::default(),
        cloud_parameters: Default::default(),
        bioeffects_parameters: Default::default(),
        treatment_frequency: config.acoustic_params.prf,
        num_shock_waves: (config.duration * config.acoustic_params.prf) as usize,
        interpulse_delay: 1.0 / config.acoustic_params.prf,
        stone_geometry,
    };
    LithotripsySimulator::new(lithotripsy_params, grid.clone())
}

/// Create stone geometry from target volume via CT imaging workflow.
///
/// Clinical workflow: CT imaging → registration → segmentation → morphological refinement.
fn create_stone_geometry(config: &TherapySessionConfig, grid: &Grid) -> Array3<f64> {
    use crate::physics::acoustics::imaging::registration::ImageRegistration;

    let mut geometry = Array3::zeros(grid.dimensions());

    let ct_data = load_ct_imaging_data(config).unwrap_or_else(|_| generate_synthetic_ct_data(grid));

    let registration = ImageRegistration::default();
    let identity_transform = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let registered_ct = registration.apply_transform(&ct_data, &identity_transform);

    let stone_threshold_hu = 200.0;
    let stone_geometry = segment_stone_from_ct(&registered_ct, stone_threshold_hu, grid);
    let refined_geometry = morphological_refinement(&stone_geometry, grid);

    for i in 0..grid.dimensions().0 {
        for j in 0..grid.dimensions().1 {
            for k in 0..grid.dimensions().2 {
                geometry[[i, j, k]] = refined_geometry[[i, j, k]];
            }
        }
    }

    geometry
}

/// Load CT imaging data from medical imaging sources.
///
/// Returns error to trigger fallback to synthetic data when CT unavailable.
fn load_ct_imaging_data(config: &TherapySessionConfig) -> KwaversResult<Array3<f64>> {
    if let Some(ct_path) = &config.imaging_data_path {
        if ct_path.ends_with(".nii") || ct_path.ends_with(".nii.gz") {
            #[cfg(feature = "nifti")]
            {
                let mut loader = CTImageLoader::new();
                match loader.load(ct_path) {
                    Ok(ct_data) => {
                        let metadata = loader.ct_metadata().unwrap();
                        info!(
                            "Loaded CT scan: {} voxels, {:.2}mm spacing, HU range [{:.0}, {:.0}]",
                            format_args!(
                                "{}×{}×{}",
                                metadata.dimensions.0,
                                metadata.dimensions.1,
                                metadata.dimensions.2
                            ),
                            metadata.voxel_spacing_mm.0,
                            metadata.hu_range.0,
                            metadata.hu_range.1
                        );
                        return Ok(ct_data);
                    }
                    Err(e) => {
                        warn!(
                            "Failed to load NIFTI CT data: {}. Using synthetic fallback.",
                            e
                        );
                    }
                }
            }

            #[cfg(not(feature = "nifti"))]
            {
                warn!(
                    "NIFTI feature not enabled. Rebuild with --features nifti to load {}. \
                     Using synthetic fallback.",
                    ct_path
                );
            }
        } else if ct_path.ends_with(".dcm") {
            warn!(
                "DICOM loading via ritk-io not wired into therapy orchestrator yet for {}. \
                 Use ritk_io::load_dicom_series directly (see examples/skull_ct_phase_correction.rs) \
                 or supply a NIFTI path. Using synthetic fallback.",
                ct_path
            );
        }
    }

    Err(crate::core::error::KwaversError::Validation(
        crate::core::error::ValidationError::InvalidValue {
            parameter: "CT imaging data".to_string(),
            value: 0.0,
            reason: "CT data loading from DICOM requires the ritk-io bridge \
                     (see backlog 'DICOM SSOT consolidation'); supply a NIFTI \
                     path or call ritk_io::load_dicom_series directly."
                .to_string(),
        },
    ))
}

/// Generate synthetic CT data for testing when real CT data unavailable.
///
/// HU values: kidney tissue ~40, calcium oxalate stone ~1200, background ~-200.
///
/// # References
///
/// - Williams et al. (2010): "Characterization of kidney stone composition"
fn generate_synthetic_ct_data(grid: &Grid) -> Array3<f64> {
    let (nx, ny, nz) = grid.dimensions();
    let mut ct_data = Array3::zeros((nx, ny, nz));

    let center_x = nx / 2;
    let center_y = ny / 2;
    let center_z = nz / 2;

    let kidney_a = nx as f64 * 0.3;
    let kidney_b = ny as f64 * 0.2;
    let kidney_c = nz as f64 * 0.15;

    let stone_center_x = center_x + 10;
    let stone_center_y = center_y + 5;
    let stone_center_z = center_z;
    let stone_radius = 3.0;

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let dx_kidney = (i as f64 - center_x as f64) / kidney_a;
                let dy_kidney = (j as f64 - center_y as f64) / kidney_b;
                let dz_kidney = (k as f64 - center_z as f64) / kidney_c;

                if dx_kidney * dx_kidney + dy_kidney * dy_kidney + dz_kidney * dz_kidney <= 1.0 {
                    ct_data[[i, j, k]] = 40.0;

                    let dx_stone = i as f64 - stone_center_x as f64;
                    let dy_stone = j as f64 - stone_center_y as f64;
                    let dz_stone = k as f64 - stone_center_z as f64;
                    let dist =
                        (dx_stone * dx_stone + dy_stone * dy_stone + dz_stone * dz_stone).sqrt();

                    if dist <= stone_radius {
                        ct_data[[i, j, k]] = 1200.0;
                    }
                } else {
                    ct_data[[i, j, k]] = -200.0;
                }
            }
        }
    }

    ct_data
}

/// Segment stone from CT data using HU-based thresholding.
///
/// # References
///
/// - Williams et al. (2010): "Characterization of kidney stone composition"
fn segment_stone_from_ct(ct_data: &Array3<f64>, threshold_hu: f64, grid: &Grid) -> Array3<f64> {
    let (nx, ny, nz) = grid.dimensions();
    let mut stone_mask = Array3::zeros((nx, ny, nz));

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if ct_data[[i, j, k]] >= threshold_hu {
                    stone_mask[[i, j, k]] = 1.0;
                }
            }
        }
    }

    stone_mask
}

/// Apply morphological closing to refine stone boundary.
///
/// Fills voxels with 4+ stone face-neighbors to smooth segmentation boundaries.
fn morphological_refinement(stone_mask: &Array3<f64>, grid: &Grid) -> Array3<f64> {
    let (nx, ny, nz) = grid.dimensions();
    let mut refined = stone_mask.clone();

    for i in 1..(nx - 1) {
        for j in 1..(ny - 1) {
            for k in 1..(nz - 1) {
                let is_stone = stone_mask[[i, j, k]] > 0.5;
                let neighbors = [
                    stone_mask[[i - 1, j, k]],
                    stone_mask[[i + 1, j, k]],
                    stone_mask[[i, j - 1, k]],
                    stone_mask[[i, j + 1, k]],
                    stone_mask[[i, j, k - 1]],
                    stone_mask[[i, j, k + 1]],
                ];
                let stone_neighbors = neighbors.iter().filter(|&&n| n > 0.5).count();

                if is_stone || stone_neighbors >= 4 {
                    refined[[i, j, k]] = 1.0;
                }
            }
        }
    }

    refined
}
