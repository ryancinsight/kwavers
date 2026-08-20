//! Synthetic and NIfTI-backed skull medium construction.

use super::config::GridSpec;
use super::seismic_medium::SkullModel;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;
use ritk_io::domain::ImageReader;

/// Build the analytical coronal skull phantom used by the example.
pub(crate) fn build_skull_phantom() -> KwaversResult<SkullModel> {
    let cx = (GridSpec::NX / 2) as f64;
    let cz = (GridSpec::NZ / 2) as f64;
    let mut hu = Array3::<f64>::from_elem(
        (GridSpec::NX, GridSpec::NY, GridSpec::NZ),
        GridSpec::HU_WATER,
    );

    for i in 0..GridSpec::NX {
        for k in 0..GridSpec::NZ {
            let dx = i as f64 - cx;
            let dz = k as f64 - cz;
            let radius = (dx * dx + dz * dz).sqrt();
            let voxel_hu = if radius > GridSpec::R_HEAD {
                GridSpec::HU_WATER
            } else if radius > GridSpec::R_SKULL_OUT {
                GridSpec::HU_SCALP
            } else if radius > GridSpec::R_DIPLOE {
                GridSpec::HU_CORTICAL_OUT
            } else if radius > GridSpec::R_SKULL_IN {
                GridSpec::HU_DIPLOE
            } else if radius > GridSpec::R_BRAIN {
                GridSpec::HU_CORTICAL_IN
            } else {
                GridSpec::HU_BRAIN
            };

            for j in 0..GridSpec::NY {
                hu[[i, j, k]] = voxel_hu;
            }
        }
    }

    SkullModel::from_hu(hu)
}

/// Load a coronal CT slice from a NIfTI file and resample it to the demo grid.
pub(crate) fn load_ct_slice(
    ct_nifti_path: &str,
    _mri_nifti_path: &str,
    _slice_index: usize,
) -> KwaversResult<SkullModel> {
    use ritk_io::format::nifti::native::NiftiReader;

    let reader = NiftiReader::new(coeus_core::SequentialBackend);
    let object = reader.read(ct_nifti_path).map_err(|error| {
        KwaversError::InvalidInput(format!(
            "CT NIfTI read failed for {ct_nifti_path}: {error:?}"
        ))
    })?;
    let dimensions = object.shape();
    if dimensions.contains(&0) {
        return Err(KwaversError::InvalidInput(format!(
            "CT NIfTI has an empty dimension: {dimensions:?}"
        )));
    }

    let values: Vec<f64> = object
        .data_vec()
        .iter()
        .map(|&value| value as f64)
        .collect();
    let (source_nx, source_ny, source_nz) = (dimensions[0], dimensions[1], dimensions[2]);
    let coronal_index = source_ny / 2;
    let scale_x = source_nx as f64 / GridSpec::NX as f64;
    let scale_z = source_nz as f64 / GridSpec::NZ as f64;
    let mut hu = Array3::<f64>::from_elem(
        (GridSpec::NX, GridSpec::NY, GridSpec::NZ),
        GridSpec::HU_WATER,
    );

    for i in 0..GridSpec::NX {
        for k in 0..GridSpec::NZ {
            let source_i = ((i as f64 * scale_x) as usize).min(source_nx - 1);
            let source_k = ((k as f64 * scale_z) as usize).min(source_nz - 1);
            let index = source_i
                .checked_mul(source_ny)
                .and_then(|value| value.checked_mul(source_nz))
                .and_then(|value| value.checked_add(coronal_index * source_nz))
                .and_then(|value| value.checked_add(source_k))
                .ok_or_else(|| KwaversError::InvalidInput("CT voxel index overflow".to_owned()))?;
            let voxel = *values.get(index).ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "CT voxel index {index} exceeds {} loaded values",
                    values.len()
                ))
            })?;
            for j in 0..GridSpec::NY {
                hu[[i, j, k]] = voxel;
            }
        }
    }

    SkullModel::from_hu(hu)
}

#[cfg(test)]
mod tests {
    use super::build_skull_phantom;
    use crate::config::GridSpec;

    #[test]
    fn synthetic_phantom_contains_water_and_brain_regions() {
        let phantom = build_skull_phantom().expect("synthetic phantom dimensions are valid");
        let hu = phantom.hu();
        assert_eq!(hu.shape(), [GridSpec::NX, GridSpec::NY, GridSpec::NZ]);
        assert_eq!(hu[[0, 0, 0]], GridSpec::HU_WATER);
        assert_eq!(
            hu[[GridSpec::NX / 2, 0, GridSpec::NZ / 2]],
            GridSpec::HU_BRAIN
        );
        assert!(hu.iter().any(|&value| value == GridSpec::HU_CORTICAL_OUT));
    }
}
