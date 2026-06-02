//! `run_theranostic_inverse_from_ritk` pyfunction.

use kwavers::clinical::imaging::reconstruction::transcranial_ust::{
    resample_head_slice, select_head_slice,
};
use kwavers::clinical::therapy::theranostic_guidance::{
    build_abdominal_placement_context, build_brain_placement_context, prepare_abdominal_slice,
    prepare_brain_slice, run_theranostic_inverse,
    synthetic::{
        synthetic_abdominal_kidney_phantom, synthetic_abdominal_liver_phantom,
        synthetic_brain_phantom,
    },
    AnatomyKind, BrainTargetSelection, PassiveReconstructionMode, TheranosticInverseConfig,
    TransmitScheduleConfig, TransmitScheduleStrategy, WaveformMisfit,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::Path;

use super::super::helpers::{kwavers_to_py, labels_from_volume};
use super::result_serializer::{brain_target_index, resampled_crop_index_xy, result_to_dict};
use crate::ritk_image::load_ritk_nifti;

#[pyfunction]
#[pyo3(signature = (
    ct_nifti_path,
    segmentation_nifti_path = None,
    anatomy = "brain",
    grid_size = 64,
    element_count = None,
    iterations = 12,
    frequencies_hz = None,
    receiver_offsets = None,
    source_pressure_pa = None,
    target_fraction_xyz = None,
    noise_fraction = 0.012,
    inverse_encoding_rows_per_code = 2,
    transmit_schedule_strategy = "full",
    transmit_budget = None,
    elastic_fwi_iterations = 3,
    waveform_misfit = "charbonnier",
    waveform_misfit_scale_fraction = 0.012,
    passive_reconstruction = "operator"
))]
#[allow(clippy::too_many_arguments)]
pub fn run_theranostic_inverse_from_ritk<'py>(
    py: Python<'py>,
    ct_nifti_path: &str,
    segmentation_nifti_path: Option<&str>,
    anatomy: &str,
    grid_size: usize,
    element_count: Option<usize>,
    iterations: usize,
    frequencies_hz: Option<Vec<f64>>,
    receiver_offsets: Option<Vec<usize>>,
    source_pressure_pa: Option<f64>,
    target_fraction_xyz: Option<(f64, f64, f64)>,
    noise_fraction: f64,
    inverse_encoding_rows_per_code: usize,
    transmit_schedule_strategy: &str,
    transmit_budget: Option<usize>,
    elastic_fwi_iterations: usize,
    waveform_misfit: &str,
    waveform_misfit_scale_fraction: f64,
    passive_reconstruction: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let anatomy = AnatomyKind::from_name(anatomy).map_err(kwavers_to_py)?;
    let ct_path = Path::new(ct_nifti_path);
    let (ct, spacing_mm) = if ct_path.exists() {
        let (mut ct, spacing_mm) = load_ritk_nifti(ct_path)?;
        ct.mapv_inplace(|hu| hu.clamp(-1024.0, 3071.0));
        (ct, spacing_mm)
    } else {
        match anatomy {
            AnatomyKind::Brain => synthetic_brain_phantom(),
            AnatomyKind::Liver => {
                let (ct, _, spacing_mm) = synthetic_abdominal_liver_phantom();
                (ct, spacing_mm)
            }
            AnatomyKind::Kidney => {
                let (ct, _, spacing_mm) = synthetic_abdominal_kidney_phantom();
                (ct, spacing_mm)
            }
        }
    };
    let mut config = TheranosticInverseConfig::new(anatomy);
    config.grid_size = grid_size;
    config.iterations = iterations;
    config.noise_fraction = noise_fraction;
    config.inverse_encoding_rows_per_code = inverse_encoding_rows_per_code;
    config.elastic_fwi_iterations = elastic_fwi_iterations;
    config.transmit_schedule = TransmitScheduleConfig {
        strategy: TransmitScheduleStrategy::from_name(transmit_schedule_strategy)
            .map_err(kwavers_to_py)?,
        budget: transmit_budget,
    };
    config.waveform_misfit = WaveformMisfit::from_name(waveform_misfit)
        .ok_or_else(|| PyValueError::new_err("waveform_misfit must be 'charbonnier' or 'l2'"))?;
    config.waveform_misfit_scale_fraction = waveform_misfit_scale_fraction;
    config.passive_reconstruction =
        PassiveReconstructionMode::from_name(passive_reconstruction).map_err(kwavers_to_py)?;
    if let Some(count) = element_count {
        config.element_count = count;
    }
    if let Some(freqs) = frequencies_hz {
        config.frequencies_hz = freqs;
    }
    if let Some(offsets) = receiver_offsets {
        config.receiver_offsets = offsets;
    }
    if let Some(pressure) = source_pressure_pa {
        config.source_pressure_pa = pressure;
    }

    let (prepared, placement_context) = match anatomy {
        AnatomyKind::Brain => {
            let target_fraction = target_fraction_xyz.map(|(x, y, z)| [x, y, z]);
            let target_index = if let Some(fraction) = target_fraction {
                Some(brain_target_index(&ct, fraction).map_err(kwavers_to_py)?)
            } else {
                None
            };
            let selected = if let Some(index) = target_index {
                index[2]
            } else {
                select_head_slice(&ct).map_err(kwavers_to_py)?
            };
            let resampled =
                resample_head_slice(&ct, spacing_mm, selected, grid_size).map_err(kwavers_to_py)?;
            let crop_bounds_index = resampled.crop_bounds_index;
            let source_dimensions = resampled.source_dimensions;
            let source_spacing_m = resampled.source_spacing_m;
            let target_selection =
                target_index.map_or(BrainTargetSelection::OrganCentroid, |index| {
                    BrainTargetSelection::ResampledIndex(resampled_crop_index_xy(
                        index,
                        crop_bounds_index,
                        grid_size,
                    ))
                });
            let placement_context =
                build_brain_placement_context(&ct, spacing_mm, selected, &config, target_fraction)
                    .map_err(kwavers_to_py)?;
            let mut prepared = prepare_brain_slice(
                resampled.hu,
                resampled.spacing_m,
                selected,
                target_selection,
            )
            .map_err(kwavers_to_py)?;
            prepared.source_dimensions = source_dimensions;
            prepared.source_spacing_m = source_spacing_m;
            prepared.crop_bounds_index = crop_bounds_index;
            (prepared, placement_context)
        }
        AnatomyKind::Liver | AnatomyKind::Kidney => {
            // Load segmentation labels: from NIfTI when available, else synthetic.
            let labels = if let Some(seg_path_str) = segmentation_nifti_path {
                let seg_path = Path::new(seg_path_str);
                if seg_path.exists() {
                    let (seg, _) = load_ritk_nifti(seg_path)?;
                    labels_from_volume(seg)
                } else {
                    // Paired synthetic segmentation matching the CT phantom.
                    match anatomy {
                        AnatomyKind::Liver => synthetic_abdominal_liver_phantom().1,
                        AnatomyKind::Kidney => synthetic_abdominal_kidney_phantom().1,
                        _ => unreachable!(),
                    }
                }
            } else if !ct_path.exists() {
                // No seg path provided and CT is also synthetic: use synthetic label.
                match anatomy {
                    AnatomyKind::Liver => synthetic_abdominal_liver_phantom().1,
                    AnatomyKind::Kidney => synthetic_abdominal_kidney_phantom().1,
                    _ => unreachable!(),
                }
            } else {
                return Err(PyValueError::new_err(
                    "segmentation_nifti_path is required for liver and kidney when using real CT",
                ));
            };
            let placement_context =
                build_abdominal_placement_context(anatomy, &ct, &labels, spacing_mm, &config)
                    .map_err(kwavers_to_py)?;
            (
                prepare_abdominal_slice(anatomy, &ct, &labels, spacing_mm, grid_size)
                    .map_err(kwavers_to_py)?,
                placement_context,
            )
        }
    };
    let result = py
        .detach(|| run_theranostic_inverse(prepared, &config))
        .map_err(kwavers_to_py)?;
    result_to_dict(py, result, &config, placement_context, target_fraction_xyz)
}
