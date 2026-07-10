//! PyO3 wrapper for Rust-owned Ali 2025 breast phantom HDF5 ingest.

use super::complex_compat::leto3_to_nd3;
use super::helpers::kwavers_to_py;
use kwavers_diagnostics::reconstruction::breast_ust_fwi::{
    load_ali_2025_breast_phantom_with_config, BreastUstAliPhantomFileFormat,
    BreastUstAliPhantomHdf5Config, BreastUstAliPhantomLoadConfig, BreastUstAliPhantomMat5Config,
    BreastUstMriBreastSide, BreastUstPhantomStorageOrder, BreastUstSoundSpeedUnit,
};
use numpy::ToPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

#[pyfunction]
#[pyo3(signature = (
    path,
    sound_speed_dataset_path = None,
    spacing_m = None,
    sound_speed_unit = "meters_per_second",
    storage_order = "fortran_contiguous",
    file_format = "auto",
    mat5_output_shape = None,
    mat5_grid_spacing_m = 4.0e-4,
    mat5_breast_side = "right",
    mat5_mri_variable_name = "breast_mri",
    mat5_tissue_threshold = 40.0
))]
pub fn load_ali_2025_breast_fwi_phantom<'py>(
    py: Python<'py>,
    path: String,
    sound_speed_dataset_path: Option<String>,
    spacing_m: Option<f64>,
    sound_speed_unit: &str,
    storage_order: &str,
    file_format: &str,
    mat5_output_shape: Option<(usize, usize, usize)>,
    mat5_grid_spacing_m: f64,
    mat5_breast_side: &str,
    mat5_mri_variable_name: &str,
    mat5_tissue_threshold: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let path = PathBuf::from(path);
    let unit = parse_sound_speed_unit(sound_speed_unit)?;
    let order = parse_storage_order(storage_order)?;
    let hdf5 = BreastUstAliPhantomHdf5Config {
        sound_speed_dataset_path,
        spacing_m,
        sound_speed_unit: unit,
        storage_order: order,
    };
    let shape = mat5_output_shape.unwrap_or((192, 192, 96));
    let mat5 = BreastUstAliPhantomMat5Config {
        mri_variable_name: mat5_mri_variable_name.to_owned(),
        output_shape: [shape.0, shape.1, shape.2],
        grid_spacing_m: spacing_m.unwrap_or(mat5_grid_spacing_m),
        breast_side: parse_breast_side(mat5_breast_side)?,
        tissue_threshold: mat5_tissue_threshold,
    };
    let config = BreastUstAliPhantomLoadConfig {
        format: parse_file_format(file_format)?,
        hdf5,
        mat5,
    };
    let phantom = py
        .detach(|| load_ali_2025_breast_phantom_with_config(&path, config))
        .map_err(kwavers_to_py)?;

    let out = PyDict::new(py);
    out.set_item(
        "sound_speed_m_s",
        leto3_to_nd3(phantom.sound_speed_m_s).to_pyarray(py),
    )?;
    out.set_item("spacing_m", phantom.spacing_m)?;
    out.set_item("dataset_path", phantom.dataset_path)?;
    out.set_item("source_path", phantom.source_path.display().to_string())?;
    out.set_item("storage_order", phantom.storage_order.label())?;
    out.set_item("model_family", phantom.model_family)?;
    Ok(out)
}

fn parse_file_format(format: &str) -> PyResult<BreastUstAliPhantomFileFormat> {
    match format {
        "auto" => Ok(BreastUstAliPhantomFileFormat::Auto),
        "hdf5" => Ok(BreastUstAliPhantomFileFormat::Hdf5),
        "mat5" => Ok(BreastUstAliPhantomFileFormat::Mat5),
        other => Err(PyValueError::new_err(format!(
            "unknown file_format '{other}'"
        ))),
    }
}

fn parse_breast_side(side: &str) -> PyResult<BreastUstMriBreastSide> {
    match side {
        "left" => Ok(BreastUstMriBreastSide::Left),
        "right" => Ok(BreastUstMriBreastSide::Right),
        other => Err(PyValueError::new_err(format!(
            "unknown mat5_breast_side '{other}'"
        ))),
    }
}

fn parse_sound_speed_unit(unit: &str) -> PyResult<BreastUstSoundSpeedUnit> {
    match unit {
        "meters_per_second" => Ok(BreastUstSoundSpeedUnit::MetersPerSecond),
        "kilometers_per_second" => Ok(BreastUstSoundSpeedUnit::KilometersPerSecond),
        other => Err(PyValueError::new_err(format!(
            "unknown sound_speed_unit '{other}'"
        ))),
    }
}

fn parse_storage_order(order: &str) -> PyResult<BreastUstPhantomStorageOrder> {
    match order {
        "c_contiguous" => Ok(BreastUstPhantomStorageOrder::CContiguous),
        "fortran_contiguous" => Ok(BreastUstPhantomStorageOrder::FortranContiguous),
        other => Err(PyValueError::new_err(format!(
            "unknown storage_order '{other}'"
        ))),
    }
}

