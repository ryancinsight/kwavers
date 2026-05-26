//! HDF5 phantom ingest for Ali et al. 2025 breast UST FWI replication.
//!
//! This module is the clinical anti-corruption layer between external
//! HDF5/MAT-v7.3 phantom containers and the solver-owned FWI contracts. It
//! performs storage decoding, unit normalization, shape validation, and grid
//! spacing resolution before any physics or inversion module sees the data.

mod decode;

#[cfg(test)]
mod tests;

use crate::clinical::imaging::reconstruction::breast_ust_fwi::phantom_hdf5::decode::{
    decode_sound_speed_values, read_dataset_payload, validate_sound_speed_domain,
    volume_from_storage_order,
};
use crate::clinical::imaging::reconstruction::breast_ust_fwi::phantom_types::{
    BreastUstAliPhantom, BreastUstPhantomStorageOrder, BreastUstSoundSpeedUnit,
};
use crate::core::error::{KwaversError, KwaversResult};
use consus_core::AttributeValue;
use consus_hdf5::attribute::Hdf5Attribute;
use consus_hdf5::file::Hdf5File;
use std::path::Path;

pub const BREAST_UST_ALI_2025_PHANTOM_MODEL: &str = "clinical_breast_ust_ali_2025_hdf5_phantom";

const SOUND_SPEED_DATASET_CANDIDATES: &[&str] = &[
    "/phantom/sound_speed_m_s",
    "/phantom/sound_speed",
    "/phantom/sos",
    "/model/sound_speed_m_s",
    "/model/sound_speed",
    "/model/sos",
    "/data/sound_speed_m_s",
    "/data/sound_speed",
    "/data/sos",
    "/sound_speed_m_s",
    "/sound_speed",
    "/sos",
    "/SOS",
    "/c",
];

const SPACING_ATTRS_M: &[&str] = &["spacing_m", "voxel_spacing_m", "dx_m", "dxi_m"];
const SPACING_ATTRS_MM: &[&str] = &["spacing_mm", "voxel_spacing_mm", "dx_mm", "dxi_mm"];

/// HDF5 ingest configuration for Ali et al. 2025 phantom volumes.
#[derive(Clone, Debug)]
pub struct BreastUstAliPhantomHdf5Config {
    /// Explicit dataset path. When `None`, known sound-speed dataset names are
    /// searched in deterministic order.
    pub sound_speed_dataset_path: Option<String>,
    /// Uniform voxel spacing [m]. When `None`, spacing must be present as an
    /// HDF5 attribute on the dataset or root group.
    pub spacing_m: Option<f64>,
    /// Unit of the stored dataset values.
    pub sound_speed_unit: BreastUstSoundSpeedUnit,
    /// Linearization order of the raw HDF5 payload.
    pub storage_order: BreastUstPhantomStorageOrder,
}

impl Default for BreastUstAliPhantomHdf5Config {
    fn default() -> Self {
        Self {
            sound_speed_dataset_path: None,
            spacing_m: None,
            sound_speed_unit: BreastUstSoundSpeedUnit::MetersPerSecond,
            storage_order: BreastUstPhantomStorageOrder::FortranContiguous,
        }
    }
}

/// Load an Ali et al. 2025 breast phantom HDF5 file using default ingest rules.
///
/// # Errors
/// Returns an error when the file is invalid HDF5, no sound-speed dataset can
/// be resolved, the dataset is not a numeric 3-D array, or spacing is missing.
pub fn load_ali_2025_breast_phantom_from_hdf5<P: AsRef<Path>>(
    path: P,
) -> KwaversResult<BreastUstAliPhantom> {
    load_ali_2025_breast_phantom_from_hdf5_with_config(
        path,
        BreastUstAliPhantomHdf5Config::default(),
    )
}

/// Load an Ali et al. 2025 breast phantom HDF5 file.
///
/// # Theorem
/// Let `v[a]` be the raw scalar sequence stored by HDF5 and `D=(nx,ny,nz)`.
/// For `CContiguous`, output index `(i,j,k)` receives
/// `v[(i ny + j) nz + k]`. For `FortranContiguous`, output index `(i,j,k)`
/// receives `v[i + nx (j + ny k)]`. These are exactly the C and MATLAB
/// contiguous linearization maps for a rank-3 tensor with dimensions `D`.
///
/// # Errors
/// Returns an error when file, dataset, datatype, unit, storage order, or
/// spacing invariants are violated.
pub fn load_ali_2025_breast_phantom_from_hdf5_with_config<P: AsRef<Path>>(
    path: P,
    config: BreastUstAliPhantomHdf5Config,
) -> KwaversResult<BreastUstAliPhantom> {
    validate_config(&config)?;
    let path_ref = path.as_ref();
    let file = std::fs::File::open(path_ref)?;
    let hdf5 = Hdf5File::open(file)
        .map_err(|err| KwaversError::InvalidInput(format!("HDF5 open failed: {err}")))?;

    let (dataset_path, dataset_addr) = resolve_sound_speed_dataset(&hdf5, &config)?;
    let dataset = hdf5.dataset_at(dataset_addr).map_err(|err| {
        KwaversError::InvalidInput(format!("HDF5 dataset metadata failed: {err}"))
    })?;
    let dims = rank3_dims(&dataset.shape.current_dims())?;
    let raw = read_dataset_payload(&hdf5, dataset_addr, &dataset)?;
    let values = decode_sound_speed_values(&raw, &dataset.datatype, config.sound_speed_unit)?;
    let sound_speed_m_s = volume_from_storage_order(dims, values, config.storage_order)?;
    validate_sound_speed_domain(&sound_speed_m_s)?;
    let spacing_m = resolve_spacing_m(&hdf5, dataset_addr, &config)?;

    Ok(BreastUstAliPhantom {
        sound_speed_m_s,
        spacing_m,
        dataset_path,
        source_path: path_ref.to_path_buf(),
        storage_order: config.storage_order,
        model_family: BREAST_UST_ALI_2025_PHANTOM_MODEL,
    })
}

fn validate_config(config: &BreastUstAliPhantomHdf5Config) -> KwaversResult<()> {
    if let Some(spacing_m) = config.spacing_m {
        validate_spacing(spacing_m)?;
    }
    if let Some(path) = &config.sound_speed_dataset_path {
        if path.trim().is_empty() {
            return Err(KwaversError::InvalidInput(
                "sound_speed_dataset_path must not be empty".to_owned(),
            ));
        }
    }
    Ok(())
}

fn resolve_sound_speed_dataset<R: consus_io::ReadAt + Sync>(
    hdf5: &Hdf5File<R>,
    config: &BreastUstAliPhantomHdf5Config,
) -> KwaversResult<(String, u64)> {
    if let Some(path) = &config.sound_speed_dataset_path {
        let addr = hdf5.open_path(path).map_err(|err| {
            KwaversError::InvalidInput(format!("cannot locate HDF5 dataset '{path}': {err}"))
        })?;
        return Ok((path.clone(), addr));
    }

    for candidate in SOUND_SPEED_DATASET_CANDIDATES {
        if let Ok(addr) = hdf5.open_path(candidate) {
            if hdf5.dataset_at(addr).is_ok() {
                return Ok(((*candidate).to_owned(), addr));
            }
        }
    }

    Err(KwaversError::InvalidInput(format!(
        "no breast phantom sound-speed dataset found; tried {}",
        SOUND_SPEED_DATASET_CANDIDATES.join(", ")
    )))
}

fn rank3_dims(dims: &[usize]) -> KwaversResult<[usize; 3]> {
    if dims.len() != 3 {
        return Err(KwaversError::DimensionMismatch(format!(
            "breast phantom sound-speed dataset must be rank 3, got rank {} with dims {:?}",
            dims.len(),
            dims
        )));
    }
    if dims.contains(&0) {
        return Err(KwaversError::InvalidInput(format!(
            "breast phantom dimensions must be nonzero, got {dims:?}"
        )));
    }
    Ok([dims[0], dims[1], dims[2]])
}

fn resolve_spacing_m<R: consus_io::ReadAt + Sync>(
    hdf5: &Hdf5File<R>,
    dataset_addr: u64,
    config: &BreastUstAliPhantomHdf5Config,
) -> KwaversResult<f64> {
    if let Some(spacing_m) = config.spacing_m {
        return Ok(spacing_m);
    }

    let mut attrs = hdf5.attributes_at(dataset_addr).map_err(|err| {
        KwaversError::InvalidInput(format!("HDF5 dataset attributes failed: {err}"))
    })?;
    let root = hdf5.root_group();
    attrs.extend(
        hdf5.attributes_at(root.object_header_address)
            .map_err(|err| {
                KwaversError::InvalidInput(format!("HDF5 root attributes failed: {err}"))
            })?,
    );
    spacing_from_attrs(&attrs)?.ok_or_else(|| {
        KwaversError::InvalidInput(
            "breast phantom spacing_m missing; pass spacing_m or store spacing_m/dxi_m attribute"
                .to_owned(),
        )
    })
}

fn spacing_from_attrs(attrs: &[Hdf5Attribute]) -> KwaversResult<Option<f64>> {
    for attr in attrs {
        if SPACING_ATTRS_M.contains(&attr.name.as_str()) {
            return attr_scalar(attr, 1.0).map(Some);
        }
        if SPACING_ATTRS_MM.contains(&attr.name.as_str()) {
            return attr_scalar(attr, 1.0e-3).map(Some);
        }
    }
    Ok(None)
}

fn attr_scalar(attr: &Hdf5Attribute, scale: f64) -> KwaversResult<f64> {
    let decoded = attr.decode_value().map_err(|err| {
        KwaversError::InvalidInput(format!(
            "spacing attribute '{}' decode failed: {err}",
            attr.name
        ))
    })?;
    let value = match decoded {
        AttributeValue::Float(value) => value,
        AttributeValue::Int(value) => value as f64,
        AttributeValue::Uint(value) => value as f64,
        AttributeValue::FloatArray(values) => uniform_attr_value(&attr.name, &values)?,
        AttributeValue::IntArray(values) => uniform_attr_value(&attr.name, &values)? as f64,
        AttributeValue::UintArray(values) => uniform_attr_value(&attr.name, &values)? as f64,
        other => {
            return Err(KwaversError::InvalidInput(format!(
                "spacing attribute '{}' must be numeric, got {other:?}",
                attr.name
            )));
        }
    } * scale;
    validate_spacing(value)?;
    Ok(value)
}

fn uniform_attr_value<T>(name: &str, values: &[T]) -> KwaversResult<T>
where
    T: Copy + PartialEq + std::fmt::Debug,
{
    let (&first, rest) = values.split_first().ok_or_else(|| {
        KwaversError::InvalidInput(format!("spacing attribute '{name}' must not be empty"))
    })?;
    if rest.iter().any(|value| *value != first) {
        return Err(KwaversError::InvalidInput(format!(
            "spacing attribute '{name}' must be scalar or uniform, got {values:?}"
        )));
    }
    Ok(first)
}

fn validate_spacing(spacing_m: f64) -> KwaversResult<()> {
    if !spacing_m.is_finite() || spacing_m <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "spacing_m must be positive and finite, got {spacing_m}"
        )));
    }
    Ok(())
}
