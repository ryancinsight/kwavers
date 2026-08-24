use crate::reconstruction::breast_ust_fwi::phantom_types::{
    BreastUstPhantomStorageOrder, BreastUstSoundSpeedUnit,
};
use consus_core::{decode_to_f64, Datatype};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

pub(super) fn read_dataset_payload<R: consus_io::ReadAt + Sync>(
    hdf5: &Hdf5File<R>,
    dataset_addr: u64,
    dataset: &consus_hdf5::dataset::Hdf5Dataset,
) -> KwaversResult<Vec<u8>> {
    match dataset.layout {
        StorageLayout::Contiguous => {
            let data_addr = dataset.data_address.ok_or_else(|| {
                KwaversError::InvalidInput("contiguous HDF5 dataset has no data address".to_owned())
            })?;
            let elem_size = dataset.datatype.element_size().ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "variable-length HDF5 datatype is invalid for sound speed: {:?}",
                    dataset.datatype
                ))
            })?;
            let total_bytes = dataset
                .shape
                .num_elements()
                .checked_mul(elem_size)
                .ok_or_else(|| {
                    KwaversError::InvalidInput("HDF5 dataset byte count overflow".into())
                })?;
            let mut raw = vec![0u8; total_bytes];
            hdf5.read_contiguous_dataset_bytes(data_addr, 0, &mut raw)
                .map_err(|err| {
                    KwaversError::InvalidInput(format!("HDF5 payload read failed: {err}"))
                })?;
            Ok(raw)
        }
        StorageLayout::Chunked => {
            hdf5.read_chunked_dataset_all_bytes(dataset_addr)
                .map_err(|err| {
                    KwaversError::InvalidInput(format!("HDF5 chunked payload read failed: {err}"))
                })
        }
        other => Err(KwaversError::InvalidInput(format!(
            "unsupported HDF5 storage layout for breast phantom: {other:?}"
        ))),
    }
}

pub(super) fn decode_sound_speed_values(
    raw: &[u8],
    datatype: &Datatype,
    unit: BreastUstSoundSpeedUnit,
) -> KwaversResult<Vec<f64>> {
    let scale = unit.scale_to_meters_per_second();
    let mut values = decode_to_f64(raw, datatype).map_err(|err| {
        KwaversError::InvalidInput(format!("sound-speed HDF5 decode failed: {err}"))
    })?;
    for value in &mut values {
        *value *= scale;
    }
    Ok(values)
}

pub(super) fn volume_from_storage_order(
    dims: [usize; 3],
    values: Vec<f64>,
    order: BreastUstPhantomStorageOrder,
) -> KwaversResult<Array3<f64>> {
    let expected = dims.iter().product::<usize>();
    if values.len() != expected {
        return Err(KwaversError::DimensionMismatch(format!(
            "decoded sound-speed element count {} does not match dims {:?}",
            values.len(),
            dims
        )));
    }
    match order {
        BreastUstPhantomStorageOrder::CContiguous => {
            Array3::from_shape_vec((dims[0], dims[1], dims[2]), values)
                .map_err(|e| KwaversError::Shape(e.to_string()))
        }
        BreastUstPhantomStorageOrder::FortranContiguous => Ok(Array3::from_shape_fn(
            (dims[0], dims[1], dims[2]),
            |[i, j, k]| values[i + dims[0] * (j + dims[1] * k)],
        )),
    }
}

pub(super) fn validate_sound_speed_domain(sound_speed_m_s: &Array3<f64>) -> KwaversResult<()> {
    for &speed in sound_speed_m_s.iter() {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "sound speed must be positive and finite, got {speed}"
            )));
        }
    }
    Ok(())
}
