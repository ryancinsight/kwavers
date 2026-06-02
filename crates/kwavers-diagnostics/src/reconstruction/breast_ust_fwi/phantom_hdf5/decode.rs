use crate::reconstruction::breast_ust_fwi::phantom_types::{
    BreastUstPhantomStorageOrder, BreastUstSoundSpeedUnit,
};
use kwavers_core::error::{KwaversError, KwaversResult};
use consus_core::{ByteOrder, Datatype};
use consus_hdf5::dataset::StorageLayout;
use consus_hdf5::file::Hdf5File;
use ndarray::Array3;

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
            let elem_size = fixed_element_size(&dataset.datatype)?;
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
    let elem_size = fixed_element_size(datatype)?;
    if !raw.len().is_multiple_of(elem_size) {
        return Err(KwaversError::InvalidInput(format!(
            "HDF5 payload length {} is not divisible by element size {}",
            raw.len(),
            elem_size
        )));
    }
    let scale = unit.scale_to_meters_per_second();
    let mut values = match datatype {
        Datatype::Float { bits, byte_order } => decode_float_values(raw, bits.get(), *byte_order)?,
        Datatype::Integer {
            bits,
            byte_order,
            signed,
        } => decode_integer_values(raw, bits.get(), *byte_order, *signed)?,
        other => {
            return Err(KwaversError::InvalidInput(format!(
                "unsupported sound-speed HDF5 datatype: {other:?}"
            )));
        }
    };
    values.iter_mut().for_each(|value| *value *= scale);
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
            Array3::from_shape_vec((dims[0], dims[1], dims[2]), values).map_err(KwaversError::from)
        }
        BreastUstPhantomStorageOrder::FortranContiguous => Ok(Array3::from_shape_fn(
            (dims[0], dims[1], dims[2]),
            |(i, j, k)| values[i + dims[0] * (j + dims[1] * k)],
        )),
    }
}

pub(super) fn validate_sound_speed_domain(sound_speed_m_s: &Array3<f64>) -> KwaversResult<()> {
    for &speed in sound_speed_m_s {
        if !speed.is_finite() || speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "sound speed must be positive and finite, got {speed}"
            )));
        }
    }
    Ok(())
}

fn fixed_element_size(datatype: &Datatype) -> KwaversResult<usize> {
    datatype.element_size().ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "variable-length HDF5 datatype is invalid for sound speed: {datatype:?}"
        ))
    })
}

fn decode_float_values(raw: &[u8], bits: usize, byte_order: ByteOrder) -> KwaversResult<Vec<f64>> {
    match (bits, byte_order) {
        (32, ByteOrder::LittleEndian) => Ok(raw
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        (32, ByteOrder::BigEndian) => Ok(raw
            .chunks_exact(4)
            .map(|c| f64::from(f32::from_be_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        (64, ByteOrder::LittleEndian) => Ok(raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()),
        (64, ByteOrder::BigEndian) => Ok(raw
            .chunks_exact(8)
            .map(|c| f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()),
        _ => Err(KwaversError::InvalidInput(format!(
            "unsupported float width for breast phantom sound speed: {bits}"
        ))),
    }
}

fn decode_integer_values(
    raw: &[u8],
    bits: usize,
    byte_order: ByteOrder,
    signed: bool,
) -> KwaversResult<Vec<f64>> {
    match (bits, byte_order, signed) {
        (8, _, false) => Ok(raw.iter().copied().map(f64::from).collect()),
        (8, _, true) => Ok(raw.iter().copied().map(|b| f64::from(b as i8)).collect()),
        (16, ByteOrder::LittleEndian, false) => Ok(raw
            .chunks_exact(2)
            .map(|c| f64::from(u16::from_le_bytes([c[0], c[1]])))
            .collect()),
        (16, ByteOrder::LittleEndian, true) => Ok(raw
            .chunks_exact(2)
            .map(|c| f64::from(i16::from_le_bytes([c[0], c[1]])))
            .collect()),
        (16, ByteOrder::BigEndian, false) => Ok(raw
            .chunks_exact(2)
            .map(|c| f64::from(u16::from_be_bytes([c[0], c[1]])))
            .collect()),
        (16, ByteOrder::BigEndian, true) => Ok(raw
            .chunks_exact(2)
            .map(|c| f64::from(i16::from_be_bytes([c[0], c[1]])))
            .collect()),
        (32, ByteOrder::LittleEndian, false) => Ok(raw
            .chunks_exact(4)
            .map(|c| f64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        (32, ByteOrder::LittleEndian, true) => Ok(raw
            .chunks_exact(4)
            .map(|c| f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        (32, ByteOrder::BigEndian, false) => Ok(raw
            .chunks_exact(4)
            .map(|c| f64::from(u32::from_be_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        (32, ByteOrder::BigEndian, true) => Ok(raw
            .chunks_exact(4)
            .map(|c| f64::from(i32::from_be_bytes([c[0], c[1], c[2], c[3]])))
            .collect()),
        _ => Err(KwaversError::InvalidInput(format!(
            "unsupported integer storage for breast phantom sound speed: bits={bits}, signed={signed}"
        ))),
    }
}
