//! Minimal MATLAB Level-5 numeric-volume parser.

use kwavers_core::error::{KwaversError, KwaversResult};
use flate2::read::ZlibDecoder;
use std::fs::File;
use std::io::Read;
use std::path::Path;

const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;

#[derive(Clone, Debug)]
pub(super) struct Mat5NumericVolume {
    pub dims: [usize; 3],
    pub values: Vec<f64>,
    pub name: String,
}

#[derive(Clone, Copy, Debug)]
struct DataElement<'a> {
    data_type: u32,
    payload: &'a [u8],
}

pub(super) fn read_mat5_numeric_volume(
    path: &Path,
    requested_name: &str,
) -> KwaversResult<Mat5NumericVolume> {
    let mut bytes = Vec::new();
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() < 128 {
        return Err(KwaversError::InvalidInput(
            "MAT5 file is shorter than the 128-byte header".to_owned(),
        ));
    }
    if bytes[124..126] != [0x00, 0x01] || &bytes[126..128] != b"IM" {
        return Err(KwaversError::InvalidInput(
            "only little-endian MATLAB Level-5 files are supported".to_owned(),
        ));
    }
    parse_elements(&bytes[128..], requested_name)?.ok_or_else(|| {
        KwaversError::InvalidInput(format!(
            "MAT5 variable '{requested_name}' was not found as a real numeric 3-D array"
        ))
    })
}

fn parse_elements(bytes: &[u8], requested_name: &str) -> KwaversResult<Option<Mat5NumericVolume>> {
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let Some(element) = read_element(bytes, &mut cursor)? else {
            break;
        };
        match element.data_type {
            MI_COMPRESSED => {
                let mut inflated = Vec::new();
                ZlibDecoder::new(element.payload)
                    .read_to_end(&mut inflated)
                    .map_err(|err| {
                        KwaversError::InvalidInput(format!("MAT5 zlib payload failed: {err}"))
                    })?;
                if let Some(volume) = parse_elements(&inflated, requested_name)? {
                    return Ok(Some(volume));
                }
            }
            MI_MATRIX => {
                if let Some(volume) = parse_matrix(element.payload, requested_name)? {
                    return Ok(Some(volume));
                }
            }
            _ => {}
        }
    }
    Ok(None)
}

fn parse_matrix(bytes: &[u8], requested_name: &str) -> KwaversResult<Option<Mat5NumericVolume>> {
    let mut cursor = 0usize;
    let _flags = required_element(bytes, &mut cursor, "array flags")?;
    let dims_element = required_element(bytes, &mut cursor, "dimensions")?;
    let dims = decode_dimensions(dims_element)?;
    let name_element = required_element(bytes, &mut cursor, "array name")?;
    let name = std::str::from_utf8(name_element.payload)
        .map_err(|err| {
            KwaversError::InvalidInput(format!("MAT5 variable name is not UTF-8: {err}"))
        })?
        .to_owned();
    if name != requested_name {
        return Ok(None);
    }
    if dims.len() != 3 {
        return Err(KwaversError::InvalidInput(format!(
            "MAT5 variable '{name}' must be rank 3, got dimensions {:?}",
            dims
        )));
    }
    let data = required_element(bytes, &mut cursor, "real payload")?;
    let dims3 = [dims[0], dims[1], dims[2]];
    let expected = dims3.iter().product::<usize>();
    let values = decode_numeric_values(data)?;
    if values.len() != expected {
        return Err(KwaversError::DimensionMismatch(format!(
            "MAT5 variable '{name}' decoded {} elements for dims {:?}",
            values.len(),
            dims3
        )));
    }
    Ok(Some(Mat5NumericVolume {
        dims: dims3,
        values,
        name,
    }))
}

fn required_element<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    label: &str,
) -> KwaversResult<DataElement<'a>> {
    read_element(bytes, cursor)?.ok_or_else(|| {
        KwaversError::InvalidInput(format!("MAT5 matrix is missing {label} element"))
    })
}

fn read_element<'a>(bytes: &'a [u8], cursor: &mut usize) -> KwaversResult<Option<DataElement<'a>>> {
    skip_zero_padding(bytes, cursor);
    if *cursor >= bytes.len() {
        return Ok(None);
    }
    if bytes.len() - *cursor < 8 {
        return Err(KwaversError::InvalidInput(
            "MAT5 data element tag is truncated".to_owned(),
        ));
    }
    let tag0 = u32::from_le_bytes(bytes[*cursor..*cursor + 4].try_into().unwrap());
    let small_type = tag0 & 0xffff;
    let small_len = tag0 >> 16;
    if small_len > 0 {
        let payload_start = *cursor + 4;
        let payload_end = payload_start + small_len as usize;
        if payload_end > *cursor + 8 {
            return Err(KwaversError::InvalidInput(
                "MAT5 small data element length exceeds inline tag".to_owned(),
            ));
        }
        let payload = &bytes[payload_start..payload_end];
        *cursor += 8;
        return Ok(Some(DataElement {
            data_type: small_type,
            payload,
        }));
    }

    let data_type = tag0;
    let len = u32::from_le_bytes(bytes[*cursor + 4..*cursor + 8].try_into().unwrap()) as usize;
    let payload_start = *cursor + 8;
    let payload_end = payload_start.checked_add(len).ok_or_else(|| {
        KwaversError::InvalidInput("MAT5 data element length overflow".to_owned())
    })?;
    if payload_end > bytes.len() {
        return Err(KwaversError::InvalidInput(
            "MAT5 data element payload is truncated".to_owned(),
        ));
    }
    *cursor = align_to_8(payload_end);
    Ok(Some(DataElement {
        data_type,
        payload: &bytes[payload_start..payload_end],
    }))
}

fn skip_zero_padding(bytes: &[u8], cursor: &mut usize) {
    while *cursor < bytes.len() && bytes[*cursor] == 0 {
        *cursor += 1;
    }
}

fn align_to_8(value: usize) -> usize {
    (value + 7) & !7
}

fn decode_dimensions(element: DataElement<'_>) -> KwaversResult<Vec<usize>> {
    if element.data_type != MI_INT32 && element.data_type != MI_UINT32 {
        return Err(KwaversError::InvalidInput(format!(
            "MAT5 dimensions use unsupported datatype {}",
            element.data_type
        )));
    }
    if !element.payload.len().is_multiple_of(4) {
        return Err(KwaversError::InvalidInput(
            "MAT5 dimension payload is not 32-bit aligned".to_owned(),
        ));
    }
    element
        .payload
        .chunks_exact(4)
        .map(|chunk| {
            let value = i32::from_le_bytes(chunk.try_into().unwrap());
            if value <= 0 {
                return Err(KwaversError::InvalidInput(format!(
                    "MAT5 dimension must be positive, got {value}"
                )));
            }
            Ok(value as usize)
        })
        .collect()
}

fn decode_numeric_values(element: DataElement<'_>) -> KwaversResult<Vec<f64>> {
    match element.data_type {
        MI_UINT8 => Ok(element.payload.iter().copied().map(f64::from).collect()),
        MI_INT8 => Ok(element
            .payload
            .iter()
            .copied()
            .map(|value| f64::from(value as i8))
            .collect()),
        MI_UINT16 => decode_chunks(element.payload, 2, |c| {
            f64::from(u16::from_le_bytes([c[0], c[1]]))
        }),
        MI_INT16 => decode_chunks(element.payload, 2, |c| {
            f64::from(i16::from_le_bytes([c[0], c[1]]))
        }),
        MI_UINT32 => decode_chunks(element.payload, 4, |c| {
            f64::from(u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        }),
        MI_INT32 => decode_chunks(element.payload, 4, |c| {
            f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        }),
        MI_SINGLE => decode_chunks(element.payload, 4, |c| {
            f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        }),
        MI_DOUBLE => decode_chunks(element.payload, 8, |c| {
            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
        }),
        MI_UINT64 => decode_chunks(element.payload, 8, |c| {
            u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64
        }),
        MI_INT64 => decode_chunks(element.payload, 8, |c| {
            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64
        }),
        other => Err(KwaversError::InvalidInput(format!(
            "MAT5 real payload datatype {other} is not numeric"
        ))),
    }
}

fn decode_chunks<F>(payload: &[u8], width: usize, decode: F) -> KwaversResult<Vec<f64>>
where
    F: Fn(&[u8]) -> f64,
{
    if !payload.len().is_multiple_of(width) {
        return Err(KwaversError::InvalidInput(format!(
            "MAT5 numeric payload length {} is not divisible by {width}",
            payload.len()
        )));
    }
    Ok(payload.chunks_exact(width).map(decode).collect())
}
