use super::*;
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use std::io::Write;

#[test]
fn mat5_loader_decodes_compressed_uint16_mri_and_maps_sound_speed() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("BreastPhantomFromMRI.mat");
    let dims = [140usize, 140usize, 80usize];
    let values = (0..dims.iter().product::<usize>())
        .map(|idx| 50u16 + (idx % 500) as u16)
        .collect::<Vec<_>>();
    write_mat5_uint16_volume(&path, "breast_mri", dims, &values, true);

    let phantom = load_ali_2025_breast_phantom_from_mat5_with_config(
        &path,
        BreastUstAliPhantomMat5Config {
            output_shape: [5, 5, 3],
            grid_spacing_m: 1.0e-3,
            breast_side: BreastUstMriBreastSide::Right,
            tissue_threshold: 40.0,
            ..BreastUstAliPhantomMat5Config::default()
        },
    )
    .expect("mat5 phantom");

    assert_eq!(phantom.dimensions(), (5, 5, 3));
    assert_eq!(phantom.dataset_path, "breast_mri");
    assert_eq!(phantom.model_family, BREAST_UST_ALI_2025_MAT5_PHANTOM_MODEL);
    assert!(phantom
        .sound_speed_m_s
        .iter()
        .all(|value| value.is_finite()));
    assert!(phantom
        .sound_speed_m_s
        .iter()
        .any(|&value| value > SOUND_SPEED_WATER_SIM && value <= 1750.0));
}

#[test]
fn mat5_loader_rejects_missing_variable() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fixture.mat");
    write_mat5_uint16_volume(&path, "other", [2, 2, 2], &[1, 2, 3, 4, 5, 6, 7, 8], false);

    let err = load_ali_2025_breast_phantom_from_mat5_with_config(
        &path,
        BreastUstAliPhantomMat5Config {
            output_shape: [2, 2, 2],
            mri_variable_name: "breast_mri".to_owned(),
            ..BreastUstAliPhantomMat5Config::default()
        },
    )
    .expect_err("missing variable must fail");

    assert!(err.to_string().contains("breast_mri"));
}

fn write_mat5_uint16_volume(
    path: &std::path::Path,
    name: &str,
    dims: [usize; 3],
    values: &[u16],
    compressed: bool,
) {
    let mut matrix = Vec::new();
    write_element(&mut matrix, 6, &[6, 0, 0, 0, 0, 0, 0, 0]);
    let dims_bytes = dims
        .iter()
        .flat_map(|&value| (value as i32).to_le_bytes())
        .collect::<Vec<_>>();
    write_element(&mut matrix, 5, &dims_bytes);
    write_element(&mut matrix, 1, name.as_bytes());
    let payload = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    write_element(&mut matrix, 4, &payload);

    let mut bytes = Vec::new();
    let mut header = [b' '; 128];
    let text = b"MATLAB 5.0 MAT-file, kwavers test fixture";
    header[..text.len()].copy_from_slice(text);
    header[124..126].copy_from_slice(&[0x00, 0x01]);
    header[126..128].copy_from_slice(b"IM");
    bytes.extend_from_slice(&header);

    if compressed {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        let mut inner = Vec::new();
        write_element(&mut inner, 14, &matrix);
        encoder.write_all(&inner).expect("compress");
        let compressed_payload = encoder.finish().expect("finish");
        write_element(&mut bytes, 15, &compressed_payload);
    } else {
        write_element(&mut bytes, 14, &matrix);
    }
    std::fs::write(path, bytes).expect("write mat5 fixture");
}

fn write_element(out: &mut Vec<u8>, data_type: u32, payload: &[u8]) {
    out.extend_from_slice(&data_type.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
    out.extend_from_slice(payload);
    while out.len() % 8 != 0 {
        out.push(0);
    }
}
