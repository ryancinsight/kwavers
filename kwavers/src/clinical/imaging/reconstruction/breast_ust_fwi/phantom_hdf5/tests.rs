use super::*;
use consus_core::{ByteOrder as ConsusByteOrder, Datatype, Shape};
use consus_hdf5::file::writer::{DatasetCreationProps, FileCreationProps, Hdf5FileBuilder};
use std::num::NonZeroUsize;

#[test]
fn loader_reads_fortran_order_dataset_with_spacing_attribute() {
    let path = tempfile::NamedTempFile::new()
        .expect("temp")
        .into_temp_path();
    let dims = [2, 3, 2];
    let values: Vec<f64> = (0..dims[2])
        .flat_map(|k| {
            (0..dims[1]).flat_map(move |j| {
                (0..dims[0]).map(move |i| 1450.0 + i as f64 + 10.0 * j as f64 + 100.0 * k as f64)
            })
        })
        .collect();
    write_dataset(
        &path,
        "/phantom/sound_speed_m_s",
        &values,
        dims,
        Some(8.0e-4),
        DatasetCreationProps::default(),
    );

    let phantom = load_ali_2025_breast_phantom_from_hdf5(&path).expect("phantom");

    assert_eq!(phantom.model_family, BREAST_UST_ALI_2025_PHANTOM_MODEL);
    assert_eq!(phantom.dataset_path, "/phantom/sound_speed_m_s");
    assert_eq!(phantom.dimensions(), (2, 3, 2));
    assert_eq!(phantom.spacing_m, 8.0e-4);
    assert_eq!(phantom.sound_speed_m_s[[1, 2, 1]], 1571.0);
    assert_eq!(
        phantom.physical_extent_m(),
        [2.0 * 8.0e-4, 3.0 * 8.0e-4, 2.0 * 8.0e-4]
    );
}

#[test]
fn loader_reads_chunked_dataset_and_converts_kilometers_per_second() {
    let path = tempfile::NamedTempFile::new()
        .expect("temp")
        .into_temp_path();
    let dims = [2, 2, 2];
    let values: Vec<f64> = vec![1.45, 1.46, 1.47, 1.48, 1.49, 1.50, 1.51, 1.52];
    let dcpl = DatasetCreationProps {
        layout: consus_hdf5::property_list::DatasetLayout::Chunked,
        chunk_dims: Some(vec![1, 2, 2]),
        ..DatasetCreationProps::default()
    };
    write_dataset(&path, "/sos", &values, dims, None, dcpl);

    let phantom = load_ali_2025_breast_phantom_from_hdf5_with_config(
        &path,
        BreastUstAliPhantomHdf5Config {
            sound_speed_dataset_path: Some("/sos".to_owned()),
            spacing_m: Some(1.6e-3),
            sound_speed_unit: BreastUstSoundSpeedUnit::KilometersPerSecond,
            storage_order: BreastUstPhantomStorageOrder::CContiguous,
        },
    )
    .expect("phantom");

    assert_eq!(phantom.sound_speed_m_s[[0, 0, 0]], 1450.0);
    assert_eq!(phantom.sound_speed_m_s[[1, 1, 1]], 1520.0);
    assert_eq!(phantom.storage_order.label(), "c_contiguous");
    assert_eq!(phantom.spacing_m, 1.6e-3);
}

#[test]
fn loader_rejects_missing_spacing_when_config_omits_spacing() {
    let path = tempfile::NamedTempFile::new()
        .expect("temp")
        .into_temp_path();
    let dims = [1, 2, 2];
    let values: Vec<f64> = vec![1450.0, 1460.0, 1470.0, 1480.0];
    write_dataset(
        &path,
        "/sos",
        &values,
        dims,
        None,
        DatasetCreationProps::default(),
    );

    let err = load_ali_2025_breast_phantom_from_hdf5(&path).expect_err("missing spacing must fail");

    assert!(err.to_string().contains("spacing_m missing"));
}

fn write_dataset(
    path: &Path,
    dataset_path: &str,
    values: &[f64],
    dims: [usize; 3],
    spacing_m: Option<f64>,
    dcpl: DatasetCreationProps,
) {
    let datatype = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ConsusByteOrder::LittleEndian,
    };
    let shape = Shape::fixed(&dims);
    let raw: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let spacing_datatype = Datatype::Float {
        bits: NonZeroUsize::new(64).unwrap(),
        byte_order: ConsusByteOrder::LittleEndian,
    };
    let spacing_shape = Shape::scalar();
    let spacing_raw = spacing_m.map(f64::to_le_bytes);
    let attrs = spacing_raw
        .as_ref()
        .map(|raw| {
            vec![(
                "spacing_m",
                &spacing_datatype,
                &spacing_shape,
                raw.as_slice(),
            )]
        })
        .unwrap_or_default();

    let mut builder = Hdf5FileBuilder::new(FileCreationProps::default());
    let components: Vec<&str> = dataset_path
        .split('/')
        .filter(|part| !part.is_empty())
        .collect();
    match components.as_slice() {
        [name] => {
            builder
                .add_dataset_with_attributes(name, &datatype, &shape, &raw, &dcpl, &attrs)
                .expect("dataset");
        }
        [group, name] => {
            let mut group_builder = builder.begin_group(group);
            group_builder
                .add_dataset_with_attributes(name, &datatype, &shape, &raw, &dcpl, &attrs)
                .expect("dataset");
            group_builder.finish_with_attributes(&[]).expect("group");
        }
        _ => panic!("test helper supports root or one group"),
    }
    std::fs::write(path, builder.finish().expect("hdf5 bytes")).expect("write");
}
