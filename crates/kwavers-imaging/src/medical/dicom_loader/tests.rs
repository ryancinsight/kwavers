use super::loader::DicomImageLoader;
use super::types::DicomModality;
use crate::medical::MedicalImageLoader;

#[test]
fn test_dicom_loader_creation() {
    let loader = DicomImageLoader::new();
    assert!(loader.data().is_none());
    assert!(loader.dicom_metadata().is_none());
}

#[test]
fn test_dicom_modality_display() {
    assert_eq!(format!("{}", DicomModality::CT), "CT");
    assert_eq!(format!("{}", DicomModality::MR), "MR");
    assert_eq!(format!("{}", DicomModality::US), "US");
    assert_eq!(format!("{}", DicomModality::RD), "RD");
    assert_eq!(format!("{}", DicomModality::Other), "Other");
}

#[test]
fn test_dicom_modality_from_code() {
    assert_eq!(DicomModality::from_code("CT"), DicomModality::CT);
    assert_eq!(DicomModality::from_code("MR"), DicomModality::MR);
    assert_eq!(DicomModality::from_code("US"), DicomModality::US);
    assert_eq!(DicomModality::from_code("UNKNOWN"), DicomModality::Other);
}

#[test]
fn test_dicom_metadata_equality() {
    let m1 = DicomModality::CT;
    let m2 = DicomModality::CT;
    assert_eq!(m1, m2);
}

#[test]
fn test_dicom_loader_metadata_default() {
    let loader = DicomImageLoader::new();
    let metadata = loader.metadata();
    assert_eq!(metadata.dimensions, (0, 0, 0));
    assert!(metadata.modality.contains("Unknown"));
}

#[test]
fn test_dicom_loader_name() {
    let loader = DicomImageLoader::new();
    assert_eq!(loader.name(), "DICOM");
}

#[test]
fn test_dicom_to_hounsfield_units() {
    let hu = DicomImageLoader::to_hounsfield_units(0.0, 1.0, -1024.0);
    assert!((hu - (-1024.0)).abs() < 1e-6);

    let hu = DicomImageLoader::to_hounsfield_units(1024.0, 1.0, -1024.0);
    assert!((hu - 0.0).abs() < 1e-6);

    let hu = DicomImageLoader::to_hounsfield_units(2048.0, 1.0, -1024.0);
    assert!((hu - 1024.0).abs() < 1e-6);
}

#[test]
fn test_dicom_identity_affine() {
    let affine = DicomImageLoader::identity_affine();
    assert_eq!(affine[0][0], 1.0);
    assert_eq!(affine[1][1], 1.0);
    assert_eq!(affine[2][2], 1.0);
    assert_eq!(affine[3][3], 1.0);
    assert_eq!(affine[0][1], 0.0);
    assert_eq!(affine[3][0], 0.0);
}

#[test]
fn test_dicom_compute_affine() {
    let image_pos = [0.0, 0.0, 0.0];
    let image_orient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    let pixel_spacing = [1.0, 1.0];
    let slice_thickness = 2.0;

    let affine = DicomImageLoader::compute_affine(
        &image_pos,
        &image_orient,
        &pixel_spacing,
        slice_thickness,
    );

    assert_eq!(affine[0][0], 1.0);
    assert_eq!(affine[1][1], 1.0);
    assert_eq!(affine[2][2], 2.0);
    assert_eq!(affine[3][3], 1.0);
}

#[test]
fn test_dicom_single_file_error() {
    let mut loader = DicomImageLoader::new();
    let result = loader.load("test.dcm");
    assert!(result.is_err());
}

#[test]
fn test_dicom_invalid_path() {
    let mut loader = DicomImageLoader::new();
    let result = loader.load("/nonexistent/path/to/dicom");
    assert!(result.is_err());
}

#[test]
fn test_dicom_loader_default() {
    let loader = DicomImageLoader::default();
    assert!(loader.data().is_none());
}
