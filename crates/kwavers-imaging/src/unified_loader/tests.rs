use super::{MedicalImageBatchLoader, UnifiedMedicalImageLoader};
use kwavers_core::test_support::assert_rejects;

#[test]
fn test_unified_loader_creation_ct() {
    let _loader = UnifiedMedicalImageLoader::ct_loader();
}

#[test]
fn test_unified_loader_creation_dicom() {
    let _loader = UnifiedMedicalImageLoader::dicom_loader();
}

#[test]
fn test_unified_loader_invalid_path() {
    let result = UnifiedMedicalImageLoader::from_path("nonexistent.nii.gz");
    assert_rejects(result, "Medical image file not found");
}

#[test]
fn test_unified_loader_unsupported_format() {
    // `from_path` checks existence before extension, so an unsupported format
    // is only reachable through a file that exists. The previous form passed
    // "test.xyz", which does not, so this test and
    // `test_unified_loader_invalid_path` both exercised the missing-file
    // branch and the format branch was never covered.
    let directory = tempfile::tempdir().expect("temp dir");
    let file = directory.path().join("scan.xyz");
    std::fs::write(&file, b"not a medical image").expect("write the placeholder file");
    let result = UnifiedMedicalImageLoader::from_path(&file.to_string_lossy());
    assert_rejects(result, "Unsupported medical image format");
}

#[test]
fn test_unified_loader_is_loaded() {
    let loader_ct = UnifiedMedicalImageLoader::ct_loader();
    assert!(!loader_ct.is_loaded());

    let loader_dicom = UnifiedMedicalImageLoader::dicom_loader();
    assert!(!loader_dicom.is_loaded());
}

#[test]
fn test_batch_loader_new() {
    let batch = MedicalImageBatchLoader::new();
    assert_eq!(batch.queued_count(), 0);
    assert_eq!(batch.loaded_count(), 0);
}

#[test]
fn test_batch_loader_add_invalid() {
    let mut batch = MedicalImageBatchLoader::new();
    let result = batch.add("nonexistent.nii");
    assert_rejects(result, "Medical image file not found");
}

#[test]
fn test_batch_loader_clear() {
    let mut batch = MedicalImageBatchLoader::new();
    batch.paths.push("test.nii".to_string());
    assert_eq!(batch.queued_count(), 1);

    batch.clear();
    assert_eq!(batch.queued_count(), 0);
}

#[test]
fn test_batch_loader_default() {
    let batch = MedicalImageBatchLoader::default();
    assert_eq!(batch.queued_count(), 0);
}

#[test]
fn test_batch_loader_get_nonexistent() {
    let batch = MedicalImageBatchLoader::new();
    assert!(batch.get_image(0).is_none());
    assert!(batch.get_metadata(0).is_none());
}
