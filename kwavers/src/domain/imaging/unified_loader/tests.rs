use super::{MedicalImageBatchLoader, UnifiedMedicalImageLoader};

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
    assert!(result.is_err());
}

#[test]
fn test_unified_loader_unsupported_format() {
    let result = UnifiedMedicalImageLoader::from_path("test.xyz");
    assert!(result.is_err());
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
    assert!(result.is_err());
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
