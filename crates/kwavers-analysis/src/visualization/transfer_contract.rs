//! Provider-neutral visualization transfer seam.
//!
//! Owns the role contracts that [`crate::visualization::data_pipeline::DataPipeline`]
//! consumes. The domain crate keeps configuration, backend-neutral field
//! metadata, CPU preprocessing, statistics, and the public contract; concrete
//! device acquisition, resource allocation, and transfers are implemented by
//! providers (the Hephaestus-backed WGPU provider lives in `kwavers-gpu`).
//!
//! This seam is intentionally minimal: every role here has a current caller in
//! the visualization pipeline. Unavailable GPU capability surfaces as the
//! typed [`kwavers_core::error::KwaversError::System`] resource-unavailable error from the
//! provider's constructor; a requested GPU operation never degrades to CPU
//! execution.

use kwavers_core::error::KwaversResult;
use kwavers_field::UnifiedFieldType;

use super::data_pipeline::TransferMode;

/// Provider-neutral field-transfer role.
///
/// Implementations own concrete GPU buffers and queue submission for field
/// uploads. Callers hand over preprocessed, contiguous f32 samples; the
/// implementation decides buffer allocation, double-buffer selection, and
/// submission synchronization.
pub trait VisualizationTransferProvider: std::fmt::Debug + Send {
    /// Return the selected device name for diagnostics.
    fn device_name(&self) -> &str;

    /// Return true when the provider acquired real device resources.
    ///
    /// Providers are constructed through fallible constructors, so a
    /// constructed provider normally reports true; the accessor exists so the
    /// pipeline can distinguish a real adapter from a diagnostic stub without
    /// inferring from transfer success.
    fn is_available(&self) -> bool;

    /// Upload one field's samples into the provider's device buffers.
    ///
    /// Implementations must preserve per-field-type buffer identity: two
    /// distinct field types never alias one buffer, and streaming mode
    /// double-buffers within a single field type.
    ///
    /// # Errors
    ///
    /// Propagates provider allocation, mapping, or submission failures.
    fn transfer_field(
        &mut self,
        field_type: UnifiedFieldType,
        samples: &[f32],
        mode: TransferMode,
    ) -> KwaversResult<()>;

    /// Report provider-tracked device memory usage in bytes.
    fn memory_usage(&self) -> usize;
}
