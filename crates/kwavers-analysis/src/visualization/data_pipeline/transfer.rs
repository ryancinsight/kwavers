//! Core data pipeline orchestration (provider-generic).
//!
//! Owns CPU preprocessing, backend-neutral field metadata, transfer options,
//! and statistics. Concrete device ownership lives behind
//! [`VisualizationTransferProvider`]; the Hephaestus-backed implementation is
//! `kwavers_gpu::visualization::HephaestusVisualizationProvider`.

use super::super::transfer_contract::VisualizationTransferProvider;
use super::{ProcessingOperation, ProcessingStage, TransferStatistics};
use kwavers_core::error::KwaversResult;
use kwavers_field::UnifiedFieldType;
use leto::Array3;
use log::debug;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Transfer mode for data pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferMode {
    /// Synchronous blocking transfer
    Blocking,
    /// Asynchronous non-blocking transfer
    Async,
    /// Streaming with double buffering
    Streaming,
}

/// Transfer options
#[derive(Debug, Clone)]
pub struct TransferOptions {
    /// Scheduling and synchronization mode used by the provider.
    pub mode: TransferMode,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            mode: TransferMode::Async,
        }
    }
}

/// Provider-generic visualization data pipeline.
///
/// The pipeline owns everything that is not device-specific: preprocessing
/// operations per field type, field dimension and value-range metadata,
/// transfer options, and timing statistics. Device buffers are owned by the
/// injected [`VisualizationTransferProvider`].
#[derive(Debug)]
pub struct DataPipeline {
    provider: Box<dyn VisualizationTransferProvider>,
    transfer_stats: Mutex<TransferStatistics>,
    processing_stage: ProcessingStage,

    // Field metadata cache
    field_dimensions: HashMap<UnifiedFieldType, (u32, u32, u32)>,
    field_ranges: HashMap<UnifiedFieldType, (f32, f32)>,
    processing_operations: HashMap<UnifiedFieldType, ProcessingOperation>,
    transfer_options: TransferOptions,
}

impl DataPipeline {
    /// Create a new data pipeline over an acquired provider.
    ///
    /// Providers are constructed fallibly at the boundary that owns GPU
    /// capability (for example `kwavers-gpu`'s WGPU provider); construction
    /// failure surfaces there as a typed resource-unavailable error instead of
    /// silently degrading to CPU execution.
    pub fn new(provider: Box<dyn VisualizationTransferProvider>) -> Self {
        Self::with_transfer_mode(provider, TransferMode::Async)
    }

    /// Create a data pipeline with an explicit provider transfer mode.
    #[must_use]
    pub fn with_transfer_mode(
        provider: Box<dyn VisualizationTransferProvider>,
        mode: TransferMode,
    ) -> Self {
        Self {
            provider,
            transfer_stats: Mutex::new(TransferStatistics::default()),
            processing_stage: ProcessingStage::new(Default::default()),
            field_dimensions: HashMap::new(),
            field_ranges: HashMap::new(),
            processing_operations: HashMap::new(),
            transfer_options: TransferOptions { mode },
        }
    }

    /// Select how subsequent provider transfers are submitted and synchronized.
    pub fn set_transfer_mode(&mut self, mode: TransferMode) {
        self.transfer_options.mode = mode;
    }

    /// Transfer field data through the provider.
    ///
    /// Applies the configured CPU preprocessing, hands contiguous f32 samples
    /// to the provider's device buffers, and commits backend-neutral metadata
    /// (dimensions, value range) after the provider accepts the transfer.
    ///
    /// This method is synchronous because the provider contract includes an
    /// explicit blocking transfer mode. An asynchronous application runtime
    /// must invoke blocking-mode transfers on its blocking executor.
    ///
    /// # Errors
    ///
    /// Propagates any [`KwaversError`] returned by the provider.
    pub fn transfer_field(
        &mut self,
        field_type: UnifiedFieldType,
        data: &Array3<f64>,
    ) -> KwaversResult<()> {
        let start = Instant::now();

        let operation = self
            .processing_operations
            .get(&field_type)
            .copied()
            .unwrap_or(ProcessingOperation::None);

        let processed;
        let view = if operation.requires_preprocessing() {
            debug!("Applying {:?} to field {:?}", operation, field_type);
            processed = {
                let mut tmp = data.clone();
                self.processing_stage.apply(operation, &mut tmp);
                tmp
            };
            &processed
        } else {
            data
        };

        let [nx, ny, nz] = view.shape();
        let dimensions = (
            u32::try_from(nx).map_err(|_| {
                kwavers_core::error::KwaversError::InvalidInput(
                    "visualization field x dimension exceeds u32".to_string(),
                )
            })?,
            u32::try_from(ny).map_err(|_| {
                kwavers_core::error::KwaversError::InvalidInput(
                    "visualization field y dimension exceeds u32".to_string(),
                )
            })?,
            u32::try_from(nz).map_err(|_| {
                kwavers_core::error::KwaversError::InvalidInput(
                    "visualization field z dimension exceeds u32".to_string(),
                )
            })?,
        );

        let min_val = view.iter().fold(f64::INFINITY, |a, &b| a.min(b)) as f32;
        let max_val = view.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) as f32;
        let range = (min_val, max_val);

        let data_f32: Vec<f32> = view.iter().map(|&v| v as f32).collect();

        self.provider
            .transfer_field(field_type, &data_f32, self.transfer_options.mode)?;

        self.field_dimensions.insert(field_type, dimensions);
        self.field_ranges.insert(field_type, range);

        let elapsed = start.elapsed();
        debug!("Field transfer completed in {:?}", elapsed);

        if let Ok(mut stats) = self.transfer_stats.lock() {
            stats.record_transfer(
                data_f32.len() * std::mem::size_of::<f32>(),
                elapsed.as_secs_f32() * 1000.0,
            );
        }

        Ok(())
    }

    /// Upload field data through the provider (alias for [`Self::transfer_field`]).
    ///
    /// # Errors
    ///
    /// Propagates any [`KwaversError`] returned by the provider.
    ///
    pub fn upload_field(
        &mut self,
        data: &Array3<f64>,
        field_type: UnifiedFieldType,
    ) -> KwaversResult<()> {
        self.transfer_field(field_type, data)
    }

    /// Set processing operation for a field type
    pub fn set_processing(&mut self, field_type: UnifiedFieldType, operation: ProcessingOperation) {
        self.processing_operations.insert(field_type, operation);
    }

    /// Get field dimensions
    pub fn get_field_dimensions(&self, field_type: UnifiedFieldType) -> Option<(u32, u32, u32)> {
        self.field_dimensions.get(&field_type).copied()
    }

    /// Get field value range
    pub fn get_field_range(&self, field_type: UnifiedFieldType) -> Option<(f32, f32)> {
        self.field_ranges.get(&field_type).copied()
    }

    pub fn get_transfer_statistics(&self) -> Option<TransferStatistics> {
        self.transfer_stats.lock().ok().map(|s| s.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_core::error::KwaversError;

    /// Diagnostic provider recording per-field uploads without device
    /// resources; proves the pipeline's CPU-side semantics are preserved
    /// regardless of backend.
    type Upload = (UnifiedFieldType, Vec<f32>, TransferMode);

    #[derive(Debug)]
    struct RecordingProvider {
        uploads: std::sync::Arc<std::sync::Mutex<Vec<Upload>>>,
        available: bool,
        fail_with: Option<&'static str>,
    }

    impl RecordingProvider {
        fn new() -> (Self, std::sync::Arc<std::sync::Mutex<Vec<Upload>>>) {
            let uploads = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
            (
                Self {
                    uploads: std::sync::Arc::clone(&uploads),
                    available: true,
                    fail_with: None,
                },
                uploads,
            )
        }
    }

    impl super::super::super::transfer_contract::VisualizationTransferProvider for RecordingProvider {
        fn device_name(&self) -> &str {
            "recording-stub"
        }

        fn is_available(&self) -> bool {
            self.available
        }

        fn transfer_field(
            &mut self,
            field_type: UnifiedFieldType,
            samples: &[f32],
            mode: TransferMode,
        ) -> KwaversResult<()> {
            if let Some(message) = self.fail_with {
                return Err(KwaversError::System(
                    kwavers_core::error::SystemError::ResourceUnavailable {
                        resource: message.to_string(),
                    },
                ));
            }
            if let Ok(mut log) = self.uploads.lock() {
                log.push((field_type, samples.to_vec(), mode));
            }
            Ok(())
        }

        fn memory_usage(&self) -> usize {
            self.uploads
                .lock()
                .map(|log| {
                    log.iter()
                        .map(|(_, s, _)| s.len() * std::mem::size_of::<f32>())
                        .sum()
                })
                .unwrap_or(0)
        }
    }

    fn sample_field(value: f64) -> Array3<f64> {
        Array3::from_elem((2, 2, 2), value)
    }

    #[test]
    fn single_field_transfer_reaches_provider_with_f32_samples() {
        let (provider, log) = RecordingProvider::new();
        let mut pipeline = DataPipeline::new(Box::new(provider));
        pipeline
            .transfer_field(UnifiedFieldType::Pressure, &sample_field(1.5))
            .expect("single-field transfer succeeds");

        let log = log.lock().expect("upload log");
        assert_eq!(log.len(), 1);
        let (field_type, samples, mode) = &log[0];
        assert_eq!(*field_type, UnifiedFieldType::Pressure);
        assert_eq!(samples, &[1.5f32; 8]);
        assert_eq!(*mode, TransferMode::Async);
    }

    #[test]
    fn multi_field_transfers_preserve_distinct_field_identity() {
        let (provider, log) = RecordingProvider::new();
        let mut pipeline = DataPipeline::new(Box::new(provider));
        pipeline
            .transfer_field(UnifiedFieldType::Pressure, &sample_field(1.0))
            .expect("pressure transfer succeeds");
        pipeline
            .transfer_field(UnifiedFieldType::Temperature, &sample_field(2.0))
            .expect("temperature transfer succeeds");

        let log = log.lock().expect("upload log");
        assert_eq!(log.len(), 2);
        assert_ne!(log[0].0, log[1].0);
        assert_eq!(log[1].1, &[2.0f32; 8]);
    }

    #[test]
    fn distinct_input_values_update_metadata_and_samples() {
        let (provider, log) = RecordingProvider::new();
        let mut pipeline = DataPipeline::new(Box::new(provider));
        pipeline
            .transfer_field(UnifiedFieldType::Pressure, &sample_field(1.0))
            .expect("first transfer succeeds");
        pipeline
            .transfer_field(UnifiedFieldType::Pressure, &sample_field(4.0))
            .expect("second transfer succeeds");

        assert_eq!(
            pipeline.get_field_range(UnifiedFieldType::Pressure),
            Some((4.0, 4.0))
        );
        assert_eq!(
            pipeline.get_field_dimensions(UnifiedFieldType::Pressure),
            Some((2, 2, 2))
        );
        let log = log.lock().expect("upload log");
        assert_eq!(log[1].1, &[4.0f32; 8]);
    }

    #[test]
    fn provider_failure_propagates_without_silent_cpu_degradation() {
        let (mut provider, _) = RecordingProvider::new();
        provider.fail_with = Some("GPU adapter for visualization");
        let mut pipeline = DataPipeline::new(Box::new(provider));

        let result = pipeline.transfer_field(UnifiedFieldType::Pressure, &sample_field(1.0));
        assert!(matches!(
            result,
            Err(KwaversError::System(
                kwavers_core::error::SystemError::ResourceUnavailable { .. }
            ))
        ));
        assert_eq!(
            pipeline.get_field_dimensions(UnifiedFieldType::Pressure),
            None
        );
        assert_eq!(pipeline.get_field_range(UnifiedFieldType::Pressure), None);
    }

    #[test]
    fn statistics_record_bytes_and_calls() {
        let (provider, _) = RecordingProvider::new();
        let mut pipeline = DataPipeline::new(Box::new(provider));
        pipeline
            .transfer_field(UnifiedFieldType::Pressure, &sample_field(1.0))
            .expect("transfer succeeds");

        let stats = pipeline
            .get_transfer_statistics()
            .expect("statistics recorded");
        assert_eq!(
            stats.total_bytes_transferred,
            8 * std::mem::size_of::<f32>()
        );
        assert_eq!(stats.num_transfers, 1);
    }

    #[test]
    fn explicit_transfer_modes_reach_provider() {
        for mode in [TransferMode::Blocking, TransferMode::Streaming] {
            let (provider, log) = RecordingProvider::new();
            let mut pipeline = DataPipeline::new(Box::new(provider));
            pipeline.set_transfer_mode(mode);
            pipeline
                .transfer_field(UnifiedFieldType::Pressure, &sample_field(1.0))
                .expect("configured transfer succeeds");

            let log = log.lock().expect("upload log");
            assert_eq!(log[0].2, mode);
        }
    }
}
