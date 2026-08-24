//! Visualization backend selection at the Kwavers composition boundary.
//!
//! Kwavers selects the runtime backend and injects its provider into
//! `kwavers-analysis`. Concrete provider implementations remain in
//! `kwavers-gpu`; Hephaestus owns the WGPU device, queue, synchronization, and
//! typed buffers used by the GPU provider.

use kwavers_analysis::visualization::TransferMode;
use kwavers_analysis::visualization::VisualizationTransferProvider;
use kwavers_core::error::KwaversResult;
use kwavers_field::UnifiedFieldType;
use kwavers_gpu::visualization::{HephaestusVisualizationProvider, LetoVisualizationProvider};

/// Select the concrete visualization transfer backend.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationBackend {
    /// Keep transferred fields in Leto-compatible host storage.
    Leto,
    /// Acquire and use a Hephaestus GPU device.
    Hephaestus,
}

/// Provider selected at the Kwavers composition boundary.
///
/// This closed-set enum preserves runtime backend selection while dispatching
/// each transfer without a vtable. The large Hephaestus state is boxed once at
/// construction so the Leto variant does not inherit its stack footprint.
#[non_exhaustive]
#[derive(Debug)]
pub enum VisualizationProvider {
    /// Leto-compatible host storage.
    Leto(LetoVisualizationProvider),
    /// Hephaestus-owned WGPU storage and submission.
    Hephaestus(Box<HephaestusVisualizationProvider>),
}

impl VisualizationTransferProvider for VisualizationProvider {
    fn device_name(&self) -> &str {
        match self {
            Self::Leto(provider) => provider.device_name(),
            Self::Hephaestus(provider) => provider.device_name(),
        }
    }

    fn is_available(&self) -> bool {
        match self {
            Self::Leto(provider) => provider.is_available(),
            Self::Hephaestus(provider) => provider.is_available(),
        }
    }

    fn transfer_field(
        &mut self,
        field_type: UnifiedFieldType,
        samples: &[f32],
        mode: TransferMode,
    ) -> KwaversResult<()> {
        match self {
            Self::Leto(provider) => provider.transfer_field(field_type, samples, mode),
            Self::Hephaestus(provider) => provider.transfer_field(field_type, samples, mode),
        }
    }

    fn memory_usage(&self) -> usize {
        match self {
            Self::Leto(provider) => provider.memory_usage(),
            Self::Hephaestus(provider) => provider.memory_usage(),
        }
    }
}

/// Create the selected visualization transfer provider.
///
/// The Hephaestus branch performs real device acquisition and returns its
/// failure. It never substitutes the Leto provider when no GPU is available.
/// Selection occurs once at this application composition boundary; field
/// transfers remain behind the provider-neutral analysis contract.
///
/// # Errors
///
/// Returns the typed acquisition or allocation error from the selected
/// provider.
///
/// # Examples
///
/// ```
/// use kwavers::visualization::{create_visualization_provider, VisualizationBackend};
/// use kwavers_analysis::visualization::{VisualizationConfig, VisualizationEngine};
///
/// let provider = create_visualization_provider(VisualizationBackend::Leto)?;
/// let mut engine = VisualizationEngine::create(VisualizationConfig::default())?
///     .set_transfer_provider(provider);
/// engine.initialize_gpu()?;
/// assert!(engine.is_gpu_initialized());
/// # Ok::<(), kwavers_core::error::KwaversError>(())
/// ```
///
/// An unconfigured engine cannot initialize GPU visualization:
///
/// ```compile_fail
/// use kwavers_analysis::visualization::{VisualizationConfig, VisualizationEngine};
///
/// let mut engine = VisualizationEngine::create(VisualizationConfig::default()).unwrap();
/// engine.initialize_gpu().unwrap();
/// ```
pub fn create_visualization_provider(
    backend: VisualizationBackend,
) -> KwaversResult<VisualizationProvider> {
    match backend {
        VisualizationBackend::Leto => {
            Ok(VisualizationProvider::Leto(LetoVisualizationProvider::new()))
        }
        VisualizationBackend::Hephaestus => Ok(VisualizationProvider::Hephaestus(Box::new(
            HephaestusVisualizationProvider::new()?,
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_analysis::visualization::{VisualizationConfig, VisualizationEngine};
    use kwavers_core::error::KwaversError;

    #[test]
    fn leto_selection_reaches_host_provider_contract() {
        let mut provider = create_visualization_provider(VisualizationBackend::Leto)
            .expect("Leto provider construction succeeds");

        assert!(matches!(provider, VisualizationProvider::Leto(_)));
        assert_eq!(provider.device_name(), "leto-cpu");
        assert!(provider.is_available());
        assert!(matches!(
            provider.transfer_field(UnifiedFieldType::Pressure, &[], TransferMode::Blocking,),
            Err(KwaversError::InvalidInput(_))
        ));

        provider
            .transfer_field(
                UnifiedFieldType::Pressure,
                &[1.0, 2.0, 3.0, 4.0],
                TransferMode::Blocking,
            )
            .expect("Leto field transfer succeeds");
        assert_eq!(provider.memory_usage(), 4 * std::mem::size_of::<f32>());
    }

    #[test]
    fn leto_selection_configures_engine_without_type_erasure() {
        let provider = create_visualization_provider(VisualizationBackend::Leto)
            .expect("Leto provider construction succeeds");
        let mut engine = VisualizationEngine::create(VisualizationConfig::default())
            .expect("default visualization configuration is valid")
            .set_transfer_provider(provider);

        assert!(!engine.is_gpu_initialized());
        engine.set_transfer_mode(TransferMode::Streaming);
        engine
            .initialize_gpu()
            .expect("selected Leto provider initializes the engine pipeline");
        assert!(engine.is_gpu_initialized());
        engine.cleanup();
        assert!(!engine.is_gpu_initialized());
    }

    #[test]
    #[ignore = "requires a real WGPU adapter on the scheduled GPU runner"]
    fn hephaestus_selection_requires_real_transfer() {
        let mut provider = create_visualization_provider(VisualizationBackend::Hephaestus)
            .expect("scheduled GPU runner must acquire a Hephaestus device");

        assert!(matches!(provider, VisualizationProvider::Hephaestus(_)));
        assert!(provider.is_available());
        assert_ne!(provider.device_name(), "leto-cpu");
        assert!(!provider.device_name().is_empty());
        provider
            .transfer_field(
                UnifiedFieldType::Pressure,
                &[1.0, 2.0, 3.0, 4.0],
                TransferMode::Blocking,
            )
            .expect("scheduled GPU runner must complete a blocking transfer");
        assert_eq!(provider.memory_usage(), 2 * 4 * std::mem::size_of::<f32>());
    }
}
