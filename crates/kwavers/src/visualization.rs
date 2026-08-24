//! Visualization backend selection at the Kwavers composition boundary.
//!
//! Kwavers selects the runtime backend and injects its provider into
//! `kwavers-analysis`. Concrete provider implementations remain in
//! `kwavers-gpu`; Hephaestus owns the WGPU device, queue, synchronization, and
//! typed buffers used by the GPU provider.

use kwavers_analysis::visualization::VisualizationTransferProvider;
use kwavers_core::error::KwaversResult;
use kwavers_gpu::visualization::{HephaestusVisualizationProvider, LetoVisualizationProvider};

/// Select the concrete visualization transfer backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationBackend {
    /// Keep transferred fields in Leto-compatible host storage.
    Leto,
    /// Acquire and use a Hephaestus GPU device.
    Hephaestus,
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
pub fn create_visualization_provider(
    backend: VisualizationBackend,
) -> KwaversResult<Box<dyn VisualizationTransferProvider>> {
    match backend {
        VisualizationBackend::Leto => Ok(Box::new(LetoVisualizationProvider::new())),
        VisualizationBackend::Hephaestus => Ok(Box::new(HephaestusVisualizationProvider::new()?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_analysis::visualization::TransferMode;
    use kwavers_core::error::KwaversError;
    use kwavers_field::UnifiedFieldType;

    #[test]
    fn leto_selection_reaches_host_provider_contract() {
        let mut provider = create_visualization_provider(VisualizationBackend::Leto)
            .expect("Leto provider construction succeeds");

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
    #[ignore = "requires a real WGPU adapter on the scheduled GPU runner"]
    fn hephaestus_selection_requires_real_transfer() {
        let mut provider = create_visualization_provider(VisualizationBackend::Hephaestus)
            .expect("scheduled GPU runner must acquire a Hephaestus device");

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
