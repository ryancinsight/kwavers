//! Provider-owned visualization transfer backends.
//!
//! The [`HephaestusVisualizationProvider`] delegates device acquisition,
//! typed buffer allocation, queue writes, and synchronization to Hephaestus.
//! The [`LetoVisualizationProvider`] retains a value-preserving host copy for
//! CPU visualization. Both implement the provider-neutral role owned by
//! `kwavers-analysis`; backend selection is explicit through
//! [`VisualizationBackend`].

use hephaestus_core::{ComputeDevice, DevicePreference};
use hephaestus_wgpu::{WgpuBuffer, WgpuDevice};
use kwavers_analysis::visualization::data_pipeline::TransferMode;
use kwavers_analysis::visualization::transfer_contract::VisualizationTransferProvider;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_field::UnifiedFieldType;
use leto::Array1;
use std::collections::HashMap;

/// Select the concrete visualization transfer backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisualizationBackend {
    /// Keep transferred fields in Leto-compatible host storage.
    Leto,
    /// Acquire and use a Hephaestus GPU device.
    Hephaestus,
}

/// Create a visualization transfer provider for an explicit backend.
///
/// The Hephaestus branch performs real device acquisition and returns its
/// failure. It never substitutes the Leto provider when no GPU is available.
/// The returned trait object is constructed once at the visualization control
/// boundary; field transfers remain behind the provider contract.
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

/// Host-backed visualization transfer provider for the Leto execution path.
#[derive(Debug, Default)]
pub struct LetoVisualizationProvider {
    fields: HashMap<UnifiedFieldType, Array1<f32>>,
    memory_bytes: usize,
}

impl LetoVisualizationProvider {
    /// Create an empty host-backed provider.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl VisualizationTransferProvider for LetoVisualizationProvider {
    fn device_name(&self) -> &'static str {
        "leto-cpu"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn transfer_field(
        &mut self,
        field_type: UnifiedFieldType,
        samples: &[f32],
        _mode: TransferMode,
    ) -> KwaversResult<()> {
        if samples.is_empty() {
            return Err(KwaversError::InvalidInput(
                "visualization transfer requires at least one sample".to_string(),
            ));
        }

        let replacement_bytes = samples
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                message: "visualization host buffer size overflows usize".to_string(),
            })?;
        let replacement =
            Array1::from_shape_vec([samples.len()], samples.to_vec()).map_err(|error| {
                KwaversError::InvalidInput(format!("invalid Leto visualization buffer: {error}"))
            })?;

        let previous_bytes = match self.fields.get(&field_type) {
            Some(field) => field
                .len()
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                    message: "visualization host buffer size overflows usize".to_string(),
                })?,
            None => 0,
        };
        self.fields.insert(field_type, replacement);
        self.memory_bytes = self
            .memory_bytes
            .checked_sub(previous_bytes)
            .and_then(|bytes| bytes.checked_add(replacement_bytes))
            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                message: "visualization host memory accounting overflowed".to_string(),
            })?;
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.memory_bytes
    }
}

/// Double-buffered typed device storage for one field.
#[derive(Debug)]
struct FieldBuffers {
    first: WgpuBuffer<f32>,
    second: WgpuBuffer<f32>,
    next_is_first: bool,
    bytes: usize,
}

impl FieldBuffers {
    fn allocate(device: &WgpuDevice, len: usize) -> KwaversResult<Self> {
        let first = device
            .alloc_zeroed::<f32>(len)
            .map_err(|error| map_device_error("allocate first visualization buffer", error))?;
        let second = device
            .alloc_zeroed::<f32>(len)
            .map_err(|error| map_device_error("allocate second visualization buffer", error))?;
        let bytes = len.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| {
            KwaversError::ResourceLimitExceeded {
                message: "visualization device buffer size overflows usize".to_string(),
            }
        })?;
        Ok(Self {
            first,
            second,
            next_is_first: true,
            bytes,
        })
    }

    fn select(&mut self, mode: TransferMode) -> &WgpuBuffer<f32> {
        match mode {
            TransferMode::Blocking | TransferMode::Async => &self.first,
            TransferMode::Streaming => {
                let selected = self.next_is_first;
                self.next_is_first = !self.next_is_first;
                if selected {
                    &self.first
                } else {
                    &self.second
                }
            }
        }
    }
}

/// Hephaestus-backed visualization transfer provider.
///
/// Hephaestus owns the WGPU instance, adapter, device, queue, typed buffers,
/// and synchronization implementation. Kwavers only selects this provider and
/// supplies typed field samples through the neutral visualization seam.
#[derive(Debug)]
pub struct HephaestusVisualizationProvider {
    device_name: String,
    device: crate::gpu::GpuDevice<WgpuDevice>,
    fields: HashMap<UnifiedFieldType, FieldBuffers>,
    memory_bytes: usize,
}

impl HephaestusVisualizationProvider {
    /// Acquire a high-performance Hephaestus device.
    ///
    /// # Errors
    ///
    /// Returns a typed GPU acquisition error when no suitable device is
    /// available or the provider cannot satisfy its baseline limits.
    pub fn new() -> KwaversResult<Self> {
        let device = crate::gpu::GpuDevice::try_create(DevicePreference::HighPerformance)?;
        let device_name = device.info().name.clone();
        Ok(Self {
            device_name,
            device,
            fields: HashMap::new(),
            memory_bytes: 0,
        })
    }
}

impl VisualizationTransferProvider for HephaestusVisualizationProvider {
    fn device_name(&self) -> &str {
        &self.device_name
    }

    fn is_available(&self) -> bool {
        true
    }

    fn transfer_field(
        &mut self,
        field_type: UnifiedFieldType,
        samples: &[f32],
        mode: TransferMode,
    ) -> KwaversResult<()> {
        if samples.is_empty() {
            return Err(KwaversError::InvalidInput(
                "visualization transfer requires at least one sample".to_string(),
            ));
        }

        let new_bytes = samples
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                message: "visualization device buffer size overflows usize".to_string(),
            })?;

        let buffer = {
            let entry = self.fields.entry(field_type);
            let buffers = match entry {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if entry.get().bytes != new_bytes {
                        let previous_allocation =
                            entry.get().bytes.checked_mul(2).ok_or_else(|| {
                                KwaversError::ResourceLimitExceeded {
                                    message: "visualization memory accounting overflowed"
                                        .to_string(),
                                }
                            })?;
                        let new_allocation = new_bytes.checked_mul(2).ok_or_else(|| {
                            KwaversError::ResourceLimitExceeded {
                                message: "visualization memory accounting overflowed".to_string(),
                            }
                        })?;
                        let updated_memory = self
                            .memory_bytes
                            .checked_sub(previous_allocation)
                            .and_then(|bytes| bytes.checked_add(new_allocation))
                            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                                message: "visualization memory accounting overflowed".to_string(),
                            })?;
                        let replacement =
                            FieldBuffers::allocate(self.device.provider(), samples.len())?;
                        entry.insert(replacement);
                        self.memory_bytes = updated_memory;
                    }
                    entry.into_mut()
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let new_allocation = new_bytes.checked_mul(2).ok_or_else(|| {
                        KwaversError::ResourceLimitExceeded {
                            message: "visualization memory accounting overflowed".to_string(),
                        }
                    })?;
                    let updated_memory =
                        self.memory_bytes
                            .checked_add(new_allocation)
                            .ok_or_else(|| KwaversError::ResourceLimitExceeded {
                                message: "visualization memory accounting overflowed".to_string(),
                            })?;
                    let buffers = FieldBuffers::allocate(self.device.provider(), samples.len())?;
                    self.memory_bytes = updated_memory;
                    entry.insert(buffers)
                }
            };
            buffers.select(mode)
        };
        self.device
            .provider()
            .write_sub_buffer(buffer, 0, samples)
            .map_err(|error| map_device_error("upload visualization field", error))?;

        if matches!(mode, TransferMode::Blocking) {
            self.device.synchronize()?;
        }
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.memory_bytes
    }
}

fn map_device_error(operation: &str, error: impl std::fmt::Display) -> KwaversError {
    KwaversError::GpuError(format!("{operation}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leto_provider_preserves_field_identity_and_values() {
        let mut provider = LetoVisualizationProvider::new();
        assert!(provider.is_available());
        assert_eq!(provider.device_name(), "leto-cpu");

        provider
            .transfer_field(
                UnifiedFieldType::Pressure,
                &[1.0f32; 4],
                TransferMode::Async,
            )
            .expect("pressure transfer succeeds");
        provider
            .transfer_field(
                UnifiedFieldType::Temperature,
                &[2.0f32; 4],
                TransferMode::Async,
            )
            .expect("temperature transfer succeeds");
        // Same field, different values: replacement must be value-sensitive.
        provider
            .transfer_field(
                UnifiedFieldType::Pressure,
                &[3.0f32; 8],
                TransferMode::Async,
            )
            .expect("pressure replacement succeeds");

        assert_eq!(provider.memory_usage(), 48); // 16 + 16 + 32 - 16 replaced
        assert_eq!(
            provider.fields[&UnifiedFieldType::Pressure].as_slice(),
            Some(&[3.0f32; 8][..])
        );
        assert_eq!(
            provider.fields[&UnifiedFieldType::Temperature].as_slice(),
            Some(&[2.0f32; 4][..])
        );
    }

    #[test]
    fn leto_provider_rejects_empty_samples() {
        let mut provider = LetoVisualizationProvider::new();
        let result = provider.transfer_field(UnifiedFieldType::Pressure, &[], TransferMode::Async);
        assert!(matches!(result, Err(KwaversError::InvalidInput(_))));
    }

    #[test]
    fn hephaestus_provider_transfers_when_adapter_exists() {
        let Ok(mut provider) = HephaestusVisualizationProvider::new() else {
            // No adapter in this environment is an environment fact, not a
            // contract failure; the differential path is exercised on
            // adapter-equipped CI runners.
            return;
        };
        assert!(provider.is_available());
        assert!(!provider.device_name().is_empty());

        provider
            .transfer_field(
                UnifiedFieldType::Pressure,
                &[1.0f32; 16],
                TransferMode::Blocking,
            )
            .expect("device-backed blocking transfer succeeds");
        provider
            .transfer_field(
                UnifiedFieldType::Temperature,
                &[0.5f32; 16],
                TransferMode::Streaming,
            )
            .expect("device-backed streaming transfer succeeds");
        assert_eq!(provider.fields.len(), 2);
        assert_eq!(
            provider.fields[&UnifiedFieldType::Pressure].bytes,
            16 * std::mem::size_of::<f32>()
        );
        assert_eq!(
            provider.fields[&UnifiedFieldType::Temperature].bytes,
            16 * std::mem::size_of::<f32>()
        );
        assert!(!provider.fields[&UnifiedFieldType::Temperature].next_is_first);
        provider
            .transfer_field(
                UnifiedFieldType::Temperature,
                &[0.25f32; 16],
                TransferMode::Streaming,
            )
            .expect("second device-backed streaming transfer succeeds");
        assert!(provider.fields[&UnifiedFieldType::Temperature].next_is_first);
        assert!(provider.memory_usage() >= 32 * std::mem::size_of::<f32>());
    }
}
