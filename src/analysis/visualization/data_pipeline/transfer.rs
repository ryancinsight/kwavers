//! Core data transfer pipeline implementation

use super::{ProcessingOperation, ProcessingStage, TransferStatistics};
use crate::analysis::visualization::FieldType;
use crate::core::error::{KwaversError, KwaversResult};
use log::{debug, info};
use ndarray::Array3;
use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

use wgpu::*;

/// Transfer mode for data pipeline
#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub mode: TransferMode,
}

impl Default for TransferOptions {
    fn default() -> Self {
        Self {
            mode: TransferMode::Async,
        }
    }
}

/// GPU data pipeline for visualization
#[derive(Debug)]
pub struct DataPipeline {
    device: Arc<Device>,
    queue: Arc<Queue>,
    field_buffers: HashMap<FieldType, FieldBuffers>,
    transfer_stats: Mutex<TransferStatistics>,
    processing_stage: ProcessingStage,

    // Field metadata cache
    field_dimensions: HashMap<FieldType, (u32, u32, u32)>,
    field_ranges: HashMap<FieldType, (f32, f32)>,
    processing_operations: HashMap<FieldType, ProcessingOperation>,
    transfer_options: TransferOptions,
}

#[derive(Debug)]
struct FieldBuffers {
    a: Buffer,
    b: Buffer,
    next_is_a: bool,
    bytes: usize,
}

impl FieldBuffers {
    fn new(device: &Device, bytes: usize, label: &str) -> Self {
        let usage = BufferUsages::COPY_DST | BufferUsages::STORAGE;
        let a = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{label}::a")),
            size: bytes as u64,
            usage,
            mapped_at_creation: false,
        });
        let b = device.create_buffer(&BufferDescriptor {
            label: Some(&format!("{label}::b")),
            size: bytes as u64,
            usage,
            mapped_at_creation: false,
        });
        Self {
            a,
            b,
            next_is_a: true,
            bytes,
        }
    }

    fn ensure_capacity(&mut self, device: &Device, bytes: usize, label: &str) {
        if self.bytes == bytes {
            return;
        }
        *self = Self::new(device, bytes, label);
    }

    fn select_buffer(&mut self, mode: TransferMode) -> &Buffer {
        match mode {
            TransferMode::Blocking | TransferMode::Async => &self.a,
            TransferMode::Streaming => {
                let buffer = if self.next_is_a { &self.a } else { &self.b };
                self.next_is_a = !self.next_is_a;
                buffer
            }
        }
    }
}

impl DataPipeline {
    /// Create a new data pipeline
    pub async fn new() -> KwaversResult<Self> {
        info!("Initializing GPU data pipeline for visualization");

        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: "GPU adapter for visualization".to_string(),
                })
            })?;

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Kwavers Visualization Device"),
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!("GPU device for visualization: {e}"),
                })
            })?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            field_buffers: HashMap::new(),
            transfer_stats: Mutex::new(TransferStatistics::default()),
            processing_stage: ProcessingStage::new(Default::default()),
            field_dimensions: HashMap::new(),
            field_ranges: HashMap::new(),
            processing_operations: HashMap::new(),
            transfer_options: TransferOptions::default(),
        })
    }

    /// Transfer field data to GPU
    pub async fn transfer_field(
        &mut self,
        field_type: FieldType,
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

        let dims = view.dim();
        self.field_dimensions
            .insert(field_type, (dims.0 as u32, dims.1 as u32, dims.2 as u32));

        let min_val = view.iter().fold(f64::INFINITY, |a, &b| a.min(b)) as f32;
        let max_val = view.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) as f32;
        self.field_ranges.insert(field_type, (min_val, max_val));

        let data_f32: Vec<f32> = view.iter().map(|&v| v as f32).collect();
        let bytes = data_f32.len() * std::mem::size_of::<f32>();

        match self.field_buffers.entry(field_type) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().ensure_capacity(
                    &self.device,
                    bytes,
                    &format!("field::{field_type:?}"),
                );
            }
            Entry::Vacant(entry) => {
                entry.insert(FieldBuffers::new(
                    &self.device,
                    bytes,
                    &format!("field::{field_type:?}"),
                ));
            }
        }

        let buffer = self
            .field_buffers
            .get_mut(&field_type)
            .ok_or_else(|| KwaversError::InvalidInput("field buffer missing".to_string()))?
            .select_buffer(self.transfer_options.mode);

        self.queue
            .write_buffer(buffer, 0, bytemuck::cast_slice(&data_f32));

        if self.transfer_options.mode == TransferMode::Blocking {
            self.device.poll(Maintain::Wait);
        }

        let elapsed = start.elapsed();
        debug!("Field transfer completed in {:?}", elapsed);

        if let Ok(mut stats) = self.transfer_stats.lock() {
            stats.record_transfer(bytes, elapsed.as_secs_f32() * 1000.0);
        }

        Ok(())
    }

    /// Upload field data to GPU (alias for transfer_field)
    pub async fn upload_field(
        &mut self,
        data: &Array3<f64>,
        field_type: FieldType,
    ) -> KwaversResult<()> {
        self.transfer_field(field_type, data).await
    }

    /// Set processing operation for a field type
    pub fn set_processing(&mut self, field_type: FieldType, operation: ProcessingOperation) {
        self.processing_operations.insert(field_type, operation);
    }

    /// Get field dimensions
    pub fn get_field_dimensions(&self, field_type: FieldType) -> Option<(u32, u32, u32)> {
        self.field_dimensions.get(&field_type).copied()
    }

    /// Get field value range
    pub fn get_field_range(&self, field_type: FieldType) -> Option<(f32, f32)> {
        self.field_ranges.get(&field_type).copied()
    }

    pub fn get_transfer_statistics(&self) -> Option<TransferStatistics> {
        self.transfer_stats.lock().ok().map(|s| s.clone())
    }
}
