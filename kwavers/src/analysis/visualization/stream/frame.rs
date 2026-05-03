//! Frame data types for visualization stream.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use ndarray::Array3;

use crate::domain::grid::Grid;

/// Unique identifier for each visualization frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameId(pub u64);

impl FrameId {
    /// Create a new frame ID (internal counter).
    pub fn next(counter: &AtomicU64) -> Self {
        Self(counter.fetch_add(1, Ordering::SeqCst))
    }
}

/// Metadata associated with each visualization frame.
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    /// Frame sequence identifier.
    pub id: FrameId,
    /// Simulation time in seconds.
    pub simulation_time: f64,
    /// Physical grid for dimensional information.
    pub grid: Grid,
    /// Quality level indicator (0.0 to 1.0).
    pub quality_factor: f32,
    /// Optional user-defined tags.
    pub tags: Vec<String>,
}

impl FrameMetadata {
    /// Create new frame metadata.
    pub fn new(simulation_time: f64, grid: Grid) -> Self {
        let id = {
            use std::sync::OnceLock;
            static ID_COUNTER: OnceLock<AtomicU64> = OnceLock::new();
            let counter = ID_COUNTER.get_or_init(|| AtomicU64::new(0));
            FrameId::next(counter)
        };
        Self {
            id,
            simulation_time,
            grid,
            quality_factor: 1.0,
            tags: Vec::new(),
        }
    }
}

/// Complete frame containing simulation field data for visualization.
#[derive(Debug, Clone)]
pub struct VizFrame {
    /// Pressure field array.
    pub field_pressure: Array3<f32>,
    /// Optional temperature field.
    pub field_temperature: Option<Array3<f32>>,
    /// Frame timestamp for latency calculations.
    pub timestamp: Instant,
    /// Frame metadata including ID and simulation time.
    pub metadata: FrameMetadata,
}

impl VizFrame {
    /// Create a new visualization frame.
    pub fn new(
        field_pressure: Array3<f32>,
        field_temperature: Option<Array3<f32>>,
        metadata: FrameMetadata,
    ) -> Self {
        Self {
            field_pressure,
            field_temperature,
            timestamp: Instant::now(),
            metadata,
        }
    }

    /// Calculate age of this frame (time since creation).
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Calculate bytes consumed by this frame's data.
    pub fn data_size_bytes(&self) -> usize {
        let pressure_bytes = self.field_pressure.len() * std::mem::size_of::<f32>();
        let temperature_bytes = self
            .field_temperature
            .as_ref()
            .map_or(0, |t| t.len() * std::mem::size_of::<f32>());
        pressure_bytes + temperature_bytes + std::mem::size_of::<Self>()
    }
}
