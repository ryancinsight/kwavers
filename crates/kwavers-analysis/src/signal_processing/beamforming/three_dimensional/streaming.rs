//! Streaming buffer for real-time 4D ultrasound processing.
//!
//! Implements circular buffering for continuous RF data acquisition and
//! processing in real-time volumetric imaging applications.
//!
//! Only compiled when the `gpu` feature is enabled — `StreamingBuffer` is
//! stored inside `BeamformingProcessor3D` which requires GPU device handles.

#[cfg(feature = "gpu")]
use kwavers_core::error::{KwaversError, KwaversResult};
#[cfg(feature = "gpu")]
use leto::{Array3, Array4};

/// Streaming buffer for real-time data processing.
///
/// Implements a circular buffer for continuous RF data acquisition,
/// supporting 4D ultrasound (3D volumes over time) with minimal latency.
///
/// # Memory Layout
/// - RF data stored as (frames × channels × samples × 1)
/// - Circular write/read pointers for lock-free operation
/// - Configurable capacity for latency/throughput tradeoff
///
/// # References
/// - Jensen & Svendsen (1992): "Real-time ultrasound imaging systems"
/// - Tanter & Fink (2014): "Ultrafast imaging in biomedical ultrasound"
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct StreamingBuffer {
    /// RF data buffer (frames × channels × samples × 1)
    rf_buffer: Array4<f32>,
    /// Current write position
    write_pos: usize,
    /// Current read position
    read_pos: usize,
    /// Buffer capacity (number of frames)
    capacity: usize,
}

#[cfg(feature = "gpu")]
impl StreamingBuffer {
    /// Create new streaming buffer with specified capacity.
    ///
    /// # Arguments
    /// - `frames`: Number of frames to buffer
    /// - `channels`: Number of transducer channels
    /// - `samples`: Number of samples per channel
    pub fn new(frames: usize, channels: usize, samples: usize) -> Self {
        Self {
            rf_buffer: Array4::<f32>::zeros((frames, channels, samples, 1)),
            write_pos: 0,
            read_pos: 0,
            capacity: frames,
        }
    }

    /// Add a frame to the streaming buffer.
    ///
    /// Copies frame data into the circular buffer at the current write position.
    /// Advances write pointer modulo capacity.
    ///
    /// # Returns
    /// - `Ok(true)`: Buffer is full (ready for processing)
    /// - `Ok(false)`: Buffer still accumulating frames
    /// # Errors
    /// Returns `KwaversError::InvalidInput` if frame dimensions don't match the buffer configuration.
    pub fn add_frame(&mut self, frame: &Array3<f32>) -> KwaversResult<bool> {
        let frame_dims = frame.shape();
        let channels = frame_dims[0];
        let samples = frame_dims[1];

        let expected_channels = self.rf_buffer.shape()[1];
        let expected_samples = self.rf_buffer.shape()[2];

        if channels != expected_channels || samples != expected_samples {
            return Err(KwaversError::InvalidInput(format!(
                "Frame dimension mismatch: expected ({}, {}), got ({}, {})",
                expected_channels, expected_samples, channels, samples
            )));
        }

        for c in 0..channels {
            for s in 0..samples {
                self.rf_buffer[[self.write_pos, c, s, 0]] = frame[[c, s, 0]];
            }
        }

        self.write_pos = (self.write_pos + 1) % self.capacity;
        Ok(self.write_pos == self.read_pos)
    }

    /// Get current volume data from the buffer.
    ///
    /// Returns reference to the complete buffered RF data volume.
    pub fn get_volume_data(&self) -> &Array4<f32> {
        &self.rf_buffer
    }

    // Coherent ring-buffer inspection API, exercised by the gpu test module but
    // not yet consumed by non-test production code; kept as the buffer's state
    // accessors rather than deleted piecemeal.
    #[allow(dead_code)]
    /// Get buffer capacity (number of frames).
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current write position.
    #[allow(dead_code)] // ring-buffer accessor; test-only consumer (see capacity)
    pub fn write_position(&self) -> usize {
        self.write_pos
    }

    /// Get current read position.
    #[allow(dead_code)] // ring-buffer accessor; test-only consumer (see capacity)
    pub fn read_position(&self) -> usize {
        self.read_pos
    }

    /// Reset buffer to initial state.
    #[allow(dead_code)] // ring-buffer accessor; test-only consumer (see capacity)
    pub fn reset(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.rf_buffer.fill(0.0);
    }

    /// Get RF buffer size in bytes.
    ///
    /// Called from `metrics::calculate_cpu_memory_usage`.
    pub fn rf_buffer_size_bytes(&self) -> usize {
        self.rf_buffer.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_buffer_creation() {
        let buffer = StreamingBuffer::new(16, 1024, 2048);
        assert_eq!(buffer.capacity(), 16);
        assert_eq!(buffer.write_position(), 0);
        assert_eq!(buffer.read_position(), 0);
    }

    #[test]
    fn test_streaming_buffer_add_frame() {
        let mut buffer = StreamingBuffer::new(4, 8, 16);
        let frame = Array3::<f32>::zeros((8, 16, 1));

        assert!(!buffer.add_frame(&frame).unwrap());
        assert!(!buffer.add_frame(&frame).unwrap());
        assert!(!buffer.add_frame(&frame).unwrap());

        // Fourth frame fills the buffer — write_pos wraps to 0 == read_pos.
        assert!(buffer.add_frame(&frame).unwrap());
    }

    #[test]
    fn test_streaming_buffer_dimension_mismatch() {
        let mut buffer = StreamingBuffer::new(4, 8, 16);
        let wrong_frame = Array3::<f32>::zeros((10, 16, 1));
        assert!(buffer.add_frame(&wrong_frame).is_err());
    }

    #[test]
    fn test_streaming_buffer_reset() {
        let mut buffer = StreamingBuffer::new(4, 8, 16);
        let frame = Array3::<f32>::ones((8, 16, 1));

        buffer.add_frame(&frame).unwrap();
        buffer.reset();

        assert_eq!(buffer.write_position(), 0);
        assert_eq!(buffer.read_position(), 0);
    }

    #[test]
    fn test_rf_buffer_size_bytes() {
        let buffer = StreamingBuffer::new(4, 8, 16);
        // 4 frames × 8 channels × 16 samples × 1 × 4 bytes/f32
        assert_eq!(buffer.rf_buffer_size_bytes(), 4 * 8 * 16 * 4);
    }
}
