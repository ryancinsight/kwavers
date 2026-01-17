//! Streaming buffer for real-time 4D ultrasound processing
//!
//! Implements circular buffering for continuous RF data acquisition and
//! processing in real-time volumetric imaging applications.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array3, Array4};

/// Streaming buffer for real-time data processing
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

impl StreamingBuffer {
    /// Create new streaming buffer with specified capacity
    ///
    /// # Arguments
    /// - `frames`: Number of frames to buffer
    /// - `channels`: Number of transducer channels
    /// - `samples`: Number of samples per channel
    ///
    /// # Returns
    /// Initialized streaming buffer with zero-filled data
    ///
    /// # Example
    /// ```ignore
    /// let buffer = StreamingBuffer::new(16, 1024, 2048);
    /// ```
    pub fn new(frames: usize, channels: usize, samples: usize) -> Self {
        Self {
            rf_buffer: Array4::<f32>::zeros((frames, channels, samples, 1)),
            write_pos: 0,
            read_pos: 0,
            capacity: frames,
        }
    }

    /// Add a frame to the streaming buffer
    ///
    /// Copies frame data into the circular buffer at the current write position.
    /// Advances write pointer modulo capacity.
    ///
    /// # Arguments
    /// - `frame`: RF data frame (channels × samples × 1)
    ///
    /// # Returns
    /// - `Ok(true)`: Buffer is full (ready for processing)
    /// - `Ok(false)`: Buffer still accumulating frames
    /// - `Err`: Frame dimensions mismatch
    ///
    /// # Errors
    /// Returns error if frame dimensions don't match buffer configuration.
    pub fn add_frame(&mut self, frame: &Array3<f32>) -> KwaversResult<bool> {
        let (channels, samples, _) = frame.dim();

        // Validate frame dimensions against buffer
        let expected_channels = self.rf_buffer.dim().1;
        let expected_samples = self.rf_buffer.dim().2;

        if channels != expected_channels || samples != expected_samples {
            return Err(KwaversError::InvalidInput(format!(
                "Frame dimension mismatch: expected ({}, {}), got ({}, {})",
                expected_channels, expected_samples, channels, samples
            )));
        }

        // Copy frame data into buffer at write position
        for c in 0..channels {
            for s in 0..samples {
                self.rf_buffer[[self.write_pos, c, s, 0]] = frame[[c, s, 0]];
            }
        }

        // Advance write pointer (circular)
        self.write_pos = (self.write_pos + 1) % self.capacity;

        // Check if buffer is full (write caught up to read)
        Ok(self.write_pos == self.read_pos)
    }

    /// Get current volume data from the buffer
    ///
    /// Returns reference to the complete buffered RF data volume.
    /// Used for volumetric reconstruction after buffer fills.
    ///
    /// # Returns
    /// Reference to RF buffer (frames × channels × samples × 1)
    pub fn get_volume_data(&self) -> &Array4<f32> {
        &self.rf_buffer
    }

    /// Get buffer capacity (number of frames)
    #[allow(dead_code)]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current write position
    #[allow(dead_code)]
    pub fn write_position(&self) -> usize {
        self.write_pos
    }

    /// Get current read position
    #[allow(dead_code)]
    pub fn read_position(&self) -> usize {
        self.read_pos
    }

    /// Reset buffer to initial state
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.write_pos = 0;
        self.read_pos = 0;
        self.rf_buffer.fill(0.0);
    }

    /// Get RF buffer size in bytes
    #[allow(dead_code)]
    pub fn rf_buffer_size_bytes(&self) -> usize {
        self.rf_buffer.len() * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
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

        // First 3 frames should not fill the buffer
        assert!(!buffer.add_frame(&frame).unwrap());
        assert!(!buffer.add_frame(&frame).unwrap());
        assert!(!buffer.add_frame(&frame).unwrap());

        // 4th frame fills the buffer
        assert!(buffer.add_frame(&frame).unwrap());
    }

    #[test]
    fn test_streaming_buffer_dimension_mismatch() {
        let mut buffer = StreamingBuffer::new(4, 8, 16);
        let wrong_frame = Array3::<f32>::zeros((10, 16, 1)); // Wrong channels

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
        let size_bytes = buffer.rf_buffer_size_bytes();

        // 4 frames × 8 channels × 16 samples × 1 × 4 bytes/f32
        let expected_size = (4 * 8 * 16) * 4;
        assert_eq!(size_bytes, expected_size);
    }
}
