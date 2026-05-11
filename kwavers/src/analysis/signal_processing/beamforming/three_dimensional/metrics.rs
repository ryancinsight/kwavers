//! Performance Metrics and Memory Usage Calculation
//!
//! Utilities for tracking and calculating memory consumption for GPU and CPU
//! resources used in 3D beamforming operations.

#[cfg(feature = "gpu")]
use crate::analysis::signal_processing::beamforming::three_dimensional::config::BeamformingConfig3D;
#[cfg(feature = "gpu")]
use crate::analysis::signal_processing::beamforming::three_dimensional::streaming::StreamingBuffer;

/// Calculate GPU memory usage based on configuration
///
/// Estimates memory usage for:
/// - RF data buffers (streaming buffer size × elements × samples)
/// - Output volume buffers (voxel count)
/// - Intermediate computation buffers
///
/// # Arguments
/// * `config` - Beamforming configuration
///
/// # Returns
/// Estimated GPU memory usage in MB
///
/// Called from `processor.rs::calculate_gpu_memory_usage()` (GPU builds) and from unit tests.
#[cfg(feature = "gpu")]
pub fn calculate_gpu_memory_usage(config: &BeamformingConfig3D) -> f64 {
    // RF data buffer size
    // Memory estimation for RF data buffering
    // Accounts for streaming buffer requirements in real-time processing
    let rf_data_size = config.streaming_buffer_size
        * config.num_elements_3d.0
        * config.num_elements_3d.1
        * config.num_elements_3d.2
        * 1024 // samples per channel
        * std::mem::size_of::<f32>();

    // Output volume size
    let volume_size = config.volume_dims.0
        * config.volume_dims.1
        * config.volume_dims.2
        * std::mem::size_of::<f32>();

    // Apodization weights buffer
    let apodization_size = config.num_elements_3d.0
        * config.num_elements_3d.1
        * config.num_elements_3d.2
        * std::mem::size_of::<f32>();

    // Element positions buffer (3 floats per element)
    let element_positions_size = config.num_elements_3d.0
        * config.num_elements_3d.1
        * config.num_elements_3d.2
        * 3
        * std::mem::size_of::<f32>();

    // Parameters uniform buffer (small)
    let params_size = 128; // Conservative estimate for Params struct

    let total_bytes =
        rf_data_size + volume_size + apodization_size + element_positions_size + params_size;

    total_bytes as f64 / (1024.0 * 1024.0) // Convert to MB
}

/// Calculate CPU memory usage for streaming buffers (GPU feature enabled)
///
/// # Arguments
/// * `streaming_buffer` - Optional streaming buffer
///
/// # Returns
/// CPU memory usage in MB
#[cfg(feature = "gpu")]
pub fn calculate_cpu_memory_usage(streaming_buffer: &Option<StreamingBuffer>) -> f64 {
    // Memory usage calculation for streaming buffers
    let rf_data_size = if let Some(buffer) = streaming_buffer {
        buffer.rf_buffer_size_bytes()
    } else {
        0
    };

    rf_data_size as f64 / (1024.0 * 1024.0) // Convert to MB
}

/// Calculate CPU memory usage (CPU-only build).
///
/// Returns `0.0`: no streaming buffer exists in CPU-only builds.
#[cfg(not(feature = "gpu"))]
pub fn calculate_cpu_memory_usage(_streaming_buffer: &Option<()>) -> f64 {
    // No GPU buffers in CPU-only mode
    0.0
}

/// GPU memory tests — only valid when GPU feature is enabled.
#[cfg(all(test, feature = "gpu"))]
mod gpu_tests {
    use super::*;

    #[test]
    fn test_gpu_memory_calculation() {
        let config = BeamformingConfig3D::default();
        let memory_mb = calculate_gpu_memory_usage(&config);

        // Analytical bounds:
        // Default: 128×128×128 volume (~8 MB) + 16 frames × 32×32×16 elements × 1024 samples × 4 B (~512 MB)
        let expected_volume = 128.0 * 128.0 * 128.0 * 4.0 / (1024.0 * 1024.0);
        let expected_rf = 16.0 * 32.0 * 32.0 * 16.0 * 1024.0 * 4.0 / (1024.0 * 1024.0);
        assert!(memory_mb > 0.0);
        assert!(memory_mb > expected_volume);
        assert!(memory_mb < expected_rf * 2.0);
    }

    #[test]
    fn test_memory_scales_with_config() {
        let config1 = BeamformingConfig3D {
            volume_dims: (64, 64, 64),
            ..Default::default()
        };
        let config2 = BeamformingConfig3D {
            volume_dims: (128, 128, 128),
            ..Default::default()
        };
        // 128³ volume is 8× larger than 64³ — larger estimate expected.
        assert!(calculate_gpu_memory_usage(&config2) > calculate_gpu_memory_usage(&config1));
    }
}

/// CPU memory tests — run in both feature configurations.
#[cfg(test)]
mod cpu_tests {
    use super::*;

    #[test]
    fn test_cpu_memory_calculation_no_buffer() {
        let memory_mb = calculate_cpu_memory_usage(&None);
        assert_eq!(memory_mb, 0.0);
    }
}
