//! Image-quality and pulse-compression metrics for the advanced ultrasound imaging demo.

use leto::{Array1, Array2};

/// Analyze image quality metrics
pub(crate) struct ImageStats {
    pub(crate) max_value: f64,
    pub(crate) mean_value: f64,
    pub(crate) dynamic_range: f64,
}

pub(crate) fn analyze_image_quality(image: &Array2<f64>) -> ImageStats {
    let mut max_val = 0.0f64;
    let mut sum = 0.0f64;
    let mut count = 0usize;

    for &val in image.iter() {
        max_val = max_val.max(val);
        sum += val;
        count += 1;
    }

    let mean_val = sum / count as f64;
    let dynamic_range = if mean_val > 0.0 {
        20.0 * (max_val / mean_val).log10()
    } else {
        0.0
    };

    ImageStats {
        max_value: max_val,
        mean_value: mean_val,
        dynamic_range,
    }
}

/// Analyze pulse compression results
pub(crate) struct CompressionStats {
    pub(crate) compression_ratio: f64,
    pub(crate) peak_sidelobe_db: f64,
    pub(crate) main_lobe_width: usize,
}

pub(crate) fn analyze_pulse_compression(
    original: &Array1<f64>,
    compressed: &Array1<f64>,
) -> CompressionStats {
    // Find peak in compressed signal
    let mut peak_idx = 0;
    let mut peak_value = 0.0f64;

    for (i, &val) in compressed.iter().enumerate() {
        if val > peak_value {
            peak_value = val;
            peak_idx = i;
        }
    }

    // Find main lobe width (points above half maximum)
    let half_max = peak_value / 2.0;
    let mut start_idx = peak_idx;
    let mut end_idx = peak_idx;

    while start_idx > 0 && compressed[start_idx] > half_max {
        start_idx -= 1;
    }
    while end_idx < compressed.len() - 1 && compressed[end_idx] > half_max {
        end_idx += 1;
    }

    let main_lobe_width = end_idx - start_idx;

    // Find peak sidelobe
    let mut max_sidelobe = 0.0f64;
    for (i, &val) in compressed.iter().enumerate() {
        if i < start_idx || i > end_idx {
            max_sidelobe = max_sidelobe.max(val);
        }
    }

    let peak_sidelobe_db = if max_sidelobe > 0.0 {
        20.0 * (max_sidelobe / peak_value).log10()
    } else {
        -100.0
    };

    let compression_ratio = original.len() as f64 / main_lobe_width as f64;

    CompressionStats {
        compression_ratio,
        peak_sidelobe_db,
        main_lobe_width,
    }
}
