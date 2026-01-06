//! Individual transducer element modeling

use std::f64::consts::PI;

/// Individual transducer element with physics-based modeling
#[derive(Debug, Clone)]
pub struct TransducerElement {
    /// Element ID
    pub id: usize,
    /// Position (x, y, z) \[m\]
    pub position: (f64, f64, f64),
    /// Width \[m\]
    pub width: f64,
    /// Height \[m\]
    pub height: f64,
    /// Phase delay \[rad\]
    pub phase_delay: f64,
    /// Amplitude weight [0.0-1.0]
    pub amplitude_weight: f64,
    /// Element sensitivity pattern
    pub sensitivity: ElementSensitivity,
}

impl TransducerElement {
    /// Create element at specified position
    #[must_use]
    pub fn at_position(id: usize, position: (f64, f64, f64), width: f64, height: f64) -> Self {
        Self {
            id,
            position,
            width,
            height,
            phase_delay: 0.0,
            amplitude_weight: 1.0,
            sensitivity: ElementSensitivity::default(),
        }
    }

    /// Calculate directivity at given angle
    #[must_use]
    pub fn directivity(&self, theta: f64, frequency: f64, sound_speed: f64) -> f64 {
        // Rectangular piston directivity pattern
        let wavelength = sound_speed / frequency;
        let ka = PI * self.width / wavelength;

        if theta.abs() < 1e-10 {
            1.0
        } else {
            let x = ka * theta.sin();
            (x.sin() / x).abs()
        }
    }

    /// Apply phase and amplitude to signal
    #[must_use]
    pub fn apply_modulation(&self, signal: f64, time: f64) -> f64 {
        signal * self.amplitude_weight * (2.0 * PI * time + self.phase_delay).cos()
    }
}

/// Element sensitivity pattern modeling
#[derive(Debug, Clone)]
pub struct ElementSensitivity {
    /// Main lobe width \[rad\]
    pub main_lobe_width: f64,
    /// Side lobe level \[dB\]
    pub side_lobe_level: f64,
    /// Frequency response coefficients
    pub frequency_response: Vec<f64>,
}

impl Default for ElementSensitivity {
    fn default() -> Self {
        Self {
            main_lobe_width: PI / 6.0, // 30 degrees
            side_lobe_level: -20.0,    // -20 dB typical
            frequency_response: vec![
                1.0,  // Normalized at center frequency
                0.85, // -1.5 dB at ±20%
                0.65, // -3.7 dB at ±40%
                0.4,  // -8 dB at ±60%
            ],
        }
    }
}

impl ElementSensitivity {
    /// Get frequency-dependent sensitivity
    #[must_use]
    pub fn at_frequency(&self, frequency: f64, center_frequency: f64) -> f64 {
        let normalized_freq = (frequency - center_frequency).abs() / center_frequency;
        let index = (normalized_freq * (self.frequency_response.len() - 1) as f64) as usize;

        if index < self.frequency_response.len() {
            self.frequency_response[index]
        } else {
            0.0 // Outside response range
        }
    }

    /// Calculate beam width at -3dB
    #[must_use]
    pub fn beam_width_3db(&self) -> f64 {
        self.main_lobe_width * 0.886 // Approximate -3dB width
    }
}
