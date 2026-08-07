//! Apodization types for transducer arrays.
//!
//! Defines the `ApodizationType` enum used throughout the kwavers stack
//! for specifying transmit and receive apodization windows.

/// Apodization type for transducer elements.
///
/// Specifies the weighting applied to elements in a transducer array
/// during transmit and receive operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ApodizationType {
    /// Uniform weighting (boxcar)
    #[default]
    Uniform,
    /// Hann/Hanning window
    Hanning,
    /// Hamming window
    Hamming,
    /// Blackman window
    Blackman,
    /// Tukey window with specified cosine fraction `r`
    Tukey { r: f64 },
    /// Gaussian window with specified sigma
    Gaussian { sigma: f64 },
    /// Kaiser window (approximated with Hamming)
    Kaiser { beta: f64 },
}

impl ApodizationType {
    /// Create an apodization implementation that can be used for beamforming.
    ///
    /// Returns a function that applies the apodization weight to a given element.
    #[must_use]
    pub fn weights(&self, _element_index: usize, _num_elements: usize) -> f64 {
        match self {
            Self::Uniform => 1.0,
            Self::Hanning => {
                0.5 * (1.0
                    - (2.0 * std::f64::consts::PI * _element_index as f64
                        / (_num_elements - 1).max(1) as f64)
                        .cos())
            }
            Self::Hamming => {
                0.54 - 0.46
                    * (2.0 * std::f64::consts::PI * _element_index as f64
                        / (_num_elements - 1).max(1) as f64)
                        .cos()
            }
            Self::Blackman => {
                let n = (_num_elements - 1).max(1) as f64;
                let i = _element_index as f64;
                0.42 - 0.5 * (2.0 * std::f64::consts::PI * i / n).cos()
                    + 0.08 * (4.0 * std::f64::consts::PI * i / n).cos()
            }
            Self::Tukey { r } => {
                let r_val = r.clamp(0.0, 1.0);
                let n = (_num_elements - 1).max(1) as f64;
                let i = _element_index as f64;
                let x = i / n;
                if x < r_val / 2.0 {
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * x / r_val).cos())
                } else if x > 1.0 - r_val / 2.0 {
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * (1.0 - x) / r_val).cos())
                } else {
                    1.0
                }
            }
            Self::Gaussian { sigma } => {
                let n = (_num_elements - 1).max(1) as f64;
                let i = _element_index as f64;
                let x = i / n;
                (-0.5 * ((x - 0.5) / sigma).powi(2)).exp()
            }
            Self::Kaiser { .. } => {
                // Kaiser with beta ≈ 0 approximates a rectangular window
                // For simplicity, use Hamming as a close approximation
                0.54 - 0.46
                    * (2.0 * std::f64::consts::PI * _element_index as f64
                        / (_num_elements - 1).max(1) as f64)
                        .cos()
            }
        }
    }
}
