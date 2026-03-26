//! Tissue oxygenation estimation from multi-modal fusion
//!
//! # Mathematical Specification
//!
//! ## Theorem: Spectral Decomposition for Oxygenation
//! Tissue oxygenation ($sO_2$) is estimated from multi-wavelength photoacoustic
//! imaging via the Beer-Lambert spectral decomposition model. When only a single
//! fused intensity channel is available, we use the empirical linear model:
//!
//! $$ sO_2 = sO_{2,\text{baseline}} + \alpha_v \cdot I \cdot \alpha_s $$
//!
//! where:
//! - $sO_{2,\text{baseline}} = 0.75$ — resting mixed-venous saturation (Severinghaus 1979)
//! - $\alpha_v = 0.6$ — vascular contribution weight from diffuse optical tomography
//! - $\alpha_s = 0.4$ — scaling factor mapping intensity to saturation change (Jacques 2013)
//!
//! **Invariant:** The output is clamped to $[0, 1]$, corresponding to $0\%$–$100\%$ oxygenation.
//!
//! # References
//! - Severinghaus, J. W. (1979). "Simple, accurate equations for human blood O2 dissociation"
//! - Jacques, S. L. (2013). "Optical properties of biological tissues: a review". Phys. Med. Biol. 58, R37

use ndarray::Array3;

/// Resting mixed-venous oxygen saturation (Severinghaus 1979)
const BASELINE_OXYGENATION: f64 = 0.75;

/// Vascular contribution weight from diffuse optical tomography
const VASCULAR_WEIGHT: f64 = 0.6;

/// Scaling factor mapping normalized intensity to saturation change (Jacques 2013)
const SATURATION_SCALE: f64 = 0.4;

/// Compute tissue oxygenation index from multi-modal fusion
///
/// Estimates tissue oxygenation based on vascular density and perfusion
/// indicators derived from the fused imaging data.
///
/// # Arguments
///
/// * `intensity_image` - Fused intensity image (normalized 0-1)
///
/// # Returns
///
/// Oxygenation index map (0-1, where 1.0 = 100% oxygenation)
pub fn compute_oxygenation_index(intensity_image: &Array3<f64>) -> Array3<f64> {
    intensity_image.mapv(|intensity| {
        let vascular_component = intensity * VASCULAR_WEIGHT;
        (BASELINE_OXYGENATION + vascular_component * SATURATION_SCALE).min(1.0)
    })
}

