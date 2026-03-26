//! Mechanical properties estimation (stiffness and density)
//!
//! # Mathematical Specification
//!
//! ## Theorem 1: Acoustic-to-Elastographic Coupling
//! Tissue stiffness (Young's modulus $E$) is estimated via an empirical coupling to acoustic 
//! impedance (often proxied by normalized backscatter intensity $I$). Standard soft 
//! tissue baseline stiffness typically centers around $E_0 \approx 20$ kPa. 
//! Pathological dense tissue (e.g., tumors) exhibits both higher acoustic scattering 
//! and elevated localized stiffness ($E_{max} \approx 60$ kPa).
//!
//! $$ E(I) = E_0 \cdot \left(1 + \kappa \cdot (1 - I)\right) $$
//!
//! where:
//! - $E_0 = 20.0$ kPa
//! - $\kappa = 2.0$ — stiffness-intensity coupling factor
//! - $I \in [0, 1]$ — normalized acoustic intensity pattern
//!
//! ## Theorem 2: Relative Density Index
//! Tissue density correlates nonlinearly with acoustic intensity, often modeled
//! using a power law with exponent $\gamma$:
//!
//! $$ \rho_{rel} = I^{\gamma} $$
//!
//! where $\gamma = 0.5$ accounts for the typical pressure-to-intensity square root relation.
//!
//! # References
//! - Ophir, J., et al. (1991). "Elastography: A quantitative method for imaging the elasticity of biological tissues." Ultrasonic Imaging, 13(2), 111-134.
//! - Sarvazyan, A. P., et al. (1998). "Shear wave elasticity imaging: a new ultrasonic technology of medical diagnostics." Ultrasound in Medicine & Biology.

use ndarray::Array3;

/// Compute composite tissue stiffness from multi-modal correlation
///
/// Estimates tissue mechanical properties (stiffness) by correlating
/// acoustic properties with elastography data. Different tissue types
/// exhibit characteristic relationships between acoustic impedance and
/// mechanical stiffness.
///
/// Baseline soft tissue stiffness (Young's modulus) in kPa
pub(crate) const BASELINE_SOFT_TISSUE_STIFFNESS_KPA: f64 = 20.0;

/// Empirical coupling factor linking inverse intensity to stiffness map scaling
pub(crate) const STIFFNESS_INTENSITY_COUPLING: f64 = 2.0;

/// Exponent for the power-law empirical correlation between acoustic intensity and tissue density
pub(crate) const DENSITY_NONLINEARITY_EXPONENT: f64 = 0.5;

/// Compute composite tissue stiffness from multi-modal correlation
///
/// Estimates tissue mechanical properties (stiffness) by correlating
/// acoustic properties with elastography data. Different tissue types
/// exhibit characteristic relationships between acoustic impedance and
/// mechanical stiffness.
///
/// # Mechanical Model
///
/// - Normal soft tissue: ~10-50 kPa
/// - Abnormal/pathological tissue: typically higher stiffness
/// - Inverse correlation with intensity often observed (denser tissue)
///
/// # Arguments
///
/// * `intensity_image` - Fused intensity image (normalized 0-1)
///
/// # Returns
///
/// Stiffness map in kPa (typical range: 20-60 kPa if min_intensity=0.0)
pub fn compute_composite_stiffness(intensity_image: &Array3<f64>) -> Array3<f64> {
    intensity_image.mapv(|intensity| {
        let intensity_factor = 1.0 - intensity;
        BASELINE_SOFT_TISSUE_STIFFNESS_KPA * (1.0 + intensity_factor * STIFFNESS_INTENSITY_COUPLING)
    })
}

/// Compute tissue density index from intensity patterns
///
/// Estimates relative tissue density based on acoustic scattering and
/// absorption characteristics from the fused multi-modal data.
///
/// # Arguments
///
/// * `intensity_image` - Fused intensity image (normalized 0-1)
///
/// # Returns
///
/// Relative density map (normalized to [0, 1])
pub fn compute_tissue_density(intensity_image: &Array3<f64>) -> Array3<f64> {
    intensity_image.mapv(|intensity| intensity.powf(DENSITY_NONLINEARITY_EXPONENT))
}
