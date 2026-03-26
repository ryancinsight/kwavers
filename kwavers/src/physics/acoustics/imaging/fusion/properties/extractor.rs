//! Property extractor orchestrator
//!
//! # Mathematical Specifications
//!
//! ## Theorem 1: Oxygenation Saturation ($sO_2$)
//! Optical fluence maps are decomposed into relative concentrations of oxy-hemoglobin ($HbO_2$)
//! and deoxy-hemoglobin ($Hb$). The oxygenation index is the scalar ratio:
//!
//! $$ sO_2 = \frac{C_{HbO_2}}{C_{HbO_2} + C_{Hb}} $$
//!
//! **Invariant:** The algorithm enforces the physical bound $0 \le sO_2 \le 1$.
//!
//! ## Theorem 2: Composite Stiffness
//! Acoustic elastography measures shear wave speed $c_s$. The corresponding Young's
//! modulus $E$ in nearly incompressible tissue (Poisson ratio $\nu \approx 0.5$) is derived via:
//!
//! $$ E = 3 \rho c_s^2 $$
//!
//! where $\rho$ is the spatial tissue density map.

use super::classification::classify_tissue_types;
use super::mechanical::compute_composite_stiffness;
use super::oxygenation::compute_oxygenation_index;
use super::super::types::FusedImageResult;
use ndarray::Array3;
use std::collections::HashMap;

/// Extract comprehensive tissue properties from fused imaging data
///
/// Analyzes the fused multi-modal data to derive clinically relevant
/// tissue properties including classification, oxygenation, and stiffness.
///
/// # Arguments
///
/// * `fused_result` - Fused imaging result containing intensity and metadata
///
/// # Returns
///
/// HashMap mapping property names to 3D spatial maps
pub fn extract_tissue_properties(fused_result: &FusedImageResult) -> HashMap<String, Array3<f64>> {
    let mut properties = HashMap::new();

    // Extract derived tissue properties from multi-modal fusion
    properties.insert(
        "tissue_classification".to_string(),
        classify_tissue_types(&fused_result.intensity_image),
    );

    properties.insert(
        "oxygenation_index".to_string(),
        compute_oxygenation_index(&fused_result.intensity_image),
    );

    properties.insert(
        "composite_stiffness".to_string(),
        compute_composite_stiffness(&fused_result.intensity_image),
    );

    properties
}
