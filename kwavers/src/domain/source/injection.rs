//! Source injection mode definitions
//!
//! This module defines how source amplitudes are applied to wave fields during simulation.
//! Different injection modes are appropriate for different source geometries and numerical methods.
//!
//! # Mathematical Specification
//!
//! ## Boundary Mode
//! For boundary plane sources (e.g., plane wave at z=0):
//! - **Dirichlet enforcement**: p(boundary) = amplitude(t)
//! - Used in FDTD for boundary sources to enforce prescribed boundary conditions
//! - Not suitable for spectral methods (PSTD) due to periodic boundary conditions
//!
//! ## Additive Mode
//! For interior and volume sources (e.g., point sources, transducers):
//! - **Normalized injection**: p += (mask / ||mask||) * amplitude(t)
//! - Where ||mask|| is the L1 norm: ||mask|| = Σᵢⱼₖ |mask[i,j,k]|
//! - Preserves energy scaling and ensures amplitude independence from discretization
//! - Used for all sources in PSTD and interior sources in FDTD
//!
//! # Design Rationale
//!
//! This enum is shared across FDTD and PSTD solvers to maintain architectural consistency
//! and avoid code duplication (Single Source of Truth principle). Each solver implements
//! its own `determine_injection_mode` logic based on its numerical properties:
//!
//! - **FDTD**: Can use both Boundary and Additive modes depending on source geometry
//! - **PSTD**: Always uses Additive mode due to FFT periodicity constraints
//!
//! # References
//!
//! - Treeby & Cox (2010), "Modeling power law absorption and dispersion for acoustic
//!   propagation using the fractional Laplacian", J. Acoust. Soc. Am. 127(5), 2741-2748
//! - k-Wave User Manual, Section on source conditions and injection methods

/// Source injection mode determines how source amplitudes are applied to fields
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceInjectionMode {
    /// Boundary plane source: enforce Dirichlet condition (p = amplitude)
    ///
    /// Used for sources that should prescribe the field value at a boundary plane.
    /// Appropriate for plane wave sources at domain boundaries in FDTD.
    ///
    /// **Note**: Not suitable for spectral methods (PSTD) due to periodic boundary conditions.
    Boundary,

    /// Interior source: additive injection with normalization scale
    ///
    /// Used for point sources, volume sources, and transducer arrays in the domain interior.
    /// The scale factor normalizes by the L1 norm of the source mask to preserve energy scaling.
    ///
    /// Formula: `p += scale * mask * amplitude(t)` where `scale = 1 / ||mask||₁`
    Additive {
        /// Normalization scale factor (typically 1/||mask||₁)
        scale: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_injection_mode_equality() {
        let boundary1 = SourceInjectionMode::Boundary;
        let boundary2 = SourceInjectionMode::Boundary;
        assert_eq!(boundary1, boundary2);

        let additive1 = SourceInjectionMode::Additive { scale: 1.0 };
        let additive2 = SourceInjectionMode::Additive { scale: 1.0 };
        assert_eq!(additive1, additive2);

        assert_ne!(boundary1, additive1);
    }

    #[test]
    fn test_injection_mode_clone() {
        let mode = SourceInjectionMode::Additive { scale: 0.5 };
        let cloned = mode;
        assert_eq!(mode, cloned);
    }

    #[test]
    fn test_additive_scale_extraction() {
        let mode = SourceInjectionMode::Additive { scale: 2.5 };
        if let SourceInjectionMode::Additive { scale } = mode {
            assert_eq!(scale, 2.5);
        } else {
            panic!("Expected Additive mode");
        }
    }
}
