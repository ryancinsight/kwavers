//! Elastic properties trait for solid media
//!
//! This module defines traits for elastic wave propagation in solid media,
//! including Lamé parameters and wave speeds.

use crate::domain::grid::Grid;
use crate::domain::medium::core::{ArrayAccess, CoreMedium};
use ndarray::Array3;

/// Trait for elastic medium properties
pub trait ElasticProperties: CoreMedium {
    /// Returns Lamé's first parameter λ (Pa)
    fn lame_lambda(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Returns Lamé's second parameter μ (shear modulus) (Pa)
    fn lame_mu(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64;

    /// Calculates shear wave speed (m/s)
    fn shear_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let mu = self.lame_mu(x, y, z, grid);
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let rho = self.density(i, j, k);
        if rho > 0.0 {
            (mu / rho).sqrt()
        } else {
            0.0
        }
    }

    /// Calculates compressional wave speed (m/s)
    fn compressional_wave_speed(&self, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        let lambda = self.lame_lambda(x, y, z, grid);
        let mu = self.lame_mu(x, y, z, grid);
        let (i, j, k) = crate::domain::medium::continuous_to_discrete(x, y, z, grid);
        let rho = self.density(i, j, k);
        if rho > 0.0 {
            ((lambda + 2.0 * mu) / rho).sqrt()
        } else {
            0.0
        }
    }
}

/// Trait for array-based elastic property access
pub trait ElasticArrayAccess: ElasticProperties + ArrayAccess {
    /// Returns a 3D array of Lamé's first parameter λ values (Pa)
    fn lame_lambda_array(&self) -> Array3<f64>;

    /// Returns a 3D array of Lamé's second parameter μ values (Pa)
    fn lame_mu_array(&self) -> Array3<f64>;

    /// Returns a 3D array of shear wave speeds (m/s)
    fn shear_sound_speed_array(&self) -> Array3<f64> {
        let mu_arr = self.lame_mu_array();
        let shape = mu_arr.dim();

        // TODO_AUDIT: P1 - Elastic Medium Shear Sound Speed - Zero-Returning Default Implementation
        //
        // PROBLEM:
        // Default trait implementation returns Array3::zeros(shape), providing zero shear wave
        // speed for all points. This is physically incorrect for elastic media and masks missing
        // implementations in concrete types.
        //
        // IMPACT:
        // - Elastic wave propagation simulations will fail (zero shear speed → infinite time step)
        // - Silent error: no compilation warning for types that don't override this method
        // - Incorrect physics: shear waves require non-zero c_s = sqrt(μ/ρ)
        // - Blocks applications: elastography, elastic wave imaging, seismology
        //
        // REQUIRED IMPLEMENTATION OPTIONS:
        //
        // Option A (Recommended): Remove default implementation, make method required
        // - Forces all implementing types to provide correct shear speed
        // - Compilation error if not implemented (catch at build time)
        // - Type safety enforces correctness
        //
        // Option B: Compute from Lamé parameters and density
        // - Requires density_array() method in trait
        // - Compute: c_s = sqrt(μ / ρ) element-wise
        // - Still requires implementors to provide μ and ρ
        //
        // MATHEMATICAL SPECIFICATION:
        // Shear wave speed in elastic medium:
        //   c_s = sqrt(μ / ρ)
        // where:
        //   - μ is Lamé's second parameter (shear modulus, Pa)
        //   - ρ is mass density (kg/m³)
        //
        // RECOMMENDED APPROACH:
        // 1. Remove this default implementation entirely
        // 2. Make shear_sound_speed_array() a required trait method
        // 3. Update all implementations:
        //    - HomogeneousElastic: return constant-filled array
        //    - HeterogeneousElastic: compute from stored μ and ρ arrays
        //    - CTBasedElastic: compute from CT-derived properties
        // 4. Add validation: assert c_s > 0 for all elements
        //
        // VALIDATION CRITERIA:
        // 1. Compilation fails if any type doesn't implement the method
        // 2. Unit tests: verify c_s = sqrt(μ/ρ) for known materials
        // 3. Property test: c_s should be in range [500, 5000] m/s for biological tissues
        // 4. Integration test: elastic wave solver runs without NaN/infinity
        //
        // REFERENCES:
        // - Landau & Lifshitz, "Theory of Elasticity" (1986), §24
        // - Graff, "Wave Motion in Elastic Solids" (1975), Ch. 1
        // - backlog.md: Sprint 211 Elastic Wave Migration
        //
        // EFFORT: ~4-6 hours (remove default, update all implementations, testing)
        // SPRINT: Sprint 211 (elastic medium infrastructure)
        //
        // Implementation will be provided by specific types
        Array3::zeros(shape)
    }

    /// Returns a 3D array of shear viscosity coefficients
    fn shear_viscosity_coeff_array(&self) -> Array3<f64> {
        let shape = self.lame_mu_array().dim();

        // TODO_AUDIT: P2 - Elastic Medium Shear Viscosity - Zero-Returning Default Implementation
        //
        // PROBLEM:
        // Default trait implementation returns Array3::zeros(shape), assuming zero viscosity
        // (perfectly elastic, no attenuation). This is acceptable for lossless simulations but
        // masks missing implementations for viscoelastic media.
        //
        // IMPACT:
        // - Cannot simulate viscoelastic wave propagation (no attenuation)
        // - Zero viscosity is physically unrealistic for biological tissues
        // - Blocks applications: tissue characterization via attenuation measurements
        // - Silent behavior: implementors may forget to override for lossy media
        //
        // REQUIRED IMPLEMENTATION OPTIONS:
        //
        // Option A (Recommended): Keep default as zero with documentation
        // - Zero viscosity is a valid physical model (elastic limit)
        // - Document that this assumes lossless propagation
        // - Types requiring viscosity should override explicitly
        // - Add warning in doc comments about attenuation
        //
        // Option B: Remove default, make method required
        // - Forces all types to explicitly choose viscosity model
        // - More verbose but prevents accidental zero viscosity
        //
        // Option C: Compute from frequency-dependent attenuation
        // - Requires additional trait methods (attenuation_coefficient, frequency)
        // - More complex but physically accurate
        //
        // MATHEMATICAL SPECIFICATION:
        // Viscoelastic shear stress tensor:
        //   τ_ij = μ ∂u_i/∂x_j + η_s ∂(∂u_i/∂x_j)/∂t
        // where:
        //   - η_s is shear viscosity coefficient (Pa·s)
        //   - For biological tissues: η_s ≈ 0.001-1.0 Pa·s
        //
        // RECOMMENDED APPROACH:
        // 1. Keep zero default (acceptable for elastic limit)
        // 2. Add clear documentation: "Returns zero by default (lossless elastic medium)"
        // 3. Add doc warning: "Override for viscoelastic media with attenuation"
        // 4. Provide helper method to compute from Q-factor: η_s = μ/(ω·Q)
        //
        // VALIDATION CRITERIA:
        // 1. Document that zero is physically meaningful (elastic limit)
        // 2. Verify viscoelastic implementations override with non-zero values
        // 3. Test attenuation: wave amplitude decreases exponentially with distance
        //
        // REFERENCES:
        // - Fung, "Biomechanics: Mechanical Properties of Living Tissues" (1993)
        // - Catheline et al., "Measurement of viscoelastic properties" (2004)
        //
        // EFFORT: ~2-3 hours (documentation update, validation tests)
        // SPRINT: Sprint 212 (viscoelastic enhancements)
        //
        Array3::zeros(shape)
    }

    /// Returns a 3D array of bulk viscosity coefficients
    fn bulk_viscosity_coeff_array(&self) -> Array3<f64> {
        let shape = self.lame_mu_array().dim();
        Array3::zeros(shape)
    }
}
