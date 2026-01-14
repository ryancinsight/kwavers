//! Elastography Inverse Solver
//!
//! Reconstructs tissue elasticity from shear wave propagation measurements.
//! Provides both linear and nonlinear inversion algorithms for comprehensive
//! tissue characterization.
//!
//! ## Module Organization
//!
//! This module follows Clean Architecture principles with clear separation of concerns:
//!
//! - **`config`**: Configuration types for inversion algorithms
//! - **`types`**: Domain types and result structures
//! - **`algorithms`**: Shared utility algorithms (smoothing, boundary handling)
//! - **`linear_methods`**: Linear elasticity inversion (TOF, phase gradient, direct)
//! - **`nonlinear_methods`**: Nonlinear parameter estimation (harmonic ratio, least squares, Bayesian)
//!
//! ## Usage Examples
//!
//! ### Linear Elasticity Reconstruction
//!
//! ```rust
//! use kwavers::solver::inverse::elastography::{ShearWaveInversion, ShearWaveInversionConfig};
//! use kwavers::domain::imaging::ultrasound::elastography::InversionMethod;
//! use kwavers::domain::grid::Grid;
//! use kwavers::physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create grid and displacement field
//! let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001)?;
//! let displacement = DisplacementField::zeros(50, 50, 50);
//!
//! // Configure inversion
//! let config = ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight)
//!     .with_density(1050.0); // kg/m³
//!
//! // Run inversion
//! let inversion = ShearWaveInversion::new(config);
//! let elasticity_map = inversion.reconstruct(&displacement, &grid)?;
//!
//! // Access results
//! println!("Young's modulus range: {:?}", elasticity_map.statistics());
//! # Ok(())
//! # }
//! ```
//!
//! ### Nonlinear Parameter Estimation
//!
//! ```rust
//! use kwavers::solver::inverse::elastography::{NonlinearInversion, NonlinearInversionConfig};
//! use kwavers::solver::inverse::elastography::NonlinearParameterMapExt;
//! use kwavers::domain::imaging::ultrasound::elastography::NonlinearInversionMethod;
//! use kwavers::domain::grid::Grid;
//! use kwavers::physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create grid and harmonic displacement field
//! let grid = Grid::new(50, 50, 50, 0.001, 0.001, 0.001)?;
//! let harmonic_field = HarmonicDisplacementField::new(50, 50, 50, 2, 10);
//!
//! // Configure nonlinear inversion
//! let config = NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio)
//!     .with_tissue_properties(1050.0, 1540.0) // density (kg/m³), speed (m/s)
//!     .with_convergence(100, 1e-6); // max iterations, tolerance
//!
//! // Run inversion
//! let inversion = NonlinearInversion::new(config);
//! let parameter_map = inversion.reconstruct(&harmonic_field, &grid)?;
//!
//! // Access nonlinearity parameters
//! let (min_ba, max_ba, mean_ba) = parameter_map.nonlinearity_statistics();
//! println!("B/A range: {} to {} (mean: {})", min_ba, max_ba, mean_ba);
//! # Ok(())
//! # }
//! ```
//!
//! ## Physics Background
//!
//! ### Linear Elasticity
//!
//! For incompressible isotropic materials (Poisson's ratio ν ≈ 0.5):
//!
//! - **Shear modulus**: μ = ρ cs²
//! - **Young's modulus**: E = 3μ
//! - **Bulk modulus**: K → ∞
//!
//! where ρ is density (kg/m³) and cs is shear wave speed (m/s).
//!
//! ### Nonlinear Elasticity
//!
//! **Acoustic nonlinearity parameter (B/A)**:
//!
//! B/A = (8/μ) × (ρ₀c₀³/(βP₀)) × (A₂/A₁)
//!
//! where:
//! - A₁, A₂: fundamental and second harmonic amplitudes
//! - ρ₀: density (kg/m³)
//! - c₀: sound speed (m/s)
//! - μ: shear modulus (Pa)
//! - β: nonlinearity coefficient
//! - P₀: acoustic pressure amplitude (Pa)
//!
//! **Higher-order elastic constants (A, B, C, D)**:
//!
//! Characterize nonlinear stress-strain relationships:
//! σ = E·ε + A·ε² + B·ε³ + C·ε⁴ + ...
//!
//! ## Inversion Methods
//!
//! ### Linear Methods
//!
//! | Method | Speed | Accuracy | Use Case |
//! |--------|-------|----------|----------|
//! | **Time-of-Flight** | Fast | Good | Real-time imaging, homogeneous tissue |
//! | **Phase Gradient** | Medium | Better | Complex geometries, frequency domain |
//! | **Direct Inversion** | Slow | Best | High-quality data, research applications |
//! | **Volumetric TOF** | Medium | Good | 3D volumes, multiple wave sources |
//! | **Directional Phase** | Medium | Better | Anisotropic media, 3D heterogeneity |
//!
//! ### Nonlinear Methods
//!
//! | Method | Speed | Accuracy | Use Case |
//! |--------|-------|----------|----------|
//! | **Harmonic Ratio** | Fast | Good | Real-time, sufficient SNR |
//! | **Least Squares** | Medium | Better | Iterative refinement |
//! | **Bayesian** | Slow | Best | Uncertainty quantification, prior knowledge |
//!
//! ## References
//!
//! ### Foundational Papers
//!
//! - **Bercoff, J., et al.** (2004). "Supersonic shear imaging: a new technique
//!   for soft tissue elasticity mapping." *IEEE Transactions on Ultrasonics,
//!   Ferroelectrics, and Frequency Control*, 51(4), 396-409.
//!   DOI: 10.1109/TUFFC.2004.1295425
//!
//! - **McLaughlin, J., & Renzi, D.** (2006). "Shear wave speed recovery in transient
//!   elastography and supersonic imaging using propagating fronts." *Inverse Problems*,
//!   22(2), 681. DOI: 10.1088/0266-5611/22/2/018
//!
//! ### Advanced Methods
//!
//! - **Deffieux, T., et al.** (2011). "On the effects of reflected waves in transient
//!   shear wave elastography." *IEEE TUFFC*, 58(10), 2032-2035.
//!   DOI: 10.1109/TUFFC.2011.2052
//!
//! - **Parker, K. J., et al.** (2011). "Sonoelasticity of organs: Shear waves ring a bell."
//!   *Journal of Ultrasound in Medicine*, 30(4), 507-515.
//!   DOI: 10.7863/jum.2011.30.4.507
//!
//! - **Urban, M. W., et al.** (2013). "A review of shearwave dispersion ultrasound
//!   vibrometry (SDUV) and its applications." *Current Medical Imaging Reviews*,
//!   8(1), 27-36. DOI: 10.2174/157340512799220625
//!
//! ### Nonlinear Elasticity
//!
//! - **Chen, S., et al.** (2013). "Quantifying elasticity and viscosity from measurement
//!   of shear wave speed dispersion." *Journal of the Acoustical Society of America*,
//!   115(6), 2781-2785. DOI: 10.1121/1.1739480
//!
//! - **Destrade, M., et al.** (2010). "Third- and fourth-order constants of incompressible
//!   soft solids and the acousto-elastic effect." *Journal of the Acoustical Society
//!   of America*, 127(5), 2759-2763. DOI: 10.1121/1.3372624
//!
//! ### Bayesian Methods
//!
//! - **Sullivan, T. J.** (2015). *Introduction to Uncertainty Quantification*.
//!   Springer Texts in Applied Mathematics, Vol. 63. ISBN: 978-3-319-23394-9
//!
//! ## Mathematical Specifications
//!
//! ### Time-of-Flight Method
//!
//! **Theorem**: For homogeneous isotropic elastic medium with constant shear wave
//! speed cs, the arrival time t at distance r from a point source is:
//!
//! t = r / cs
//!
//! **Proof**: Shear waves satisfy the wave equation ∇²u = (1/cs²)∂²u/∂t² with
//! characteristic speed cs. For radial propagation from point source at t=0,
//! the wavefront position satisfies r(t) = cs·t, yielding t = r/cs. ∎
//!
//! ### Phase Gradient Method
//!
//! **Theorem**: For monochromatic shear wave u(x,t) = A·exp(i(kx - ωt)), the
//! wavenumber k relates to phase gradient by:
//!
//! k = ∂φ/∂x
//!
//! where φ(x) = kx is the spatial phase.
//!
//! **Proof**: Phase φ = arg(u) = kx - ωt + φ₀. Taking spatial derivative:
//! ∂φ/∂x = k. Shear wave speed: cs = ω/k. ∎
//!
//! ### Harmonic Ratio Method
//!
//! **Theorem**: For weakly nonlinear wave propagation, the second harmonic
//! amplitude A₂ relates to nonlinearity parameter B/A by:
//!
//! A₂/A₁ ∝ (B/A) × (propagation distance)
//!
//! **Proof**: Perturbation analysis of nonlinear wave equation (Westervelt equation)
//! shows second harmonic grows linearly with distance in weakly nonlinear regime.
//! Proportionality constant depends on B/A. See Hamilton & Blackstock (1998),
//! Chapter 6. ∎

// Public API modules
pub mod algorithms;
pub mod config;
pub mod linear_methods;
pub mod nonlinear_methods;
pub mod types;

// Re-export primary types for convenience
pub use config::{NonlinearInversionConfig, ShearWaveInversionConfig};
pub use linear_methods::ShearWaveInversion;
pub use nonlinear_methods::NonlinearInversion;
pub use types::{elasticity_map_from_speed, ElasticityMapExt, NonlinearParameterMapExt};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::imaging::ultrasound::elastography::{
        InversionMethod, NonlinearInversionMethod,
    };
    use crate::physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;
    use crate::physics::acoustics::imaging::modalities::elastography::HarmonicDisplacementField;

    #[test]
    fn test_linear_inversion_pipeline() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let displacement = DisplacementField::zeros(20, 20, 20);

        let config = ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight);
        let inversion = ShearWaveInversion::new(config);
        let result = inversion.reconstruct(&displacement, &grid);

        assert!(result.is_ok(), "Linear inversion pipeline should succeed");
    }

    #[test]
    fn test_nonlinear_inversion_pipeline() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

        let config = NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio);
        let inversion = NonlinearInversion::new(config);
        let result = inversion.reconstruct(&harmonic_field, &grid);

        assert!(
            result.is_ok(),
            "Nonlinear inversion pipeline should succeed"
        );
    }

    #[test]
    fn test_all_linear_methods_integration() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let displacement = DisplacementField::zeros(20, 20, 20);

        for method in [
            InversionMethod::TimeOfFlight,
            InversionMethod::PhaseGradient,
            InversionMethod::DirectInversion,
            InversionMethod::VolumetricTimeOfFlight,
            InversionMethod::DirectionalPhaseGradient,
        ] {
            let config = ShearWaveInversionConfig::new(method);
            let inversion = ShearWaveInversion::new(config);
            let result = inversion.reconstruct(&displacement, &grid);

            assert!(
                result.is_ok(),
                "Linear method {:?} should succeed in integration test",
                method
            );
        }
    }

    #[test]
    fn test_all_nonlinear_methods_integration() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

        for method in [
            NonlinearInversionMethod::HarmonicRatio,
            NonlinearInversionMethod::NonlinearLeastSquares,
            NonlinearInversionMethod::BayesianInversion,
        ] {
            let config = NonlinearInversionConfig::new(method);
            let inversion = NonlinearInversion::new(config);
            let result = inversion.reconstruct(&harmonic_field, &grid);

            assert!(
                result.is_ok(),
                "Nonlinear method {:?} should succeed in integration test",
                method
            );
        }
    }

    #[test]
    fn test_config_validation() {
        let valid_config =
            ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight).with_density(1050.0);
        assert!(valid_config.validate().is_ok());

        let invalid_config =
            ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight).with_density(-100.0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_elasticity_statistics() {
        use types::ElasticityMapExt;
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let displacement = DisplacementField::zeros(10, 10, 10);

        let config = ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight);
        let inversion = ShearWaveInversion::new(config);
        let map = inversion.reconstruct(&displacement, &grid).unwrap();

        let (min, max, mean) = map.statistics();
        assert!(min <= mean && mean <= max, "Statistics should be ordered");
    }

    #[test]
    fn test_nonlinear_parameter_statistics() {
        use types::NonlinearParameterMapExt;
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();
        let harmonic_field = HarmonicDisplacementField::new(10, 10, 10, 2, 10);

        let config = NonlinearInversionConfig::new(NonlinearInversionMethod::HarmonicRatio);
        let inversion = NonlinearInversion::new(config);
        let map = inversion.reconstruct(&harmonic_field, &grid).unwrap();

        let (min, max, mean) = map.nonlinearity_statistics();
        assert!(min <= mean && mean <= max, "Statistics should be ordered");

        let (q_min, q_max, q_mean) = map.quality_statistics();
        assert!(
            q_min <= q_mean && q_mean <= q_max,
            "Quality statistics should be ordered"
        );
    }
}
