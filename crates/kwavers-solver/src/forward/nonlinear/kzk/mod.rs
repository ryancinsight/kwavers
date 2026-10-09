//! KZK (Khokhlov-Zabolotskaya-Kuznetsov) Equation Implementation
//!
//! The KZK equation is a parabolic approximation for directional sound beams,
//! widely used in medical ultrasound for modeling focused transducers.
//!
//! # KZK Equation
//!
//! ```text
//! ∂²p/∂z∂τ = (c₀/2)∇⊥²p + (δ/2c₀³)∂³p/∂τ³ + (β/2ρ₀c₀³)∂²(p²)/∂τ²
//! ```
//!
//! - z: axial coordinate (beam propagation direction)
//! - τ = t − z/c₀: retarded time
//! - ∇⊥²: transverse Laplacian (∂²/∂x² + ∂²/∂y²)
//! - δ: diffusivity of sound [m²/s]
//! - β = 1 + B/(2A): nonlinearity coefficient (dimensionless)
//! - ρ₀, c₀: ambient density [kg/m³] and speed (m/s)
//!
//! # Operator Splitting
//!
//! Strang splitting (Strang 1968) achieves second-order accuracy in Δz:
//!
//! ```text
//! U(Δz) ≈ D(Δz/2) · A(Δz/2) · N(Δz) · A(Δz/2) · D(Δz/2)
//! ```
//!
//! where D = diffraction, A = absorption, N = nonlinearity.
//!
//! # References
//!
//! - Zabolotskaya EA, Khokhlov RV (1969). Sov. Phys. Acoust. 15, 35–40.
//! - Kuznetsov VP (1971). Sov. Phys. Acoust. 16, 467–470.
//! - Aanonsen SI et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768. DOI:10.1121/1.390585
//! - Lee Y-S, Hamilton MF (1995). J. Acoust. Soc. Am. 97(2), 906–917. DOI:10.1121/1.412000
//! - Strang G (1968). SIAM J. Numer. Anal. 5(3), 506–517. DOI:10.1137/0705041
//! - Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics. Academic Press.

use kwavers_core::constants::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, REFERENCE_FREQUENCY_HZ,
};
use std::sync::Arc;

pub mod absorption;
pub mod angular_spectrum_2d;
pub mod beam_debug;
pub mod broadband_diffraction;
pub mod complex_parabolic_diffraction;
pub mod constants;
pub mod cylindrical_solver;
pub mod finite_difference_diffraction;
pub mod harmonic_tracking;
pub mod nonlinearity;
pub mod pade11_diffraction;
pub mod pade22_diffraction;
pub mod parabolic_diffraction;
pub mod phase_screen;
pub mod plane_wave_test;
pub mod plugin;
pub mod shock_capturing;
pub mod solver;
pub mod sponge;
pub mod validation;
pub mod wide_angle_diffraction;

pub use cylindrical_solver::{CylindricalKZKConfig, CylindricalKZKSolver};
pub use harmonic_tracking::{HarmonicAnalysis, HarmonicConfig, HarmonicTracker, PredictionModel};
pub use plugin::KzkPlugin;
pub use shock_capturing::{ShockCapture, ShockCapturingConfig, ShockDetectionResult};
pub use solver::KZKSolver;

pub use kwavers_physics::acoustics::wave_propagation::nonlinear::kzk::{
    BroadbandKZKSolverTrait, CylindricalKZKSolverTrait, KZKSolverTrait, WideAngleKZKSolverTrait,
};

/// Diffraction propagator used by the KZK spectral sub-step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DiffractionScheme {
    /// Paraxial (KZK) approximation H = exp(-i k_T² Δz/(2k₀)).
    /// Valid for beam half-angles < ~17°. Standard HITU/HIFU simulation.
    #[default]
    Parabolic,
    /// Padé `[1,1]` wide-angle: H = exp(i φ₁₁), φ₁₁ = k₀Δz·(−s/2)/(1−s/4).
    /// Valid ~17°–35°. Rational correction to paraxial; no exact sqrt needed.
    Pade11,
    /// Padé `[2,2]` wide-angle: H = exp(i φ₂₂), φ₂₂ = k₀Δz·(−s/2+s²/4)/(1−3s/4+s²/16).
    /// Valid ~35°–55°. Higher-order rational correction.
    Pade22,
    /// Wide-angle exact Helmholtz propagator H = exp(i(kz−k₀)Δz),
    /// kz = √(k₀²−k_T²). Valid for all propagating angles; evanescent
    /// modes (k_T > k₀) decay as exp(−|k_T²−k₀²|^½ Δz).
    /// Required for F-number < 1 transducers and steered phased arrays.
    WideAngle,
    /// Exact dispersive diffraction using per-harmonic angular spectrum.
    /// Applies H(k_T, ω_k) = exp(i(kz(ω_k)−k(ω_k))Δz) for each temporal
    /// Fourier mode independently. Eliminates harmonic phase-velocity errors
    /// at the cost of O(nt/2) extra 2-D FFTs per diffraction step.
    /// Required for broadband pulses (bandwidth > ~20% of centre frequency)
    /// or when 3rd-harmonic accuracy matters.
    Broadband,
}

/// Axial propagation direction for KZK marching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PropagationDirection {
    #[default]
    Forward,
    Backward,
}

/// KZK configuration parameters
#[derive(Debug, Clone)]
pub struct KZKConfig {
    /// Grid size in x direction (transverse)
    pub nx: usize,
    /// Grid size in y direction (transverse)
    pub ny: usize,
    /// Grid size in z direction (axial)
    pub nz: usize,
    /// Grid spacing in x and y (m)
    pub dx: f64,
    /// Grid spacing in z (m)
    pub dz: f64,
    /// Axial marching direction.
    pub propagation_direction: PropagationDirection,
    /// Time step (s)
    pub dt: f64,
    /// Number of time steps
    pub nt: usize,
    /// Sound speed (m/s)
    pub c0: f64,
    /// Density (kg/m³)
    pub rho0: f64,
    /// Medium nonlinearity ratio B/A (dimensionless).
    ///
    /// # Theorem (B/A vs β)
    ///
    /// The equation of state is expanded as a Taylor series in density:
    ///   p = ρ₀c₀²(ρ'/ρ₀) + (B/A)/2 · ρ₀c₀²(ρ'/ρ₀)² + O(ρ'³)
    ///
    /// The nonlinearity coefficient used in the KZK equation is
    ///   β = 1 + B/(2A)
    ///
    /// `b_over_a` stores the raw ratio B/A; β is computed internally by
    /// `KzkNonlinearOperator::new` as `1.0 + b_over_a / 2.0`.
    ///
    /// Typical values: water ≈ 5.0, soft tissue ≈ 6.0–7.5.
    ///
    /// Reference: Hamilton MF, Blackstock DT (1998). Nonlinear Acoustics.
    ///   Academic Press. §2.3.2, eq. (2.3.10).
    pub b_over_a: f64,
    /// Attenuation coefficient (Np/m/MHz^y)
    pub alpha0: f64,
    /// Attenuation power law exponent
    pub alpha_power: f64,
    /// Enable diffraction effects
    pub include_diffraction: bool,
    /// Spectral diffraction model used for each axial propagation sub-step.
    pub diffraction_scheme: DiffractionScheme,
    /// Absorbing sponge layer at transverse grid boundaries.
    ///
    /// `None` = periodic (FFT) boundaries. `Some(f)` = raised-cosine taper
    /// over the outer fraction `f ∈ (0, 0.5]` of each transverse dimension.
    /// A value of 0.15–0.25 (15–25%) is typical.
    pub sponge_fraction: Option<f64>,
    /// Enable absorption
    pub include_absorption: bool,
    /// Enable nonlinearity
    pub include_nonlinearity: bool,
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Optionally supply a 3-D map of relative sound-speed perturbations
    /// δ(x,y,z) = c₀/c(x,y,z) − 1 with shape `[nx, ny, nz]`. When `Some`,
    /// a phase-screen correction exp(i k₀ δ(x,y,z_n) Δz) is applied at each
    /// axial step n. `None` = homogeneous medium.
    pub speed_map: Option<Arc<leto::Array3<f64>>>,
}

impl Default for KZKConfig {
    fn default() -> Self {
        Self {
            nx: 128,
            ny: 128,
            nz: 256,
            dx: 0.5e-3, // 0.5 mm
            dz: 0.5e-3, // 0.5 mm
            propagation_direction: PropagationDirection::Forward,
            dt: 10e-9,  // 10 ns
            nt: 1000,
            c0: kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE, // water/tissue
            rho0: DENSITY_WATER_NOMINAL,
            b_over_a: 5.0,                      // B/A for water at 25°C (Beyer 1960)
            alpha0: ACOUSTIC_ABSORPTION_TISSUE, // dB/cm/MHz
            alpha_power: 1.1,
            include_diffraction: true,
            diffraction_scheme: DiffractionScheme::Parabolic,
            sponge_fraction: None,
            include_absorption: true,
            include_nonlinearity: true,
            frequency: REFERENCE_FREQUENCY_HZ, // Default 1 MHz
            speed_map: None,
        }
    }
}

/// Validate KZK configuration
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn validate_config(config: &KZKConfig) -> Result<(), String> {
    // Check grid sizes
    if config.nx < 2 || config.ny < 2 || config.nz < 2 {
        return Err("Grid dimensions must be at least 2".to_owned());
    }

    // Check physical parameters
    if config.c0 <= 0.0 {
        return Err("Sound speed must be positive".to_owned());
    }
    if config.rho0 <= 0.0 {
        return Err("Density must be positive".to_owned());
    }
    if let Some(fraction) = config.sponge_fraction {
        if !(0.0..=0.5).contains(&fraction) || fraction == 0.0 {
            return Err(format!(
                "Sponge fraction must lie in (0, 0.5], got {fraction}"
            ));
        }
    }
    if let Some(speed_map) = &config.speed_map {
        if speed_map.shape() != [config.nx, config.ny, config.nz] {
            return Err(format!(
                "Speed map shape {:?} must match [nx={}, ny={}, nz={}]",
                speed_map.shape(),
                config.nx,
                config.ny,
                config.nz
            ));
        }
    }

    // Check CFL condition for parabolic approximation
    let cfl = config.c0 * config.dt / config.dz;
    if cfl > 0.5 {
        return Err(format!("CFL number {cfl} exceeds 0.5 for stability"));
    }

    // Check diffraction-angle validity for the selected propagator. The Padé
    // variants extend the usable cone relative to the classical paraxial KZK.
    let theta_max = (config.nx as f64 * config.dx / (2.0 * config.nz as f64 * config.dz)).atan();
    let (limit_rad, label) = match config.diffraction_scheme {
        DiffractionScheme::Parabolic => (17.0_f64.to_radians(), "parabolic"),
        DiffractionScheme::Pade11 => (35.0_f64.to_radians(), "Padé [1,1]"),
        DiffractionScheme::Pade22 => (55.0_f64.to_radians(), "Padé [2,2]"),
        DiffractionScheme::WideAngle | DiffractionScheme::Broadband => {
            (f64::INFINITY, "wide-angle")
        }
    };
    if theta_max > limit_rad {
        return Err(format!(
            "Maximum angle {:.1}° exceeds {label} diffraction limit",
            theta_max.to_degrees()
        ));
    }
    if theta_max > 17.0_f64.to_radians()
        && matches!(
            config.diffraction_scheme,
            DiffractionScheme::Pade11
                | DiffractionScheme::Pade22
                | DiffractionScheme::WideAngle
                | DiffractionScheme::Broadband
        )
    {
        tracing::warn!(
            theta_max_deg = theta_max.to_degrees(),
            theta_max_rad = theta_max,
            scheme = ?config.diffraction_scheme,
            "KZK configuration exceeds the paraxial cone; using an extended-angle diffraction propagator"
        );
    }

    Ok(())
}
