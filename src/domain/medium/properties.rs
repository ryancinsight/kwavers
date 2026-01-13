//! Canonical material property data structures for multi-physics simulations
//!
//! # Domain Single Source of Truth (SSOT)
//!
//! This module defines the **canonical data structures** for all material properties
//! used throughout the kwavers framework. These structs complement the trait-based
//! architecture in `domain/medium/traits` by providing concrete, composable data types.
//!
//! ## Architecture Principle: Trait + Struct Duality
//!
//! - **Traits** (`AcousticProperties`, `ElasticProperties`, etc.): Define behavioral contracts
//!   for spatial variation and computation
//! - **Structs** (`AcousticPropertyData`, `ElasticPropertyData`, etc.): Define canonical
//!   data representation for storage, serialization, and composition
//!
//! ## Design Rules
//!
//! 1. **SSOT Enforcement**: No material property structs outside `domain/medium`
//! 2. **Derived Quantities**: Computed on-demand via methods, never stored redundantly
//! 3. **Physics Decoupling**: Each property domain is independent and composable
//! 4. **Validation**: All constructors enforce physical constraints and invariants
//!
//! ## Mathematical Foundations
//!
//! Each property struct is grounded in fundamental physics:
//! - **Acoustic**: Wave equation, impedance matching, absorption models
//! - **Elastic**: Stress-strain relations, Lamé parameters, wave speeds
//! - **Electromagnetic**: Maxwell equations, constitutive relations
//! - **Strength**: Yield criteria, fatigue models, fracture mechanics
//! - **Thermal**: Heat equation, Fourier's law, bio-heat transfer
//!
//! # Examples
//!
//! ```
//! use kwavers::domain::medium::properties::*;
//!
//! // Create acoustic properties for water
//! let water = AcousticPropertyData {
//!     density: 1000.0,
//!     sound_speed: 1500.0,
//!     absorption_coefficient: 0.0022,
//!     absorption_power: 2.0,
//!     nonlinearity: 5.0,
//! };
//! assert_eq!(water.impedance(), 1.5e6);
//!
//! // Create elastic properties from engineering parameters
//! let steel = ElasticPropertyData::from_engineering(7850.0, 200e9, 0.3);
//! assert!((steel.p_wave_speed() - 5960.0).abs() < 100.0);
//!
//! // Compose multi-physics material
//! let tissue = MaterialProperties::builder()
//!     .acoustic(water)
//!     .thermal(ThermalPropertyData {
//!         conductivity: 0.6,
//!         specific_heat: 3600.0,
//!         density: 1050.0,
//!         blood_perfusion: Some(0.5),
//!         blood_specific_heat: Some(3617.0),
//!     })
//!     .build();
//! ```

use std::fmt;

// ================================================================================================
// Acoustic Property Data
// ================================================================================================

/// Canonical acoustic material properties
///
/// # Mathematical Foundation
///
/// Wave equation with absorption and nonlinearity:
/// ```text
/// ∂²p/∂t² = c²∇²p - 2α(∂p/∂t) + (β/ρc²)(∇p)²
/// ```
///
/// Where:
/// - `p`: Acoustic pressure (Pa)
/// - `c`: Sound speed (m/s)
/// - `ρ`: Density (kg/m³)
/// - `α(f)`: Frequency-dependent absorption coefficient (Np/m)
/// - `β`: Nonlinearity coefficient (dimensionless)
///
/// ## Absorption Model
///
/// Power-law frequency dependence:
/// ```text
/// α(f) = α₀ · f^y
/// ```
/// - `α₀`: Absorption coefficient (Np/(MHz^y m))
/// - `f`: Frequency (MHz)
/// - `y`: Power exponent (typical: 1.0-2.0)
///
/// ## Impedance
///
/// Acoustic impedance:
/// ```text
/// Z = ρc  (kg/m²s or Rayl)
/// ```
///
/// ## Invariants
///
/// - `density > 0`
/// - `sound_speed > 0`
/// - `absorption_coefficient ≥ 0`
/// - `0.5 ≤ absorption_power ≤ 3.0` (physical range)
/// - `nonlinearity > 0` (typically 3-10 for biological media)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AcousticPropertyData {
    /// Density ρ (kg/m³)
    ///
    /// Physical range: 1-20000 kg/m³ (air to dense metals)
    pub density: f64,

    /// Sound speed c (m/s)
    ///
    /// Physical range: 300-6000 m/s (air to solids)
    pub sound_speed: f64,

    /// Absorption coefficient α₀ (Np/(MHz^y m))
    ///
    /// Power-law prefactor for frequency-dependent absorption.
    /// For water: ~0.002 Np/(MHz² m) with y=2
    pub absorption_coefficient: f64,

    /// Absorption power exponent y (dimensionless)
    ///
    /// Physical range: 0.5-3.0
    /// - Water: y ≈ 2.0
    /// - Soft tissue: y ≈ 1.0-1.5
    /// - Air: y ≈ 2.0
    pub absorption_power: f64,

    /// Nonlinearity parameter B/A (dimensionless)
    ///
    /// Physical range: 3-10 for biological media
    /// - Water: B/A ≈ 5.0
    /// - Tissue: B/A ≈ 6.0-8.0
    pub nonlinearity: f64,
}

impl AcousticPropertyData {
    /// Construct with validation of physical constraints
    ///
    /// # Errors
    ///
    /// Returns error message if any parameter violates physical bounds
    pub fn new(
        density: f64,
        sound_speed: f64,
        absorption_coefficient: f64,
        absorption_power: f64,
        nonlinearity: f64,
    ) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if sound_speed <= 0.0 {
            return Err(format!("Sound speed must be positive, got {}", sound_speed));
        }
        if absorption_coefficient < 0.0 {
            return Err(format!(
                "Absorption coefficient must be non-negative, got {}",
                absorption_coefficient
            ));
        }
        if !(0.5..=3.0).contains(&absorption_power) {
            return Err(format!(
                "Absorption power must be in range [0.5, 3.0], got {}",
                absorption_power
            ));
        }
        if nonlinearity <= 0.0 {
            return Err(format!(
                "Nonlinearity parameter must be positive, got {}",
                nonlinearity
            ));
        }

        Ok(Self {
            density,
            sound_speed,
            absorption_coefficient,
            absorption_power,
            nonlinearity,
        })
    }

    /// Acoustic impedance Z = ρc (kg/m²s or Rayl)
    ///
    /// Determines reflection/transmission coefficients at interfaces.
    #[inline]
    pub fn impedance(&self) -> f64 {
        self.density * self.sound_speed
    }

    /// Absorption coefficient at frequency f (MHz) → α(f) = α₀ f^y (Np/m)
    ///
    /// # Arguments
    ///
    /// - `freq_mhz`: Frequency in MHz
    ///
    /// # Returns
    ///
    /// Absorption coefficient in Np/m (Nepers per meter)
    #[inline]
    pub fn absorption_at_frequency(&self, freq_mhz: f64) -> f64 {
        self.absorption_coefficient * freq_mhz.powf(self.absorption_power)
    }

    /// Nonlinearity coefficient β = 1 + B/(2A)
    ///
    /// Alternative parameterization used in some nonlinear wave equations.
    #[inline]
    pub fn nonlinearity_coefficient(&self) -> f64 {
        1.0 + self.nonlinearity / 2.0
    }

    /// Water properties at 20°C
    pub fn water() -> Self {
        Self {
            density: 998.0,
            sound_speed: 1481.0,
            absorption_coefficient: 0.002,
            absorption_power: 2.0,
            nonlinearity: 5.0,
        }
    }

    /// Soft tissue properties (generic)
    pub fn soft_tissue() -> Self {
        Self {
            density: 1050.0,
            sound_speed: 1540.0,
            absorption_coefficient: 0.5,
            absorption_power: 1.1,
            nonlinearity: 6.5,
        }
    }

    /// Liver tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1060 kg/m³
    /// - Sound speed: ~1570 m/s
    /// - Attenuation: ~0.5 dB/(MHz·cm) = ~0.58 Np/(MHz·m)
    /// - B/A: ~6.8
    pub fn liver() -> Self {
        Self {
            density: 1060.0,
            sound_speed: 1570.0,
            absorption_coefficient: 0.58,
            absorption_power: 1.1,
            nonlinearity: 6.8,
        }
    }

    /// Brain tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1040 kg/m³
    /// - Sound speed: ~1540 m/s
    /// - Attenuation: ~0.6 dB/(MHz·cm) = ~0.69 Np/(MHz·m)
    /// - B/A: ~6.5
    pub fn brain() -> Self {
        Self {
            density: 1040.0,
            sound_speed: 1540.0,
            absorption_coefficient: 0.69,
            absorption_power: 1.0,
            nonlinearity: 6.5,
        }
    }

    /// Kidney tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1050 kg/m³
    /// - Sound speed: ~1560 m/s
    /// - Attenuation: ~0.7 dB/(MHz·cm) = ~0.81 Np/(MHz·m)
    /// - B/A: ~6.7
    pub fn kidney() -> Self {
        Self {
            density: 1050.0,
            sound_speed: 1560.0,
            absorption_coefficient: 0.81,
            absorption_power: 1.1,
            nonlinearity: 6.7,
        }
    }

    /// Muscle tissue acoustic properties
    ///
    /// Based on clinical measurements:
    /// - Density: ~1070 kg/m³
    /// - Sound speed: ~1580 m/s
    /// - Attenuation: ~1.0 dB/(MHz·cm) = ~1.15 Np/(MHz·m)
    /// - B/A: ~7.4
    pub fn muscle() -> Self {
        Self {
            density: 1070.0,
            sound_speed: 1580.0,
            absorption_coefficient: 1.15,
            absorption_power: 1.0,
            nonlinearity: 7.4,
        }
    }
}

impl fmt::Display for AcousticPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Acoustic(ρ={:.0} kg/m³, c={:.0} m/s, Z={:.2e} Rayl, α₀={:.3}, y={:.2}, B/A={:.1})",
            self.density,
            self.sound_speed,
            self.impedance(),
            self.absorption_coefficient,
            self.absorption_power,
            self.nonlinearity
        )
    }
}

// ================================================================================================
// Elastic Property Data
// ================================================================================================

/// Canonical elastic material properties
///
/// # Mathematical Foundation
///
/// Stress-strain relation (Hooke's law for isotropic linear elasticity):
/// ```text
/// σ = λ tr(ε)I + 2με
/// ```
///
/// Where:
/// - `σ`: Stress tensor (Pa)
/// - `ε`: Strain tensor (dimensionless)
/// - `λ`: Lamé's first parameter (Pa)
/// - `μ`: Lamé's second parameter (shear modulus) (Pa)
/// - `I`: Identity tensor
///
/// ## Wave Speeds
///
/// P-wave (compressional): `c_p = √((λ + 2μ)/ρ)`
/// S-wave (shear): `c_s = √(μ/ρ)`
///
/// ## Engineering Parameters
///
/// Relationships to Young's modulus E and Poisson's ratio ν:
/// ```text
/// λ = Eν / ((1+ν)(1-2ν))
/// μ = E / (2(1+ν))
/// K = λ + 2μ/3  (bulk modulus)
/// ```
///
/// ## Invariants
///
/// - `density > 0`
/// - `lambda ≥ 0`
/// - `mu > 0`
/// - `-1 < ν < 0.5` (Poisson's ratio bounds)
/// - `E > 0` (Young's modulus)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElasticPropertyData {
    /// Density ρ (kg/m³)
    pub density: f64,

    /// Lamé first parameter λ (Pa)
    ///
    /// Related to bulk compressibility. Can be zero for some materials.
    pub lambda: f64,

    /// Lamé second parameter μ (shear modulus) (Pa)
    ///
    /// Resistance to shear deformation. Must be positive.
    pub mu: f64,
}

impl ElasticPropertyData {
    /// Construct from Lamé parameters with validation
    ///
    /// # Errors
    ///
    /// Returns error if parameters violate physical constraints
    pub fn new(density: f64, lambda: f64, mu: f64) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if lambda < 0.0 {
            return Err(format!("Lamé lambda must be non-negative, got {}", lambda));
        }
        if mu <= 0.0 {
            return Err(format!("Shear modulus mu must be positive, got {}", mu));
        }

        // Check Poisson's ratio bounds: -1 < ν < 0.5
        let nu = lambda / (2.0 * (lambda + mu));
        if nu <= -1.0 || nu >= 0.5 {
            return Err(format!("Poisson's ratio {} violates bounds (-1, 0.5)", nu));
        }

        Ok(Self {
            density,
            lambda,
            mu,
        })
    }

    /// Construct from engineering parameters (Young's modulus E, Poisson's ratio ν)
    ///
    /// # Arguments
    ///
    /// - `density`: ρ (kg/m³)
    /// - `youngs_modulus`: E (Pa)
    /// - `poisson_ratio`: ν (dimensionless, must be in (-1, 0.5))
    ///
    /// # Panics
    ///
    /// Panics if parameters are unphysical (use `try_from_engineering` for fallible version)
    pub fn from_engineering(density: f64, youngs_modulus: f64, poisson_ratio: f64) -> Self {
        Self::try_from_engineering(density, youngs_modulus, poisson_ratio)
            .expect("Invalid engineering parameters")
    }

    /// Fallible version of `from_engineering`
    pub fn try_from_engineering(
        density: f64,
        youngs_modulus: f64,
        poisson_ratio: f64,
    ) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if youngs_modulus <= 0.0 {
            return Err(format!(
                "Young's modulus must be positive, got {}",
                youngs_modulus
            ));
        }
        if poisson_ratio <= -1.0 || poisson_ratio >= 0.5 {
            return Err(format!(
                "Poisson's ratio must be in (-1, 0.5), got {}",
                poisson_ratio
            ));
        }

        let lambda =
            youngs_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
        let mu = youngs_modulus / (2.0 * (1.0 + poisson_ratio));

        Ok(Self {
            density,
            lambda,
            mu,
        })
    }

    /// Young's modulus E = μ(3λ + 2μ)/(λ + μ) (Pa)
    #[inline]
    pub fn youngs_modulus(&self) -> f64 {
        self.mu * (3.0 * self.lambda + 2.0 * self.mu) / (self.lambda + self.mu)
    }

    /// Poisson's ratio ν = λ/(2(λ + μ)) (dimensionless)
    #[inline]
    pub fn poisson_ratio(&self) -> f64 {
        self.lambda / (2.0 * (self.lambda + self.mu))
    }

    /// Bulk modulus K = λ + 2μ/3 (Pa)
    #[inline]
    pub fn bulk_modulus(&self) -> f64 {
        self.lambda + 2.0 * self.mu / 3.0
    }

    /// Shear modulus (alias for μ)
    #[inline]
    pub fn shear_modulus(&self) -> f64 {
        self.mu
    }

    /// P-wave (compressional) speed c_p = √((λ + 2μ)/ρ) (m/s)
    #[inline]
    pub fn p_wave_speed(&self) -> f64 {
        ((self.lambda + 2.0 * self.mu) / self.density).sqrt()
    }

    /// S-wave (shear) speed c_s = √(μ/ρ) (m/s)
    #[inline]
    pub fn s_wave_speed(&self) -> f64 {
        (self.mu / self.density).sqrt()
    }

    /// Steel properties (generic)
    pub fn steel() -> Self {
        Self::from_engineering(7850.0, 200e9, 0.3)
    }

    /// Aluminum properties (generic)
    pub fn aluminum() -> Self {
        Self::from_engineering(2700.0, 69e9, 0.33)
    }

    /// Bone properties (cortical bone)
    pub fn bone() -> Self {
        Self::from_engineering(1850.0, 17e9, 0.3)
    }

    /// Construct from wave speeds (inverse problem)
    ///
    /// # Arguments
    ///
    /// - `density`: ρ (kg/m³)
    /// - `p_speed`: P-wave speed c_p (m/s)
    /// - `s_speed`: S-wave speed c_s (m/s)
    ///
    /// # Panics
    ///
    /// Panics if parameters are unphysical (use `try_from_wave_speeds` for fallible version)
    pub fn from_wave_speeds(density: f64, p_speed: f64, s_speed: f64) -> Self {
        Self::try_from_wave_speeds(density, p_speed, s_speed)
            .expect("Invalid wave speed parameters")
    }

    /// Fallible version of `from_wave_speeds`
    ///
    /// Recovers Lamé parameters from measured wave speeds:
    /// ```text
    /// μ = ρ c_s²
    /// λ = ρ c_p² - 2μ
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `density ≤ 0`
    /// - `p_speed ≤ 0` or `s_speed ≤ 0`
    /// - `s_speed ≥ p_speed` (physical constraint: shear waves are slower)
    pub fn try_from_wave_speeds(density: f64, p_speed: f64, s_speed: f64) -> Result<Self, String> {
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if p_speed <= 0.0 {
            return Err(format!("P-wave speed must be positive, got {}", p_speed));
        }
        if s_speed <= 0.0 {
            return Err(format!("S-wave speed must be positive, got {}", s_speed));
        }
        if s_speed >= p_speed {
            return Err(format!(
                "S-wave speed ({}) must be less than P-wave speed ({})",
                s_speed, p_speed
            ));
        }

        // Recover Lamé parameters from wave speeds
        let mu = density * s_speed * s_speed;
        let lambda = density * p_speed * p_speed - 2.0 * mu;

        Self::new(density, lambda, mu)
    }
}

impl fmt::Display for ElasticPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Elastic(ρ={:.0} kg/m³, E={:.2e} Pa, ν={:.3}, c_p={:.0} m/s, c_s={:.0} m/s)",
            self.density,
            self.youngs_modulus(),
            self.poisson_ratio(),
            self.p_wave_speed(),
            self.s_wave_speed()
        )
    }
}

// ================================================================================================
// Electromagnetic Property Data
// ================================================================================================

/// Canonical electromagnetic material properties
///
/// # Mathematical Foundation
///
/// Maxwell's equations in matter:
/// ```text
/// ∇ × E = -∂B/∂t
/// ∇ × H = J + ∂D/∂t
/// ∇ · D = ρ_charge
/// ∇ · B = 0
/// ```
///
/// Constitutive relations:
/// ```text
/// D = ε₀ε_r E  (electric displacement)
/// B = μ₀μ_r H  (magnetic flux density)
/// J = σE       (Ohm's law)
/// ```
///
/// Where:
/// - `ε_r`: Relative permittivity (dimensionless)
/// - `μ_r`: Relative permeability (dimensionless)
/// - `σ`: Electrical conductivity (S/m)
///
/// ## Wave Speed
///
/// Electromagnetic wave speed in medium:
/// ```text
/// c = c₀/√(ε_r μ_r)
/// ```
/// where c₀ = 299,792,458 m/s (speed of light in vacuum)
///
/// ## Impedance
///
/// Intrinsic impedance:
/// ```text
/// Z = Z₀√(μ_r/ε_r)
/// ```
/// where Z₀ = 376.730 Ω (vacuum impedance)
///
/// ## Invariants
///
/// - `permittivity ≥ 1.0` (vacuum is lower bound)
/// - `permeability ≥ 1.0` (most materials, exceptions for metamaterials)
/// - `conductivity ≥ 0.0`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ElectromagneticPropertyData {
    /// Relative permittivity ε_r (dimensionless)
    ///
    /// Physical range:
    /// - Vacuum: 1.0
    /// - Air: ~1.0006
    /// - Water: ~80
    /// - Biological tissue: 10-100
    pub permittivity: f64,

    /// Relative permeability μ_r (dimensionless)
    ///
    /// Physical range:
    /// - Vacuum: 1.0
    /// - Most materials: ~1.0
    /// - Ferromagnetic: 100-10000
    pub permeability: f64,

    /// Electrical conductivity σ (S/m)
    ///
    /// Physical range:
    /// - Insulators: 10^-15 S/m
    /// - Semiconductors: 10^-5 - 10^3 S/m
    /// - Conductors: 10^6 - 10^8 S/m
    pub conductivity: f64,

    /// Dielectric relaxation time τ (s)
    ///
    /// Optional parameter for frequency-dependent permittivity (Debye model).
    /// Typical range: 10^-12 to 10^-6 s
    pub relaxation_time: Option<f64>,
}

impl ElectromagneticPropertyData {
    /// Construct with validation
    pub fn new(
        permittivity: f64,
        permeability: f64,
        conductivity: f64,
        relaxation_time: Option<f64>,
    ) -> Result<Self, String> {
        if permittivity < 1.0 {
            return Err(format!(
                "Relative permittivity must be ≥ 1.0, got {}",
                permittivity
            ));
        }
        if permeability < 0.0 {
            return Err(format!(
                "Relative permeability must be non-negative, got {}",
                permeability
            ));
        }
        if conductivity < 0.0 {
            return Err(format!(
                "Conductivity must be non-negative, got {}",
                conductivity
            ));
        }
        if let Some(tau) = relaxation_time {
            if tau <= 0.0 {
                return Err(format!("Relaxation time must be positive, got {}", tau));
            }
        }

        Ok(Self {
            permittivity,
            permeability,
            conductivity,
            relaxation_time,
        })
    }

    /// Electromagnetic wave speed c = c₀/√(ε_r μ_r) (m/s)
    #[inline]
    pub fn wave_speed(&self) -> f64 {
        const C0: f64 = 299_792_458.0; // m/s
        C0 / (self.permittivity * self.permeability).sqrt()
    }

    /// Intrinsic impedance Z = Z₀√(μ_r/ε_r) (Ω)
    #[inline]
    pub fn impedance(&self) -> f64 {
        const Z0: f64 = 376.730_313_668; // Ω
        Z0 * (self.permeability / self.permittivity).sqrt()
    }

    /// Refractive index n = √(ε_r μ_r)
    #[inline]
    pub fn refractive_index(&self) -> f64 {
        (self.permittivity * self.permeability).sqrt()
    }

    /// Skin depth δ = √(2/(ωμσ)) at angular frequency ω
    ///
    /// Penetration depth for electromagnetic waves in conductive media.
    ///
    /// # Arguments
    ///
    /// - `frequency_hz`: Frequency in Hz
    pub fn skin_depth(&self, frequency_hz: f64) -> f64 {
        if self.conductivity == 0.0 {
            return f64::INFINITY;
        }
        const MU0: f64 = 1.25663706212e-6; // H/m (vacuum permeability)
        let omega = 2.0 * std::f64::consts::PI * frequency_hz;
        let mu = MU0 * self.permeability;
        (2.0 / (omega * mu * self.conductivity)).sqrt()
    }

    /// Vacuum properties
    pub fn vacuum() -> Self {
        Self {
            permittivity: 1.0,
            permeability: 1.0,
            conductivity: 0.0,
            relaxation_time: None,
        }
    }

    /// Water properties (at RF frequencies)
    pub fn water() -> Self {
        Self {
            permittivity: 80.0,
            permeability: 1.0,
            conductivity: 0.005,
            relaxation_time: Some(8.3e-12), // 8.3 ps
        }
    }

    /// Biological tissue properties (generic)
    pub fn tissue() -> Self {
        Self {
            permittivity: 50.0,
            permeability: 1.0,
            conductivity: 0.5,
            relaxation_time: Some(10e-12), // 10 ps
        }
    }
}

impl fmt::Display for ElectromagneticPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EM(ε_r={:.1}, μ_r={:.1}, σ={:.3} S/m, c={:.2e} m/s, n={:.2})",
            self.permittivity,
            self.permeability,
            self.conductivity,
            self.wave_speed(),
            self.refractive_index()
        )
    }
}

// ================================================================================================
// Strength Property Data
// ================================================================================================

/// Canonical mechanical strength properties
///
/// # Mathematical Foundation
///
/// ## Yield Criterion (Von Mises)
///
/// Plastic deformation occurs when equivalent stress reaches yield strength:
/// ```text
/// σ_eq = √(3J₂) ≤ σ_y
/// ```
/// where J₂ is the second invariant of the deviatoric stress tensor.
///
/// ## Fatigue Life (Basquin's Law)
///
/// Number of cycles to failure:
/// ```text
/// N = C (Δσ)^(-b)
/// ```
/// - `N`: Cycles to failure
/// - `Δσ`: Stress amplitude
/// - `b`: Fatigue strength exponent
/// - `C`: Material constant
///
/// ## Invariants
///
/// - `yield_strength > 0`
/// - `ultimate_strength ≥ yield_strength`
/// - `hardness > 0`
/// - `fatigue_exponent > 0` (typical range: 5-15)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrengthPropertyData {
    /// Yield strength σ_y (Pa)
    ///
    /// Stress at which plastic deformation begins.
    pub yield_strength: f64,

    /// Ultimate tensile strength σ_u (Pa)
    ///
    /// Maximum stress before fracture.
    pub ultimate_strength: f64,

    /// Hardness H (Pa)
    ///
    /// Resistance to localized plastic deformation.
    /// Approximation: H ≈ 3σ_y for metals
    pub hardness: f64,

    /// Fatigue strength exponent b (dimensionless)
    ///
    /// Material constant in Basquin's fatigue law.
    /// Typical values:
    /// - Metals: 8-12
    /// - Ceramics: 10-20
    /// - Polymers: 5-10
    pub fatigue_exponent: f64,
}

impl StrengthPropertyData {
    /// Construct with validation
    pub fn new(
        yield_strength: f64,
        ultimate_strength: f64,
        hardness: f64,
        fatigue_exponent: f64,
    ) -> Result<Self, String> {
        if yield_strength <= 0.0 {
            return Err(format!(
                "Yield strength must be positive, got {}",
                yield_strength
            ));
        }
        if ultimate_strength < yield_strength {
            return Err(format!(
                "Ultimate strength ({}) must be ≥ yield strength ({})",
                ultimate_strength, yield_strength
            ));
        }
        if hardness <= 0.0 {
            return Err(format!("Hardness must be positive, got {}", hardness));
        }
        if fatigue_exponent <= 0.0 {
            return Err(format!(
                "Fatigue exponent must be positive, got {}",
                fatigue_exponent
            ));
        }

        Ok(Self {
            yield_strength,
            ultimate_strength,
            hardness,
            fatigue_exponent,
        })
    }

    /// Estimate hardness from yield strength (H ≈ 3σ_y for metals)
    pub fn estimate_hardness(yield_strength: f64) -> f64 {
        3.0 * yield_strength
    }

    /// Steel properties (mild steel)
    pub fn steel() -> Self {
        Self {
            yield_strength: 250e6,
            ultimate_strength: 400e6,
            hardness: 750e6,
            fatigue_exponent: 10.0,
        }
    }

    /// Bone properties (cortical bone)
    pub fn bone() -> Self {
        Self {
            yield_strength: 130e6,
            ultimate_strength: 150e6,
            hardness: 390e6,
            fatigue_exponent: 12.0,
        }
    }
}

impl fmt::Display for StrengthPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Strength(σ_y={:.0} MPa, σ_u={:.0} MPa, H={:.0} MPa, b={:.1})",
            self.yield_strength / 1e6,
            self.ultimate_strength / 1e6,
            self.hardness / 1e6,
            self.fatigue_exponent
        )
    }
}

// ================================================================================================
// Thermal Property Data
// ================================================================================================

/// Canonical thermal material properties
///
/// # Mathematical Foundation
///
/// ## Heat Equation (Fourier's Law)
///
/// ```text
/// ρc ∂T/∂t = ∇·(k∇T) + Q
/// ```
///
/// Where:
/// - `T`: Temperature (K)
/// - `k`: Thermal conductivity (W/m/K)
/// - `ρ`: Density (kg/m³)
/// - `c`: Specific heat capacity (J/kg/K)
/// - `Q`: Heat source (W/m³)
///
/// ## Bio-Heat Equation (Pennes)
///
/// For biological tissue:
/// ```text
/// ρc ∂T/∂t = ∇·(k∇T) + w_b c_b (T_b - T) + Q_met + Q_ext
/// ```
///
/// Additional terms:
/// - `w_b`: Blood perfusion rate (kg/m³/s)
/// - `c_b`: Blood specific heat (J/kg/K)
/// - `T_b`: Arterial blood temperature (K)
///
/// ## Thermal Diffusivity
///
/// ```text
/// α = k/(ρc)  (m²/s)
/// ```
///
/// ## Invariants
///
/// - `conductivity > 0`
/// - `specific_heat > 0`
/// - `density > 0`
/// - `blood_perfusion ≥ 0` (if present)
/// - `blood_specific_heat > 0` (if present)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThermalPropertyData {
    /// Thermal conductivity k (W/m/K)
    ///
    /// Physical range:
    /// - Insulators: 0.01-1 W/m/K
    /// - Water: ~0.6 W/m/K
    /// - Tissue: 0.4-0.6 W/m/K
    /// - Metals: 10-400 W/m/K
    pub conductivity: f64,

    /// Specific heat capacity c (J/kg/K)
    ///
    /// Physical range:
    /// - Metals: 100-1000 J/kg/K
    /// - Water: ~4180 J/kg/K
    /// - Tissue: 3000-4000 J/kg/K
    pub specific_heat: f64,

    /// Density ρ (kg/m³)
    pub density: f64,

    /// Blood perfusion rate w_b (kg/m³/s)
    ///
    /// Optional: Only for biological tissue in bio-heat equation.
    /// Typical range: 0.5-10 kg/m³/s
    pub blood_perfusion: Option<f64>,

    /// Blood specific heat c_b (J/kg/K)
    ///
    /// Optional: Only for biological tissue in bio-heat equation.
    /// Typical value: ~3617 J/kg/K
    pub blood_specific_heat: Option<f64>,
}

impl ThermalPropertyData {
    /// Construct with validation
    pub fn new(
        conductivity: f64,
        specific_heat: f64,
        density: f64,
        blood_perfusion: Option<f64>,
        blood_specific_heat: Option<f64>,
    ) -> Result<Self, String> {
        if conductivity <= 0.0 {
            return Err(format!(
                "Thermal conductivity must be positive, got {}",
                conductivity
            ));
        }
        if specific_heat <= 0.0 {
            return Err(format!(
                "Specific heat must be positive, got {}",
                specific_heat
            ));
        }
        if density <= 0.0 {
            return Err(format!("Density must be positive, got {}", density));
        }
        if let Some(w_b) = blood_perfusion {
            if w_b < 0.0 {
                return Err(format!("Blood perfusion must be non-negative, got {}", w_b));
            }
        }
        if let Some(c_b) = blood_specific_heat {
            if c_b <= 0.0 {
                return Err(format!("Blood specific heat must be positive, got {}", c_b));
            }
        }

        Ok(Self {
            conductivity,
            specific_heat,
            density,
            blood_perfusion,
            blood_specific_heat,
        })
    }

    /// Thermal diffusivity α = k/(ρc) (m²/s)
    #[inline]
    pub fn thermal_diffusivity(&self) -> f64 {
        self.conductivity / (self.density * self.specific_heat)
    }

    /// Check if bio-heat parameters are present
    #[inline]
    pub fn has_bioheat_parameters(&self) -> bool {
        self.blood_perfusion.is_some() && self.blood_specific_heat.is_some()
    }

    /// Water properties (at 20°C)
    pub fn water() -> Self {
        Self {
            conductivity: 0.598,
            specific_heat: 4182.0,
            density: 998.0,
            blood_perfusion: None,
            blood_specific_heat: None,
        }
    }

    /// Soft tissue properties (generic)
    pub fn soft_tissue() -> Self {
        Self {
            conductivity: 0.5,
            specific_heat: 3600.0,
            density: 1050.0,
            blood_perfusion: Some(0.5),
            blood_specific_heat: Some(3617.0),
        }
    }

    /// Bone properties
    pub fn bone() -> Self {
        Self {
            conductivity: 0.32,
            specific_heat: 1300.0,
            density: 1850.0,
            blood_perfusion: None,
            blood_specific_heat: None,
        }
    }
}

impl fmt::Display for ThermalPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Thermal(k={:.2} W/m/K, c={:.0} J/kg/K, ρ={:.0} kg/m³, α={:.2e} m²/s",
            self.conductivity,
            self.specific_heat,
            self.density,
            self.thermal_diffusivity()
        )?;
        if self.has_bioheat_parameters() {
            write!(f, ", bio-heat enabled")?;
        }
        write!(f, ")")
    }
}

// ================================================================================================
// Optical Property Data
// ================================================================================================

/// Canonical optical material properties for light propagation and scattering
///
/// # Mathematical Foundation
///
/// ## Radiative Transfer Equation (RTE)
///
/// Light propagation in scattering media:
/// ```text
/// dI/ds = -μ_t I + μ_s ∫ p(θ) I(s') dΩ'
/// ```
///
/// Where:
/// - `I`: Radiance (W/m²/sr)
/// - `s`: Path length (m)
/// - `μ_t = μ_a + μ_s`: Total attenuation coefficient (m⁻¹)
/// - `μ_a`: Absorption coefficient (m⁻¹)
/// - `μ_s`: Scattering coefficient (m⁻¹)
/// - `p(θ)`: Phase function (angular scattering probability)
///
/// ## Henyey-Greenstein Phase Function
///
/// Anisotropic scattering model:
/// ```text
/// p(θ) = (1 - g²) / (4π (1 + g² - 2g cos θ)^(3/2))
/// ```
/// - `g`: Anisotropy factor (⟨cos θ⟩)
/// - `g = 0`: Isotropic scattering
/// - `g > 0`: Forward scattering (typical for biological tissue)
/// - `g < 0`: Backward scattering
///
/// ## Beer-Lambert Law
///
/// Attenuation in non-scattering media:
/// ```text
/// I(z) = I₀ exp(-μ_a z)
/// ```
///
/// ## Refractive Index and Snell's Law
///
/// ```text
/// n₁ sin θ₁ = n₂ sin θ₂
/// ```
///
/// ## Invariants
///
/// - `absorption_coefficient ≥ 0` (m⁻¹)
/// - `scattering_coefficient ≥ 0` (m⁻¹)
/// - `-1 ≤ anisotropy ≤ 1` (dimensionless)
/// - `refractive_index ≥ 1.0` (vacuum is lower bound)
///
/// # Physical Context
///
/// Optical properties are critical for:
/// - **Photoacoustic imaging**: Light absorption generates acoustic waves
/// - **Optical coherence tomography (OCT)**: Scattering provides contrast
/// - **Diffuse optical tomography (DOT)**: Light propagation models tissue structure
/// - **Laser therapy**: Energy deposition depends on absorption/scattering balance
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalPropertyData {
    /// Absorption coefficient μ_a (m⁻¹)
    ///
    /// Quantifies light energy converted to heat per unit path length.
    ///
    /// Physical range:
    /// - Water (visible): 0.001-10 m⁻¹
    /// - Blood (hemoglobin-dependent): 1-1000 m⁻¹
    /// - Soft tissue: 0.1-100 m⁻¹
    /// - Tumor (enhanced vascularity): 5-500 m⁻¹
    ///
    /// Wavelength-dependent (strongly peaked at hemoglobin absorption bands).
    pub absorption_coefficient: f64,

    /// Scattering coefficient μ_s (m⁻¹)
    ///
    /// Quantifies directional change events per unit path length.
    ///
    /// Physical range:
    /// - Water (visible): ~0.01 m⁻¹
    /// - Soft tissue: 50-500 m⁻¹
    /// - Blood: 100-300 m⁻¹
    /// - Bone: 200-1000 m⁻¹
    ///
    /// Typically exceeds absorption in biological tissue (μ_s ≫ μ_a).
    pub scattering_coefficient: f64,

    /// Anisotropy factor g = ⟨cos θ⟩ (dimensionless)
    ///
    /// Average cosine of scattering angle:
    /// - `g = 0`: Isotropic scattering (equal probability in all directions)
    /// - `g = 1`: Complete forward scattering (no directional change)
    /// - `g = -1`: Complete backward scattering
    ///
    /// Physical range for biological tissue: 0.7-0.99 (highly forward-scattering)
    ///
    /// Typical values:
    /// - Soft tissue: 0.8-0.95
    /// - Blood: 0.95-0.99
    /// - Water: 0.0 (isotropic, minimal scattering)
    pub anisotropy: f64,

    /// Refractive index n (dimensionless)
    ///
    /// Ratio of light speed in vacuum to light speed in medium: n = c₀/c
    ///
    /// Physical range:
    /// - Vacuum: 1.0 (by definition)
    /// - Air: ~1.0003
    /// - Water: ~1.33
    /// - Biological tissue: 1.35-1.55
    /// - Glass: 1.5-1.9
    ///
    /// Determines reflection/refraction at interfaces via Fresnel equations.
    pub refractive_index: f64,
}

impl OpticalPropertyData {
    /// Construct with validation of physical constraints
    ///
    /// # Errors
    ///
    /// Returns error message if any parameter violates physical bounds
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::domain::medium::properties::OpticalPropertyData;
    ///
    /// let tissue = OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).unwrap();
    /// assert_eq!(tissue.total_attenuation(), 110.0);
    /// ```
    pub fn new(
        absorption_coefficient: f64,
        scattering_coefficient: f64,
        anisotropy: f64,
        refractive_index: f64,
    ) -> Result<Self, String> {
        if absorption_coefficient < 0.0 {
            return Err(format!(
                "Absorption coefficient must be non-negative, got {}",
                absorption_coefficient
            ));
        }
        if scattering_coefficient < 0.0 {
            return Err(format!(
                "Scattering coefficient must be non-negative, got {}",
                scattering_coefficient
            ));
        }
        if !(-1.0..=1.0).contains(&anisotropy) {
            return Err(format!(
                "Anisotropy factor must be in range [-1, 1], got {}",
                anisotropy
            ));
        }
        if refractive_index < 1.0 {
            return Err(format!(
                "Refractive index must be ≥ 1.0 (vacuum limit), got {}",
                refractive_index
            ));
        }

        Ok(Self {
            absorption_coefficient,
            scattering_coefficient,
            anisotropy,
            refractive_index,
        })
    }

    /// Total attenuation coefficient μ_t = μ_a + μ_s (m⁻¹)
    ///
    /// Quantifies total extinction (absorption + scattering) per unit path length.
    #[inline]
    pub fn total_attenuation(&self) -> f64 {
        self.absorption_coefficient + self.scattering_coefficient
    }

    /// Reduced scattering coefficient μ_s' = μ_s (1 - g) (m⁻¹)
    ///
    /// Effective scattering coefficient accounting for anisotropy.
    /// Used in diffusion approximation for multiple scattering regimes.
    #[inline]
    pub fn reduced_scattering(&self) -> f64 {
        self.scattering_coefficient * (1.0 - self.anisotropy)
    }

    /// Optical penetration depth δ = 1/μ_eff (m)
    ///
    /// Characteristic depth for exponential decay in diffusion regime.
    ///
    /// Effective attenuation coefficient:
    /// ```text
    /// μ_eff = √(3 μ_a (μ_a + μ_s'))
    /// ```
    #[inline]
    pub fn penetration_depth(&self) -> f64 {
        let mu_s_prime = self.reduced_scattering();
        let mu_eff =
            (3.0 * self.absorption_coefficient * (self.absorption_coefficient + mu_s_prime)).sqrt();
        if mu_eff > 0.0 {
            1.0 / mu_eff
        } else {
            f64::INFINITY
        }
    }

    /// Mean free path l_mfp = 1/μ_t (m)
    ///
    /// Average distance traveled before absorption or scattering event.
    #[inline]
    pub fn mean_free_path(&self) -> f64 {
        let mu_t = self.total_attenuation();
        if mu_t > 0.0 {
            1.0 / mu_t
        } else {
            f64::INFINITY
        }
    }

    /// Transport mean free path l_tr = 1/(μ_a + μ_s') (m)
    ///
    /// Average distance for directional randomization (diffusion length scale).
    #[inline]
    pub fn transport_mean_free_path(&self) -> f64 {
        let mu_tr = self.absorption_coefficient + self.reduced_scattering();
        if mu_tr > 0.0 {
            1.0 / mu_tr
        } else {
            f64::INFINITY
        }
    }

    /// Albedo α = μ_s / μ_t (dimensionless)
    ///
    /// Probability of scattering vs absorption per extinction event.
    /// - α ≈ 0: Strongly absorbing (most photons absorbed)
    /// - α ≈ 1: Strongly scattering (most photons scattered)
    #[inline]
    pub fn albedo(&self) -> f64 {
        let mu_t = self.total_attenuation();
        if mu_t > 0.0 {
            self.scattering_coefficient / mu_t
        } else {
            0.0
        }
    }

    /// Fresnel reflectance at normal incidence R₀
    ///
    /// Reflection coefficient for light incident perpendicular to interface
    /// from vacuum/air (n₁ = 1.0) into this medium (n₂).
    ///
    /// ```text
    /// R₀ = ((n₁ - n₂) / (n₁ + n₂))²
    /// ```
    #[inline]
    pub fn fresnel_reflectance_normal(&self) -> f64 {
        let n1 = 1.0; // vacuum/air
        let n2 = self.refractive_index;
        let r = (n1 - n2) / (n1 + n2);
        r * r
    }

    /// Water optical properties (visible spectrum, ~550 nm)
    ///
    /// Reference: Hale & Querry (1973), Pope & Fry (1997)
    pub fn water() -> Self {
        Self {
            absorption_coefficient: 0.01,  // Very low absorption in visible
            scattering_coefficient: 0.001, // Minimal scattering
            anisotropy: 0.0,               // Isotropic
            refractive_index: 1.33,        // Standard value at visible wavelengths
        }
    }

    /// Soft tissue optical properties (generic, ~650 nm)
    ///
    /// Typical values for homogeneous soft tissue models.
    pub fn soft_tissue() -> Self {
        Self {
            absorption_coefficient: 0.5,   // Low absorption
            scattering_coefficient: 100.0, // High scattering (μ_s ≫ μ_a)
            anisotropy: 0.9,               // Highly forward-scattering
            refractive_index: 1.4,         // Typical for biological tissue
        }
    }

    /// Blood optical properties (oxygenated, ~650 nm)
    ///
    /// Wavelength-dependent; this represents visible red region where
    /// oxy-hemoglobin absorption is moderate.
    ///
    /// Note: For photoacoustic imaging, use wavelength-specific constructors
    /// or external spectral databases for accurate hemoglobin modeling.
    pub fn blood_oxygenated() -> Self {
        Self {
            absorption_coefficient: 50.0,  // Moderate absorption in red
            scattering_coefficient: 200.0, // High scattering from RBCs
            anisotropy: 0.95,              // Very forward-scattering
            refractive_index: 1.4,         // Similar to plasma
        }
    }

    /// Blood optical properties (deoxygenated, ~650 nm)
    ///
    /// Deoxy-hemoglobin has higher absorption in visible red than oxy-hemoglobin.
    pub fn blood_deoxygenated() -> Self {
        Self {
            absorption_coefficient: 80.0, // Higher absorption than oxygenated
            scattering_coefficient: 200.0,
            anisotropy: 0.95,
            refractive_index: 1.4,
        }
    }

    /// Tumor tissue optical properties (hypervascular, ~650 nm)
    ///
    /// Enhanced absorption due to increased blood content and metabolic activity.
    pub fn tumor() -> Self {
        Self {
            absorption_coefficient: 10.0, // Higher than normal tissue
            scattering_coefficient: 120.0,
            anisotropy: 0.85,
            refractive_index: 1.4,
        }
    }

    /// Brain tissue optical properties (gray matter, ~650 nm)
    pub fn brain_gray_matter() -> Self {
        Self {
            absorption_coefficient: 0.8,
            scattering_coefficient: 150.0,
            anisotropy: 0.9,
            refractive_index: 1.38,
        }
    }

    /// Brain tissue optical properties (white matter, ~650 nm)
    ///
    /// White matter has higher scattering due to myelinated fiber tracts.
    pub fn brain_white_matter() -> Self {
        Self {
            absorption_coefficient: 1.0,
            scattering_coefficient: 250.0, // Higher than gray matter
            anisotropy: 0.92,
            refractive_index: 1.38,
        }
    }

    /// Liver tissue optical properties (~650 nm)
    pub fn liver() -> Self {
        Self {
            absorption_coefficient: 2.0,
            scattering_coefficient: 120.0,
            anisotropy: 0.88,
            refractive_index: 1.39,
        }
    }

    /// Muscle tissue optical properties (~650 nm)
    pub fn muscle() -> Self {
        Self {
            absorption_coefficient: 0.8,
            scattering_coefficient: 100.0,
            anisotropy: 0.85,
            refractive_index: 1.37,
        }
    }

    /// Skin (epidermis) optical properties (~650 nm)
    pub fn skin_epidermis() -> Self {
        Self {
            absorption_coefficient: 5.0, // Higher due to melanin
            scattering_coefficient: 300.0,
            anisotropy: 0.8,
            refractive_index: 1.4,
        }
    }

    /// Skin (dermis) optical properties (~650 nm)
    pub fn skin_dermis() -> Self {
        Self {
            absorption_coefficient: 1.0,
            scattering_coefficient: 200.0,
            anisotropy: 0.85,
            refractive_index: 1.4,
        }
    }

    /// Bone (cortical) optical properties (~650 nm)
    ///
    /// Bone is highly scattering and difficult to penetrate optically.
    pub fn bone_cortical() -> Self {
        Self {
            absorption_coefficient: 5.0,
            scattering_coefficient: 500.0, // Very high scattering
            anisotropy: 0.9,
            refractive_index: 1.55,
        }
    }

    /// Fat tissue optical properties (~650 nm)
    pub fn fat() -> Self {
        Self {
            absorption_coefficient: 0.3,
            scattering_coefficient: 100.0,
            anisotropy: 0.9,
            refractive_index: 1.46,
        }
    }
}

impl fmt::Display for OpticalPropertyData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Optical(μ_a={:.2} m⁻¹, μ_s={:.1} m⁻¹, μ_s'={:.1} m⁻¹, g={:.3}, n={:.3}, δ={:.1} mm, α={:.3})",
            self.absorption_coefficient,
            self.scattering_coefficient,
            self.reduced_scattering(),
            self.anisotropy,
            self.refractive_index,
            self.penetration_depth() * 1000.0, // Convert to mm
            self.albedo()
        )
    }
}

// ================================================================================================
// Material Properties Composite
// ================================================================================================

/// Composite material properties for multi-physics simulations
///
/// # Domain Rule: Single Source of Truth
///
/// This is the **canonical composition point** for all material properties in kwavers.
/// Physics modules should extract only the properties they need from this composite.
///
/// # Design Pattern: Builder + Optional Properties
///
/// - **Acoustic**: Always present (base requirement for wave propagation)
/// - **Elastic**: Optional (only for solid media)
/// - **Electromagnetic**: Optional (only for EM simulations)
/// - **Strength**: Optional (only for damage/fracture mechanics)
/// - **Thermal**: Optional (only for thermal effects)
///
/// # Examples
///
/// ```
/// use kwavers::domain::medium::properties::*;
///
/// // Simple acoustic material (water)
/// let water = MaterialProperties::acoustic_only(AcousticPropertyData::water());
///
/// // Elastic solid (steel)
/// let steel = MaterialProperties::builder()
///     .acoustic(AcousticPropertyData {
///         density: 7850.0,
///         sound_speed: 5960.0,
///         absorption_coefficient: 0.001,
///         absorption_power: 1.0,
///         nonlinearity: 4.0,
///     })
///     .elastic(ElasticPropertyData::steel())
///     .strength(StrengthPropertyData::steel())
///     .build();
///
/// // Biological tissue (multi-physics)
/// let tissue = MaterialProperties::builder()
///     .acoustic(AcousticPropertyData::soft_tissue())
///     .thermal(ThermalPropertyData::soft_tissue())
///     .electromagnetic(ElectromagneticPropertyData::tissue())
///     .build();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct MaterialProperties {
    /// Acoustic properties (always present)
    pub acoustic: AcousticPropertyData,

    /// Elastic properties (optional, for solids)
    pub elastic: Option<ElasticPropertyData>,

    /// Electromagnetic properties (optional, for EM waves)
    pub electromagnetic: Option<ElectromagneticPropertyData>,

    /// Optical properties (optional, for light propagation)
    pub optical: Option<OpticalPropertyData>,

    /// Strength properties (optional, for damage/fracture)
    pub strength: Option<StrengthPropertyData>,

    /// Thermal properties (optional, for thermal effects)
    pub thermal: Option<ThermalPropertyData>,
}

impl MaterialProperties {
    /// Create acoustic-only material (e.g., water, air)
    pub fn acoustic_only(acoustic: AcousticPropertyData) -> Self {
        Self {
            acoustic,
            elastic: None,
            electromagnetic: None,
            optical: None,
            strength: None,
            thermal: None,
        }
    }

    /// Create elastic material with acoustic coupling
    pub fn elastic(acoustic: AcousticPropertyData, elastic: ElasticPropertyData) -> Self {
        Self {
            acoustic,
            elastic: Some(elastic),
            electromagnetic: None,
            optical: None,
            strength: None,
            thermal: None,
        }
    }

    /// Create builder for multi-physics composition
    pub fn builder() -> MaterialPropertiesBuilder {
        MaterialPropertiesBuilder::default()
    }

    /// Water at 20°C (acoustic + thermal)
    pub fn water() -> Self {
        Self::builder()
            .acoustic(AcousticPropertyData::water())
            .thermal(ThermalPropertyData::water())
            .build()
    }

    /// Soft tissue (acoustic + thermal + EM)
    pub fn soft_tissue() -> Self {
        Self::builder()
            .acoustic(AcousticPropertyData::soft_tissue())
            .thermal(ThermalPropertyData::soft_tissue())
            .electromagnetic(ElectromagneticPropertyData::tissue())
            .build()
    }

    /// Cortical bone (acoustic + elastic + thermal + strength)
    pub fn bone() -> Self {
        Self::builder()
            .acoustic(AcousticPropertyData {
                density: 1850.0,
                sound_speed: 3500.0,
                absorption_coefficient: 0.8,
                absorption_power: 1.2,
                nonlinearity: 5.5,
            })
            .elastic(ElasticPropertyData::bone())
            .thermal(ThermalPropertyData::bone())
            .strength(StrengthPropertyData::bone())
            .build()
    }

    /// Steel (acoustic + elastic + strength)
    pub fn steel() -> Self {
        Self::builder()
            .acoustic(AcousticPropertyData {
                density: 7850.0,
                sound_speed: 5960.0,
                absorption_coefficient: 0.001,
                absorption_power: 1.0,
                nonlinearity: 4.0,
            })
            .elastic(ElasticPropertyData::steel())
            .strength(StrengthPropertyData::steel())
            .build()
    }
}

impl fmt::Display for MaterialProperties {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MaterialProperties {{")?;
        writeln!(f, "  {}", self.acoustic)?;
        if let Some(ref elastic) = self.elastic {
            writeln!(f, "  {}", elastic)?;
        }
        if let Some(ref em) = self.electromagnetic {
            writeln!(f, "  {}", em)?;
        }
        if let Some(ref optical) = self.optical {
            writeln!(f, "  {}", optical)?;
        }
        if let Some(ref strength) = self.strength {
            writeln!(f, "  {}", strength)?;
        }
        if let Some(ref thermal) = self.thermal {
            writeln!(f, "  {}", thermal)?;
        }
        write!(f, "}}")
    }
}

// ================================================================================================
// Material Properties Builder
// ================================================================================================

/// Builder for constructing `MaterialProperties` with optional components
#[derive(Debug, Default)]
pub struct MaterialPropertiesBuilder {
    acoustic: Option<AcousticPropertyData>,
    elastic: Option<ElasticPropertyData>,
    electromagnetic: Option<ElectromagneticPropertyData>,
    optical: Option<OpticalPropertyData>,
    strength: Option<StrengthPropertyData>,
    thermal: Option<ThermalPropertyData>,
}

impl MaterialPropertiesBuilder {
    /// Set acoustic properties (required)
    pub fn acoustic(mut self, acoustic: AcousticPropertyData) -> Self {
        self.acoustic = Some(acoustic);
        self
    }

    /// Set elastic properties (optional)
    pub fn elastic(mut self, elastic: ElasticPropertyData) -> Self {
        self.elastic = Some(elastic);
        self
    }

    /// Set electromagnetic properties (optional)
    pub fn electromagnetic(mut self, em: ElectromagneticPropertyData) -> Self {
        self.electromagnetic = Some(em);
        self
    }

    /// Set optical properties (optional)
    pub fn optical(mut self, optical: OpticalPropertyData) -> Self {
        self.optical = Some(optical);
        self
    }

    /// Set strength properties (optional)
    pub fn strength(mut self, strength: StrengthPropertyData) -> Self {
        self.strength = Some(strength);
        self
    }

    /// Set thermal properties (optional)
    pub fn thermal(mut self, thermal: ThermalPropertyData) -> Self {
        self.thermal = Some(thermal);
        self
    }

    /// Build the composite material properties
    ///
    /// # Panics
    ///
    /// Panics if acoustic properties are not set
    pub fn build(self) -> MaterialProperties {
        MaterialProperties {
            acoustic: self.acoustic.expect("Acoustic properties are required"),
            elastic: self.elastic,
            electromagnetic: self.electromagnetic,
            optical: self.optical,
            strength: self.strength,
            thermal: self.thermal,
        }
    }
}

// ================================================================================================
// Unit Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Acoustic Property Tests
    // ================================================================================================

    #[test]
    fn test_acoustic_impedance() {
        let water = AcousticPropertyData::water();
        let expected_impedance = 998.0 * 1481.0; // ρc
        assert!((water.impedance() - expected_impedance).abs() < 1.0);
    }

    #[test]
    fn test_acoustic_absorption() {
        let props = AcousticPropertyData {
            density: 1000.0,
            sound_speed: 1500.0,
            absorption_coefficient: 0.5,
            absorption_power: 1.1,
            nonlinearity: 5.0,
        };

        let alpha_1mhz = props.absorption_at_frequency(1.0);
        let alpha_2mhz = props.absorption_at_frequency(2.0);

        // α(2f) = α₀(2f)^y = 2^y α(f)
        let expected_ratio = 2.0_f64.powf(1.1);
        assert!((alpha_2mhz / alpha_1mhz - expected_ratio).abs() < 1e-10);
    }

    #[test]
    fn test_acoustic_validation() {
        // Negative density should fail
        assert!(AcousticPropertyData::new(-1000.0, 1500.0, 0.5, 1.1, 5.0).is_err());

        // Invalid absorption power should fail
        assert!(AcousticPropertyData::new(1000.0, 1500.0, 0.5, 5.0, 5.0).is_err());

        // Valid parameters should succeed
        assert!(AcousticPropertyData::new(1000.0, 1500.0, 0.5, 1.1, 5.0).is_ok());
    }

    // Elastic Property Tests
    // ================================================================================================

    #[test]
    fn test_elastic_engineering_conversion() {
        let density = 7850.0; // kg/m³
        let youngs = 200e9; // Pa
        let poisson = 0.3;

        let elastic = ElasticPropertyData::from_engineering(density, youngs, poisson);

        // Verify round-trip conversion
        assert!((elastic.youngs_modulus() - youngs).abs() / youngs < 1e-10);
        assert!((elastic.poisson_ratio() - poisson).abs() < 1e-10);
    }

    #[test]
    fn test_elastic_wave_speeds() {
        let steel = ElasticPropertyData::steel();

        // Steel P-wave speed should be ~5960 m/s (typical range: 5800-6100 m/s)
        let cp = steel.p_wave_speed();
        assert!(
            cp > 5000.0 && cp < 7000.0,
            "P-wave speed {} out of expected range",
            cp
        );

        // S-wave speed should be less than P-wave speed
        let cs = steel.s_wave_speed();
        assert!(cs < cp);

        // S-wave speed should be ~3200 m/s (typical range: 3000-3400 m/s)
        assert!(
            cs > 2500.0 && cs < 4000.0,
            "S-wave speed {} out of expected range",
            cs
        );
    }

    #[test]
    fn test_elastic_poisson_bounds() {
        let density = 1000.0;

        // ν = 0.5 (incompressible) should fail
        assert!(ElasticPropertyData::try_from_engineering(density, 1e9, 0.5).is_err());

        // ν = -1 should fail
        assert!(ElasticPropertyData::try_from_engineering(density, 1e9, -1.0).is_err());

        // Valid ν = 0.3 should succeed
        assert!(ElasticPropertyData::try_from_engineering(density, 1e9, 0.3).is_ok());
    }

    #[test]
    fn test_elastic_moduli_relations() {
        let elastic = ElasticPropertyData::from_engineering(7850.0, 200e9, 0.3);

        let e = elastic.youngs_modulus();
        let nu = elastic.poisson_ratio();
        let k = elastic.bulk_modulus();
        let mu = elastic.shear_modulus();

        // K = E / (3(1 - 2ν))
        let k_expected = e / (3.0 * (1.0 - 2.0 * nu));
        assert!((k - k_expected).abs() / k < 1e-10);

        // μ = E / (2(1 + ν))
        let mu_expected = e / (2.0 * (1.0 + nu));
        assert!((mu - mu_expected).abs() / mu < 1e-10);
    }

    #[test]
    fn test_elastic_from_wave_speeds() {
        let density = 7850.0; // Steel density
        let cp = 5960.0; // P-wave speed
        let cs = 3220.0; // S-wave speed

        let elastic = ElasticPropertyData::from_wave_speeds(density, cp, cs);

        // Verify wave speeds are recovered
        assert!((elastic.p_wave_speed() - cp).abs() < 1e-6);
        assert!((elastic.s_wave_speed() - cs).abs() < 1e-6);

        // Verify density
        assert_eq!(elastic.density, density);

        // Verify Lamé parameters are positive
        assert!(elastic.lambda > 0.0);
        assert!(elastic.mu > 0.0);
    }

    #[test]
    fn test_elastic_from_wave_speeds_validation() {
        let density = 1000.0;

        // S-wave speed >= P-wave speed should fail (physical constraint)
        assert!(ElasticPropertyData::try_from_wave_speeds(density, 1500.0, 1600.0).is_err());

        // Negative density should fail
        assert!(ElasticPropertyData::try_from_wave_speeds(-1000.0, 1500.0, 1000.0).is_err());

        // Negative wave speeds should fail
        assert!(ElasticPropertyData::try_from_wave_speeds(density, -1500.0, 1000.0).is_err());
        assert!(ElasticPropertyData::try_from_wave_speeds(density, 1500.0, -1000.0).is_err());

        // Valid parameters should succeed
        assert!(ElasticPropertyData::try_from_wave_speeds(density, 1500.0, 1000.0).is_ok());
    }

    #[test]
    fn test_elastic_wave_speed_round_trip() {
        // Start with engineering parameters
        let original = ElasticPropertyData::from_engineering(2700.0, 69e9, 0.33);

        // Extract wave speeds
        let cp = original.p_wave_speed();
        let cs = original.s_wave_speed();

        // Reconstruct from wave speeds
        let reconstructed = ElasticPropertyData::from_wave_speeds(original.density, cp, cs);

        // Verify Lamé parameters match (within numerical tolerance)
        assert!((reconstructed.lambda - original.lambda).abs() / original.lambda < 1e-10);
        assert!((reconstructed.mu - original.mu).abs() / original.mu < 1e-10);

        // Verify derived properties match
        assert!(
            (reconstructed.youngs_modulus() - original.youngs_modulus()).abs()
                / original.youngs_modulus()
                < 1e-10
        );
        assert!((reconstructed.poisson_ratio() - original.poisson_ratio()).abs() < 1e-10);
    }

    // Electromagnetic Property Tests
    // ================================================================================================

    #[test]
    fn test_em_wave_speed() {
        let vacuum = ElectromagneticPropertyData::vacuum();
        const C0: f64 = 299_792_458.0;

        assert!((vacuum.wave_speed() - C0).abs() < 1.0);
    }

    #[test]
    fn test_em_refractive_index() {
        let water = ElectromagneticPropertyData::water();

        // n = √(ε_r μ_r) ≈ √80 ≈ 8.94
        let n = water.refractive_index();
        assert!((n - 80.0_f64.sqrt()).abs() < 0.1);
    }

    #[test]
    fn test_em_skin_depth() {
        let copper = ElectromagneticPropertyData::new(1.0, 1.0, 5.96e7, None).unwrap();

        // Copper at 1 MHz should have skin depth ~66 μm
        let delta = copper.skin_depth(1e6);
        assert!((delta - 66e-6).abs() < 5e-6);
    }

    #[test]
    fn test_em_validation() {
        // ε_r < 1 should fail
        assert!(ElectromagneticPropertyData::new(0.5, 1.0, 0.0, None).is_err());

        // Valid parameters should succeed
        assert!(ElectromagneticPropertyData::new(80.0, 1.0, 0.005, None).is_ok());
    }

    // Strength Property Tests
    // ================================================================================================

    #[test]
    fn test_strength_hardness_estimate() {
        let sigma_y = 250e6; // Pa
        let hardness = StrengthPropertyData::estimate_hardness(sigma_y);
        assert!((hardness - 3.0 * sigma_y).abs() < 1e-6);
    }

    #[test]
    fn test_strength_validation() {
        // σ_u < σ_y should fail
        assert!(StrengthPropertyData::new(250e6, 200e6, 750e6, 10.0).is_err());

        // Valid parameters should succeed
        assert!(StrengthPropertyData::new(250e6, 400e6, 750e6, 10.0).is_ok());
    }

    // Thermal Property Tests
    // ================================================================================================

    #[test]
    fn test_thermal_diffusivity() {
        let water = ThermalPropertyData::water();

        // α = k/(ρc)
        let alpha = water.thermal_diffusivity();
        let expected = 0.598 / (998.0 * 4182.0);
        assert!((alpha - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_thermal_bioheat_detection() {
        let water = ThermalPropertyData::water();
        let tissue = ThermalPropertyData::soft_tissue();

        assert!(!water.has_bioheat_parameters());
        assert!(tissue.has_bioheat_parameters());
    }

    #[test]
    fn test_thermal_validation() {
        // Negative conductivity should fail
        assert!(ThermalPropertyData::new(-0.6, 4180.0, 1000.0, None, None).is_err());

        // Valid parameters should succeed
        assert!(ThermalPropertyData::new(0.6, 4180.0, 1000.0, None, None).is_ok());
    }

    // Material Properties Composite Tests
    // ================================================================================================

    #[test]
    fn test_material_acoustic_only() {
        let water = MaterialProperties::acoustic_only(AcousticPropertyData::water());

        assert!(water.elastic.is_none());
        assert!(water.electromagnetic.is_none());
        assert!(water.strength.is_none());
    }

    #[test]
    fn test_material_builder() {
        let material = MaterialProperties::builder()
            .acoustic(AcousticPropertyData::water())
            .thermal(ThermalPropertyData::water())
            .build();

        assert!(material.thermal.is_some());
        assert!(material.elastic.is_none());
    }

    #[test]
    fn test_material_presets() {
        let water = MaterialProperties::water();
        let tissue = MaterialProperties::soft_tissue();
        let bone = MaterialProperties::bone();
        let steel = MaterialProperties::steel();

        assert!(water.thermal.is_some());
        assert!(tissue.electromagnetic.is_some());
        assert!(bone.elastic.is_some());
        assert!(steel.strength.is_some());
    }

    #[test]
    #[should_panic(expected = "Acoustic properties are required")]
    fn test_material_builder_missing_acoustic() {
        MaterialProperties::builder()
            .thermal(ThermalPropertyData::water())
            .build();
    }

    // Property Test: Physical Bounds
    // ================================================================================================

    #[test]
    fn test_property_physical_bounds() {
        // Test acoustic bounds
        let acoustic = AcousticPropertyData::water();
        assert!(acoustic.density > 0.0);
        assert!(acoustic.sound_speed > 0.0);
        assert!(acoustic.impedance() > 0.0);

        // Test elastic bounds
        let elastic = ElasticPropertyData::steel();
        assert!(elastic.density > 0.0);
        assert!(elastic.mu > 0.0);
        assert!(elastic.lambda >= 0.0);
        assert!(elastic.poisson_ratio() > -1.0 && elastic.poisson_ratio() < 0.5);

        // Test EM bounds
        let em = ElectromagneticPropertyData::water();
        assert!(em.permittivity >= 1.0);
        assert!(em.permeability >= 0.0);
        assert!(em.conductivity >= 0.0);

        // Test optical bounds
        let optical = OpticalPropertyData::soft_tissue();
        assert!(optical.absorption_coefficient >= 0.0);
        assert!(optical.scattering_coefficient >= 0.0);
        assert!(optical.anisotropy >= -1.0 && optical.anisotropy <= 1.0);
        assert!(optical.refractive_index >= 1.0);
    }

    // Optical Property Tests
    // ================================================================================================

    #[test]
    fn test_optical_total_attenuation() {
        let props = OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).unwrap();
        assert_eq!(props.total_attenuation(), 110.0);
    }

    #[test]
    fn test_optical_reduced_scattering() {
        let props = OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).unwrap();
        approx::assert_relative_eq!(props.reduced_scattering(), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_optical_albedo() {
        let props = OpticalPropertyData::new(10.0, 90.0, 0.8, 1.4).unwrap();
        assert!((props.albedo() - 0.9).abs() < 1e-10); // 90/100
    }

    #[test]
    fn test_optical_mean_free_path() {
        let props = OpticalPropertyData::new(1.0, 99.0, 0.9, 1.4).unwrap();
        assert_eq!(props.mean_free_path(), 0.01); // 1/100
    }

    #[test]
    fn test_optical_fresnel_reflectance() {
        let water = OpticalPropertyData::water();
        let reflectance = water.fresnel_reflectance_normal();
        // R = ((1 - 1.33)/(1 + 1.33))^2 ≈ 0.02
        assert!((reflectance - 0.02).abs() < 0.001);
    }

    #[test]
    fn test_optical_validation() {
        // Negative absorption should fail
        assert!(OpticalPropertyData::new(-1.0, 100.0, 0.9, 1.4).is_err());

        // Invalid anisotropy should fail
        assert!(OpticalPropertyData::new(10.0, 100.0, 1.5, 1.4).is_err());

        // Invalid refractive index should fail
        assert!(OpticalPropertyData::new(10.0, 100.0, 0.9, 0.5).is_err());

        // Valid parameters should succeed
        assert!(OpticalPropertyData::new(10.0, 100.0, 0.9, 1.4).is_ok());
    }

    #[test]
    fn test_optical_presets() {
        let water = OpticalPropertyData::water();
        let tissue = OpticalPropertyData::soft_tissue();
        let blood = OpticalPropertyData::blood_oxygenated();

        assert!(water.absorption_coefficient < tissue.absorption_coefficient);
        assert!(blood.scattering_coefficient > water.scattering_coefficient);
        assert!(tissue.anisotropy > 0.8); // Forward-scattering
    }

    #[test]
    fn test_optical_penetration_depth() {
        let props = OpticalPropertyData::soft_tissue();
        let depth = props.penetration_depth();

        // Should be positive and reasonable for tissue
        assert!(depth > 0.0);
        assert!(depth < 1.0); // Less than 1 meter for typical tissue
    }
}
