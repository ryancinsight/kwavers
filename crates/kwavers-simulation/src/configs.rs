//! Focused configuration types for simulation solver orchestration.
//!
//! These config structs replace the scattered field-setter pattern on the Python
//! `Simulation` class. Each config bundles all parameters relevant to one
//! simulation concern (PML boundary, Helmholtz wavenumber, nonlinear acoustics,
//! thermal coupling) so the solver dispatch can accept a small number of
//! well-typed config objects instead of 20+ loose parameters.

use aequitas::systems::si::quantities::{
    Frequency, MassDensity, ReciprocalTime, SpecificHeatCapacity, ThermalConductivity,
    ThermodynamicTemperature, Time, VolumetricPowerDensity,
};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

// ============================================================================
// PmlConfig — PML absorbing boundary configuration
// ============================================================================

/// PML (perfectly matched layer) absorbing boundary configuration.
///
/// Controls the thickness, absorption profile, and placement of convolutional
/// perfectly matched layers at domain boundaries. Equivalent to k-Wave's
/// `PMLSize`, `PMLInside`, and `PMLAlpha` settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PmlConfig {
    /// Uniform PML thickness (grid cells on each face).
    /// `None` means auto-compute from grid dimensions (default 20).
    pub size: Option<usize>,
    /// Per-axis PML thickness `(x, y, z)` for k-Wave parity.
    /// When set, takes precedence over `size` for per-dimension control.
    pub size_xyz: Option<(usize, usize, usize)>,
    /// Whether PML is inside the computational domain (`true`) or padded
    /// outside it (`false`). Default: `true` (k-Wave default).
    pub inside: bool,
    /// Per-dimension PML absorption factor (k-Wave `pml_alpha`).
    /// `None` means use the CPML default. Set to `Some((ax, ay, az))` to
    /// control the absorption ramp steepness per axis.
    pub alpha_xyz: Option<(f64, f64, f64)>,
}

impl Default for PmlConfig {
    fn default() -> Self {
        Self {
            size: None,
            size_xyz: None,
            inside: true,
            alpha_xyz: None,
        }
    }
}

impl PmlConfig {
    /// Builder: set uniform PML thickness.
    #[must_use]
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// Builder: set per-axis PML thickness.
    #[must_use]
    pub fn with_size_xyz(mut self, x: usize, y: usize, z: usize) -> Self {
        self.size = Some(x.max(y).max(z));
        self.size_xyz = Some((x, y, z));
        self
    }

    /// Builder: set PML inside the domain.
    #[must_use]
    pub fn with_inside(mut self, inside: bool) -> Self {
        self.inside = inside;
        self
    }

    /// Builder: set uniform PML absorption factor.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha_xyz = Some((alpha, alpha, alpha));
        self
    }

    /// Builder: set per-axis PML absorption factors.
    #[must_use]
    pub fn with_alpha_xyz(mut self, ax: f64, ay: f64, az: f64) -> Self {
        self.alpha_xyz = Some((ax, ay, az));
        self
    }

    /// Compute the effective PML thickness in cells, clamping to domain limits.
    #[must_use]
    pub fn effective_thickness(&self, nx: usize, ny: usize, nz: usize) -> (usize, usize) {
        let mut min_dim = usize::MAX;
        for dim in [nx, ny, nz] {
            if dim > 1 {
                min_dim = min_dim.min(dim);
            }
        }
        let min_dim = if min_dim == usize::MAX { 1 } else { min_dim };
        let max_allowed = (min_dim.saturating_sub(2)) / 2;
        let default_thickness = 20_usize.min(max_allowed).max(2);
        let thickness = self.size.unwrap_or(default_thickness).min(max_allowed);
        (thickness, max_allowed)
    }

    /// True if the alpha factors are all zero (effectively no PML absorption).
    #[must_use]
    pub fn alpha_is_zero(&self) -> bool {
        self.alpha_xyz
            .map(|(ax, ay, az)| ax == 0.0 && ay == 0.0 && az == 0.0)
            .unwrap_or(false)
    }
}

// ============================================================================
// HelmholtzConfig — Helmholtz frequency-domain solver configuration
// ============================================================================

/// Helmholtz frequency-domain solver configuration.
///
/// Controls the wavenumber derivation for Helmholtz/BEM solvers, decoupling
/// the frequency-domain solve from the time step `dt`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct HelmholtzConfig {
    /// Source frequency in hertz for wavenumber derivation.
    /// `k = 2π · frequency / cₘₐₓ`.
    /// When `None`, falls back to `k = 2π / (cₘₐₓ · dt)`.
    pub frequency: Option<f64>,
}

impl HelmholtzConfig {
    /// Builder: set the Helmholtz frequency.
    #[must_use]
    pub fn with_frequency(mut self, frequency_hz: f64) -> Self {
        self.frequency = Some(frequency_hz);
        self
    }
}

// ============================================================================
// NonlinearConfig — Nonlinear acoustics configuration
// ============================================================================

/// Nonlinear acoustics configuration.
///
/// Controls the Westervelt nonlinear source term and power-law absorption
/// parameters. Equivalent to k-Wave's `medium.BonA` (nonlinearity) and
/// `medium.alpha_coeff` / `medium.alpha_power` (absorption).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NonlinearConfig {
    /// Enable the Westervelt nonlinear source term in time-domain solvers.
    pub enabled: bool,
    /// Medium absorption coefficient [dB/(MHz^y·cm)] — k-Wave convention.
    /// `0.0` means lossless propagation.
    pub alpha_coeff: f64,
    /// Medium absorption power-law exponent (k-Wave default: 1.5 for tissue).
    pub alpha_power: f64,
}

impl Default for NonlinearConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            alpha_coeff: 0.0,
            alpha_power: 1.5,
        }
    }
}

impl NonlinearConfig {
    /// Builder: enable the Westervelt nonlinear term.
    #[must_use]
    pub fn with_enabled(mut self) -> Self {
        self.enabled = true;
        self
    }

    /// Builder: set absorption coefficient [dB/(MHz^y·cm)].
    #[must_use]
    pub fn with_alpha_coeff(mut self, coeff: f64) -> Self {
        self.alpha_coeff = coeff;
        self
    }

    /// Builder: set absorption power-law exponent.
    #[must_use]
    pub fn with_alpha_power(mut self, power: f64) -> Self {
        self.alpha_power = power;
        self
    }
}

// ============================================================================
// PoroelasticConfig — Biot poroelastic material configuration
// ============================================================================

/// Biot poroelastic material configuration.
///
/// Provides poroelastic-specific material properties for the Biot time-domain
/// solver. When `None` on a run request, the dispatch derives sensible defaults
/// from the `Medium` trait (frame moduli for `porosity` and `tortuosity`) and
/// SSOT constants (water density, viscosity, bulk modulus).
///
/// # Defaults (soft tissue / water-saturated)
///
/// | Parameter | Default | Unit |
/// |-----------|---------|------|
/// | `porosity` | 0.15 | — |
/// | `permeability` | 1e-11 | m² |
/// | `tortuosity` | computed from porosity | — |
/// | `fluid_density` | DENSITY_WATER_NOMINAL | kg/m³ |
/// | `fluid_bulk_modulus` | 2.25e9 | Pa |
/// | `fluid_viscosity` | VISCOSITY_WATER | Pa·s |
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoroelasticConfig {
    /// Porosity (0 < φ < 1).
    pub porosity: f64,
    /// Intrinsic permeability in square metres.
    pub permeability: f64,
    /// Tortuosity (α ≥ 1).  Derived from porosity when `None`:
    /// `tortuosity = 1.0 / porosity.sqrt()`.
    pub tortuosity: Option<f64>,
    /// Fluid density [kg/m³].
    pub fluid_density: f64,
    /// Fluid bulk modulus in pascals.
    pub fluid_bulk_modulus: f64,
    /// Fluid dynamic viscosity [Pa·s].
    pub fluid_viscosity: f64,
}

impl Default for PoroelasticConfig {
    fn default() -> Self {
        Self {
            porosity: 0.15,
            permeability: 1e-11,
            tortuosity: None,
            fluid_density: 1000.0,
            fluid_bulk_modulus: 2.25e9,
            fluid_viscosity: 0.001,
        }
    }
}

impl PoroelasticConfig {
    /// Builder: set porosity.
    #[must_use]
    pub fn with_porosity(mut self, porosity: f64) -> Self {
        self.porosity = porosity;
        self
    }

    /// Builder: set permeability in square metres.
    #[must_use]
    pub fn with_permeability(mut self, permeability: f64) -> Self {
        self.permeability = permeability;
        self
    }

    /// Builder: set explicit tortuosity (overrides porosity-derived default).
    #[must_use]
    pub fn with_tortuosity(mut self, tortuosity: f64) -> Self {
        self.tortuosity = Some(tortuosity);
        self
    }

    /// Builder: set fluid properties in one call.
    #[must_use]
    pub fn with_fluid(mut self, density: f64, bulk_modulus: f64, viscosity: f64) -> Self {
        self.fluid_density = density;
        self.fluid_bulk_modulus = bulk_modulus;
        self.fluid_viscosity = viscosity;
        self
    }

    /// Effective tortuosity — explicit or porosity-derived.
    #[must_use]
    pub fn effective_tortuosity(&self) -> f64 {
        self.tortuosity
            .unwrap_or_else(|| 1.0 / self.porosity.sqrt())
    }
}

// ============================================================================
// ThermalConfig — Acoustic→thermal coupling configuration
// ============================================================================

/// Acoustic→thermal coupling configuration for PSTD-thermal co-simulation.
///
/// Drives a coupled PSTD acoustic + Pennes bioheat simulation. The acoustic
/// volumetric heat source Q = 2α·c·e [W/m³] (Nyborg 1981) is computed from
/// PSTD pressure/velocity fields and passed to the thermal diffusion solver.
///
/// # Thermal properties (defaults: soft tissue, ICRU Report 44)
///
/// | Parameter | Default | Unit |
/// |-----------|---------|------|
/// | `thermal_conductivity` | 0.5 | W/(m·K) |
/// | `density` | 1000.0 | kg/m³ |
/// | `specific_heat` | 3600.0 | J/(kg·K) |
/// | `perfusion_rate` | 5e-3 | 1/s |
/// | `blood_density` | 1050.0 | kg/m³ |
/// | `blood_specific_heat` | 3840.0 | J/(kg·K) |
/// | `arterial_temperature` | 310.15 | K |
/// | `initial_temperature` | 310.15 | K |
#[derive(Debug, Clone, PartialEq)]
pub struct ThermalConfig {
    /// Thermal conductivity k [W/(m·K)].
    pub thermal_conductivity: ThermalConductivity<f64>,
    /// Tissue density ρ [kg/m³].
    pub density: MassDensity<f64>,
    /// Tissue specific heat capacity cp [J/(kg·K)].
    pub specific_heat: SpecificHeatCapacity<f64>,
    /// Enable Pennes bioheat (perfusion + metabolic) terms.
    pub enable_bioheat: bool,
    /// Blood perfusion rate w_b [1/s].
    pub perfusion_rate: ReciprocalTime<f64>,
    /// Blood density ρ_b [kg/m³].
    pub blood_density: MassDensity<f64>,
    /// Blood specific heat c_b [J/(kg·K)].
    pub blood_specific_heat: SpecificHeatCapacity<f64>,
    /// Arterial blood temperature [K].
    pub arterial_temperature: ThermodynamicTemperature<f64>,
    /// Metabolic heat generation Q_m [W/m³].
    pub metabolic_heat: VolumetricPowerDensity<f64>,
    /// Initial tissue temperature [K].
    pub initial_temperature: ThermodynamicTemperature<f64>,
    /// Track CEM43 thermal dose field.
    pub track_thermal_dose: bool,
    /// Center frequency in hertz for α(ω_c) computation.
    pub center_frequency: Frequency<f64>,
    /// Acoustic steps per thermal update (≥ 1).
    pub n_acoustic_per_thermal: usize,
    /// Thermal time step in seconds. `None` uses `n_acoustic_per_thermal * dt_acoustic`.
    pub dt_thermal: Option<Time<f64>>,
}

#[derive(Deserialize)]
struct ThermalConfigRepr {
    thermal_conductivity: f64,
    density: f64,
    specific_heat: f64,
    enable_bioheat: bool,
    perfusion_rate: f64,
    blood_density: f64,
    blood_specific_heat: f64,
    arterial_temperature: f64,
    metabolic_heat: f64,
    initial_temperature: f64,
    track_thermal_dose: bool,
    center_frequency: f64,
    n_acoustic_per_thermal: usize,
    dt_thermal: Option<f64>,
}

impl Serialize for ThermalConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("ThermalConfig", 14)?;
        state.serialize_field(
            "thermal_conductivity",
            &self.thermal_conductivity.into_base(),
        )?;
        state.serialize_field("density", &self.density.into_base())?;
        state.serialize_field("specific_heat", &self.specific_heat.into_base())?;
        state.serialize_field("enable_bioheat", &self.enable_bioheat)?;
        state.serialize_field("perfusion_rate", &self.perfusion_rate.into_base())?;
        state.serialize_field("blood_density", &self.blood_density.into_base())?;
        state.serialize_field("blood_specific_heat", &self.blood_specific_heat.into_base())?;
        state.serialize_field(
            "arterial_temperature",
            &self.arterial_temperature.into_base(),
        )?;
        state.serialize_field("metabolic_heat", &self.metabolic_heat.into_base())?;
        state.serialize_field("initial_temperature", &self.initial_temperature.into_base())?;
        state.serialize_field("track_thermal_dose", &self.track_thermal_dose)?;
        state.serialize_field("center_frequency", &self.center_frequency.into_base())?;
        state.serialize_field("n_acoustic_per_thermal", &self.n_acoustic_per_thermal)?;
        state.serialize_field("dt_thermal", &self.dt_thermal.map(|time| time.into_base()))?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for ThermalConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let repr = ThermalConfigRepr::deserialize(deserializer)?;
        Ok(Self {
            thermal_conductivity: ThermalConductivity::from_base(repr.thermal_conductivity),
            density: MassDensity::from_base(repr.density),
            specific_heat: SpecificHeatCapacity::from_base(repr.specific_heat),
            enable_bioheat: repr.enable_bioheat,
            perfusion_rate: ReciprocalTime::from_base(repr.perfusion_rate),
            blood_density: MassDensity::from_base(repr.blood_density),
            blood_specific_heat: SpecificHeatCapacity::from_base(repr.blood_specific_heat),
            arterial_temperature: ThermodynamicTemperature::from_base(repr.arterial_temperature),
            metabolic_heat: VolumetricPowerDensity::from_base(repr.metabolic_heat),
            initial_temperature: ThermodynamicTemperature::from_base(repr.initial_temperature),
            track_thermal_dose: repr.track_thermal_dose,
            center_frequency: Frequency::from_base(repr.center_frequency),
            n_acoustic_per_thermal: repr.n_acoustic_per_thermal,
            dt_thermal: repr.dt_thermal.map(Time::from_base),
        })
    }
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            thermal_conductivity: ThermalConductivity::from_base(0.5),
            density: MassDensity::from_base(1000.0),
            specific_heat: SpecificHeatCapacity::from_base(3600.0),
            enable_bioheat: false,
            perfusion_rate: ReciprocalTime::from_base(5e-3),
            blood_density: MassDensity::from_base(1050.0),
            blood_specific_heat: SpecificHeatCapacity::from_base(3840.0),
            arterial_temperature: ThermodynamicTemperature::from_base(310.15),
            metabolic_heat: VolumetricPowerDensity::from_base(0.0),
            initial_temperature: ThermodynamicTemperature::from_base(310.15),
            track_thermal_dose: true,
            center_frequency: Frequency::from_base(1.0e6),
            n_acoustic_per_thermal: 1,
            dt_thermal: None,
        }
    }
}

impl ThermalConfig {
    /// Builder: set the center frequency for absorption evaluation.
    #[must_use]
    pub fn with_center_frequency(mut self, frequency: Frequency<f64>) -> Self {
        self.center_frequency = frequency;
        self
    }

    /// Builder: enable bioheat (Pennes perfusion) terms.
    #[must_use]
    pub fn with_bioheat(mut self) -> Self {
        self.enable_bioheat = true;
        self
    }

    /// Builder: set thermal material properties.
    #[must_use]
    pub fn with_material(
        mut self,
        conductivity: ThermalConductivity<f64>,
        density: MassDensity<f64>,
        specific_heat: SpecificHeatCapacity<f64>,
    ) -> Self {
        self.thermal_conductivity = conductivity;
        self.density = density;
        self.specific_heat = specific_heat;
        self
    }

    /// Builder: set acoustic/thermal step ratio.
    #[must_use]
    pub fn with_steps_per_thermal(mut self, n: usize) -> Self {
        self.n_acoustic_per_thermal = n;
        self
    }

    /// Builder: disable thermal dose tracking.
    #[must_use]
    pub fn without_dose(mut self) -> Self {
        self.track_thermal_dose = false;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pml_config_effective_thickness_clamps_to_domain() {
        let cfg = PmlConfig::default().with_size(100);
        let (thickness, max_allowed) = cfg.effective_thickness(32, 32, 32);
        // max_allowed = (32 - 2) / 2 = 15
        assert_eq!(max_allowed, 15);
        assert_eq!(thickness, 15); // clamped from 100 to 15
    }

    #[test]
    fn pml_config_alpha_is_zero_detection() {
        let cfg = PmlConfig::default().with_alpha(0.0);
        assert!(cfg.alpha_is_zero());

        let cfg2 = PmlConfig::default().with_alpha(2.0);
        assert!(!cfg2.alpha_is_zero());
    }

    #[test]
    fn nonlinear_config_builder_chaining() {
        let cfg = NonlinearConfig::default()
            .with_enabled()
            .with_alpha_coeff(0.75)
            .with_alpha_power(1.5);
        assert!(cfg.enabled);
        assert!((cfg.alpha_coeff - 0.75).abs() < 1e-12);
        assert!((cfg.alpha_power - 1.5).abs() < 1e-12);
    }

    #[test]
    fn thermal_config_defaults_match_icru_44_soft_tissue() {
        let cfg = ThermalConfig::default();
        assert!((cfg.thermal_conductivity.into_base() - 0.5).abs() < 1e-12);
        assert!((cfg.density.into_base() - 1000.0).abs() < 1e-12);
        assert!((cfg.specific_heat.into_base() - 3600.0).abs() < 1e-12);
        assert!((cfg.arterial_temperature.into_base() - 310.15).abs() < 1e-12);
    }

    #[test]
    fn thermal_config_serializes_si_values_at_the_boundary() {
        let cfg = ThermalConfig::default()
            .with_center_frequency(Frequency::from_base(2.0e6))
            .with_material(
                ThermalConductivity::from_base(0.6),
                MassDensity::from_base(1020.0),
                SpecificHeatCapacity::from_base(3500.0),
            );
        let encoded = toml::to_string(&cfg).expect("typed config serializes");
        let decoded: ThermalConfig = toml::from_str(&encoded).expect("typed config deserializes");
        assert_eq!(decoded, cfg);
    }

    #[test]
    fn poroelastic_config_effective_tortuosity() {
        // Explicit tortuosity takes precedence.
        let cfg = PoroelasticConfig::default().with_tortuosity(3.0);
        assert!((cfg.effective_tortuosity() - 3.0).abs() < 1e-15);

        // None → derived from porosity: 1 / sqrt(0.25) = 2.0
        let cfg = PoroelasticConfig::default().with_porosity(0.25);
        assert!((cfg.effective_tortuosity() - 2.0).abs() < 1e-15);

        // Default porosity 0.15 → 1 / sqrt(0.15)
        let cfg = PoroelasticConfig::default();
        let expected = 1.0 / 0.15_f64.sqrt();
        assert!((cfg.effective_tortuosity() - expected).abs() < 1e-15);
    }
}
