//! PyO3 config builder wrappers for focused simulation configuration.
//!
//! These Python-facing classes replace the scattered setter methods on the
//! `Simulation` pyclass. Each config type bundles all parameters for one
//! simulation concern (PML boundary, Helmholtz wavenumber, nonlinear acoustics,
//! thermal coupling) into a single well-typed object that can be built with
//! a fluent interface.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kwavers::simulation::{
    HelmholtzConfig as KwaversHelmholtzConfig,
    NonlinearConfig as KwaversNonlinearConfig,
    PmlConfig as KwaversPmlConfig,
    PoroelasticConfig as KwaversPoroelasticConfig,
    ThermalConfig as KwaversThermalConfig,
};

// ============================================================================
// PmlConfig
// ============================================================================

/// PML (perfectly matched layer) absorbing boundary configuration.
///
/// Controls the thickness, absorption profile, and placement of convolutional
/// perfectly matched layers at domain boundaries. Equivalent to k-Wave's
/// ``PMLSize``, ``PMLInside``, and ``PMLAlpha`` settings.
///
/// Examples
/// --------
/// >>> pml = PmlConfig().with_size(20).with_alpha(2.0)
/// >>> pml = PmlConfig().with_size_xyz(20, 10, 20)
#[pyclass(name = "PmlConfig")]
#[derive(Clone, Debug, Default)]
pub struct PmlConfig {
    pub(crate) inner: KwaversPmlConfig,
}

#[pymethods]
impl PmlConfig {
    /// Create a new PmlConfig with default settings.
    #[new]
    fn new() -> Self {
        Self { inner: KwaversPmlConfig::default() }
    }

    /// Set uniform PML thickness (grid cells on each face).
    ///
    /// Parameters
    /// ----------
    /// size : int
    ///     Number of grid points for PML on each face. Typical: 10-40.
    ///
    /// Returns
    /// -------
    /// PmlConfig
    ///     Self for chaining.
    fn with_size(mut slf: PyRefMut<'_, Self>, size: usize) -> PyRefMut<'_, Self> {
        slf.inner.size = Some(size);
        slf
    }

    /// Set per-axis PML thickness ``(x, y, z)`` for k-Wave parity.
    fn with_size_xyz(mut slf: PyRefMut<'_, Self>, x: usize, y: usize, z: usize) -> PyRefMut<'_, Self> {
        slf.inner.size = Some(x.max(y).max(z));
        slf.inner.size_xyz = Some((x, y, z));
        slf
    }

    /// Set whether PML is inside the computational domain.
    ///
    /// ``inside=True`` (default) places PML within the grid.
    /// ``inside=False`` pads the grid, expanding the domain.
    fn with_inside(mut slf: PyRefMut<'_, Self>, inside: bool) -> PyRefMut<'_, Self> {
        slf.inner.inside = inside;
        slf
    }

    /// Set uniform PML absorption factor (k-Wave ``pml_alpha``).
    ///
    /// Parameters
    /// ----------
    /// alpha : float
    ///     Absorption factor. Default: 2.0. Use 0.0 to disable PML.
    fn with_alpha(mut slf: PyRefMut<'_, Self>, alpha: f64) -> PyRefMut<'_, Self> {
        slf.inner.alpha_xyz = Some((alpha, alpha, alpha));
        slf
    }

    /// Set per-axis PML absorption factors.
    fn with_alpha_xyz(
        mut slf: PyRefMut<'_, Self>,
        ax: f64, ay: f64, az: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner.alpha_xyz = Some((ax, ay, az));
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "PmlConfig(size={:?}, inside={}, alpha_xyz={:?})",
            self.inner.size, self.inner.inside, self.inner.alpha_xyz
        )
    }
}

// ============================================================================
// HelmholtzConfig
// ============================================================================

/// Helmholtz frequency-domain solver configuration.
///
/// Controls the wavenumber derivation for Helmholtz/BEM solvers,
/// decoupling the frequency-domain solve from the time step ``dt``.
///
/// Examples
/// --------
/// >>> hc = HelmholtzConfig().with_frequency(1e6)  # 1 MHz
#[pyclass(name = "HelmholtzConfig")]
#[derive(Clone, Debug, Default)]
pub struct HelmholtzConfig {
    pub(crate) inner: KwaversHelmholtzConfig,
}

#[pymethods]
impl HelmholtzConfig {
    #[new]
    fn new() -> Self {
        Self { inner: KwaversHelmholtzConfig::default() }
    }

    /// Set the source frequency for wavenumber derivation.
    ///
    /// Parameters
    /// ----------
    /// frequency : float
    ///     Source frequency in Hz (e.g., ``1e6`` for 1 MHz).
    fn with_frequency(mut slf: PyRefMut<'_, Self>, frequency: f64) -> PyResult<PyRefMut<'_, Self>> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Helmholtz frequency must be positive (Hz)"));
        }
        slf.inner.frequency = Some(frequency);
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        format!("HelmholtzConfig(frequency={:?})", self.inner.frequency)
    }
}

// ============================================================================
// NonlinearConfig
// ============================================================================

/// Nonlinear acoustics configuration.
///
/// Controls the Westervelt nonlinear source term and power-law absorption
/// parameters. Equivalent to k-Wave's ``medium.BonA`` and
/// ``medium.alpha_coeff`` / ``medium.alpha_power``.
///
/// Examples
/// --------
/// >>> nl = NonlinearConfig().with_enabled().with_alpha_coeff(0.75)
#[pyclass(name = "NonlinearConfig")]
#[derive(Clone, Debug, Default)]
pub struct NonlinearConfig {
    pub(crate) inner: KwaversNonlinearConfig,
}

#[pymethods]
impl NonlinearConfig {
    #[new]
    fn new() -> Self {
        Self { inner: KwaversNonlinearConfig::default() }
    }

    /// Enable the Westervelt nonlinear source term.
    fn with_enabled(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.enabled = true;
        slf
    }

    /// Set medium absorption coefficient [dB/(MHz^y·cm)].
    ///
    /// Parameters
    /// ----------
    /// coeff : float
    ///     Absorption coefficient in dB/(MHz^y·cm). k-Wave default: 0.0.
    fn with_alpha_coeff(mut slf: PyRefMut<'_, Self>, coeff: f64) -> PyRefMut<'_, Self> {
        slf.inner.alpha_coeff = coeff;
        slf
    }

    /// Set medium absorption power-law exponent.
    ///
    /// Parameters
    /// ----------
    /// power : float
    ///     Power-law exponent. k-Wave default: 1.5 (for tissue).
    fn with_alpha_power(mut slf: PyRefMut<'_, Self>, power: f64) -> PyRefMut<'_, Self> {
        slf.inner.alpha_power = power;
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "NonlinearConfig(enabled={}, alpha_coeff={}, alpha_power={})",
            self.inner.enabled, self.inner.alpha_coeff, self.inner.alpha_power
        )
    }
}

// ============================================================================
// PoroelasticConfig
// ============================================================================

/// Biot poroelastic material configuration.
///
/// Provides poroelastic-specific material properties for the Biot
/// time-domain solver.  When attached to a ``Simulation``, values are
/// routed through the solver config instead of falling back to SSOT
/// defaults derived from the ``Medium`` trait.
///
/// Defaults represent water-saturated soft tissue.
///
/// Examples
/// --------
/// >>> poro = PoroelasticConfig()
/// >>> poro = PoroelasticConfig().with_porosity(0.3).with_permeability(1e-9)
#[pyclass(name = "PoroelasticConfig")]
#[derive(Clone, Debug, Default)]
pub struct PoroelasticConfig {
    pub(crate) inner: KwaversPoroelasticConfig,
}

#[pymethods]
impl PoroelasticConfig {
    /// Create a new PoroelasticConfig with soft-tissue defaults.
    #[new]
    fn new() -> Self {
        Self { inner: KwaversPoroelasticConfig::default() }
    }

    /// Set porosity (0 < φ < 1).
    fn with_porosity(mut slf: PyRefMut<'_, Self>, porosity: f64) -> PyResult<PyRefMut<'_, Self>> {
        if porosity <= 0.0 || porosity >= 1.0 {
            return Err(PyValueError::new_err("porosity must be in (0, 1)"));
        }
        slf.inner.porosity = porosity;
        Ok(slf)
    }

    /// Set intrinsic permeability [m²].
    fn with_permeability(mut slf: PyRefMut<'_, Self>, perm: f64) -> PyResult<PyRefMut<'_, Self>> {
        if perm <= 0.0 {
            return Err(PyValueError::new_err("permeability must be > 0"));
        }
        slf.inner.permeability = perm;
        Ok(slf)
    }

    /// Set explicit tortuosity (overrides porosity-derived default).
    fn with_tortuosity(mut slf: PyRefMut<'_, Self>, tort: f64) -> PyResult<PyRefMut<'_, Self>> {
        if tort < 1.0 {
            return Err(PyValueError::new_err("tortuosity must be >= 1"));
        }
        slf.inner.tortuosity = Some(tort);
        Ok(slf)
    }

    /// Set fluid properties in one call.
    fn with_fluid(
        mut slf: PyRefMut<'_, Self>,
        density: f64, bulk_modulus: f64, viscosity: f64,
    ) -> PyResult<PyRefMut<'_, Self>> {
        if density <= 0.0 || bulk_modulus <= 0.0 || viscosity <= 0.0 {
            return Err(PyValueError::new_err(
                "fluid density, bulk_modulus, viscosity must be > 0",
            ));
        }
        slf.inner.fluid_density = density;
        slf.inner.fluid_bulk_modulus = bulk_modulus;
        slf.inner.fluid_viscosity = viscosity;
        Ok(slf)
    }

    fn __repr__(&self) -> String {
        format!(
            "PoroelasticConfig(porosity={}, perm={:.1e}, tort={:?})",
            self.inner.porosity,
            self.inner.permeability,
            self.inner.tortuosity,
        )
    }
}

// ============================================================================
// ThermalConfig
// ============================================================================

/// Acoustic→thermal coupling configuration for PSTD-thermal co-simulation.
///
/// Drives a coupled PSTD acoustic + Pennes bioheat simulation.
/// Default properties follow ICRU Report 44 soft-tissue values.
///
/// Examples
/// --------
/// >>> thermal = ThermalConfig(center_frequency=1e6).with_bioheat()
/// >>> thermal = ThermalConfig(1e6, n_acoustic_per_thermal=10)
#[pyclass(name = "ThermalConfig")]
#[derive(Clone, Debug)]
pub struct ThermalConfig {
    pub(crate) inner: KwaversThermalConfig,
}

#[pymethods]
impl ThermalConfig {
    /// Create a new ThermalConfig.
    ///
    /// Parameters
    /// ----------
    /// center_frequency : float
    ///     Center frequency [Hz] for α(ω_c) evaluation (required).
    /// n_acoustic_per_thermal : int, default 1
    ///     Acoustic steps per thermal update.
    /// thermal_conductivity : float, default 0.5
    ///     k [W/(m·K)].
    /// density : float, default 1000.0
    ///     ρ [kg/m³].
    /// specific_heat : float, default 3600.0
    ///     cp [J/(kg·K)].
    /// enable_bioheat : bool, default False
    ///     Enable Pennes perfusion + metabolic terms.
    /// perfusion_rate : float, default 5e-3
    ///     w_b [1/s].
    /// blood_density : float, default 1050.0
    ///     ρ_b [kg/m³].
    /// blood_specific_heat : float, default 3840.0
    ///     c_b [J/(kg·K)].
    /// arterial_temperature : float, default 37.0
    ///     T_a [°C].
    /// metabolic_heat : float, default 0.0
    ///     Q_m [W/m³].
    /// initial_temperature : float, default 37.0
    ///     T_0 [°C].
    /// track_thermal_dose : bool, default True
    ///     Compute CEM43 dose field.
    /// dt_thermal : float or None
    ///     Thermal time step [s]. Default: ``n_acoustic_per_thermal * dt_acoustic``.
    #[new]
    #[pyo3(signature = (
        center_frequency,
        n_acoustic_per_thermal = 1,
        thermal_conductivity = 0.5,
        density = 1000.0,
        specific_heat = 3600.0,
        enable_bioheat = false,
        perfusion_rate = 0.005,
        blood_density = 1050.0,
        blood_specific_heat = 3840.0,
        arterial_temperature = 37.0,
        metabolic_heat = 0.0,
        initial_temperature = 37.0,
        track_thermal_dose = true,
        dt_thermal = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        center_frequency: f64,
        n_acoustic_per_thermal: usize,
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
        dt_thermal: Option<f64>,
    ) -> PyResult<Self> {
        if center_frequency <= 0.0 {
            return Err(PyValueError::new_err("center_frequency must be > 0"));
        }
        if n_acoustic_per_thermal == 0 {
            return Err(PyValueError::new_err("n_acoustic_per_thermal must be >= 1"));
        }
        if thermal_conductivity <= 0.0 || density <= 0.0 || specific_heat <= 0.0 {
            return Err(PyValueError::new_err(
                "thermal_conductivity, density, specific_heat must be > 0",
            ));
        }
        Ok(Self {
            inner: KwaversThermalConfig {
                thermal_conductivity,
                density,
                specific_heat,
                enable_bioheat,
                perfusion_rate,
                blood_density,
                blood_specific_heat,
                arterial_temperature_c: arterial_temperature,
                metabolic_heat,
                initial_temperature_c: initial_temperature,
                track_thermal_dose,
                center_frequency_hz: center_frequency,
                n_acoustic_per_thermal,
                dt_thermal,
            },
        })
    }

    /// Enable bioheat (Pennes perfusion + metabolic) terms.
    fn with_bioheat(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.inner.enable_bioheat = true;
        slf
    }

    /// Set thermal material properties.
    fn with_material(
        mut slf: PyRefMut<'_, Self>,
        k: f64, rho: f64, cp: f64,
    ) -> PyRefMut<'_, Self> {
        slf.inner.thermal_conductivity = k;
        slf.inner.density = rho;
        slf.inner.specific_heat = cp;
        slf
    }

    /// Set acoustic/thermal step ratio.
    fn with_steps_per_thermal(
        mut slf: PyRefMut<'_, Self>,
        n: usize,
    ) -> PyRefMut<'_, Self> {
        slf.inner.n_acoustic_per_thermal = n;
        slf
    }

    fn __repr__(&self) -> String {
        format!(
            "ThermalConfig(freq={:.1e}, n_therm={}, bioheat={})",
            self.inner.center_frequency_hz,
            self.inner.n_acoustic_per_thermal,
            self.inner.enable_bioheat,
        )
    }
}
