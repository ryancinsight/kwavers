//! Configuration-object setters: PML, Helmholtz, nonlinear, thermal and poroelastic configuration attached to a `Simulation`.

use super::*;

#[pymethods]
impl Simulation {
    /// Attach a pre-built PmlConfig object.
    ///
    /// Replaces any PML settings previously set via individual setters.
    ///
    /// Parameters
    /// ----------
    /// config : PmlConfig
    ///     PML configuration built with ``PmlConfig().with_size(20).with_alpha(2.0)``.
    ///
    /// Examples
    /// --------
    /// >>> pml = PmlConfig().with_size(20).with_alpha(2.0)
    /// >>> sim.set_pml_config(pml)
    fn set_pml_config(&mut self, config: PyPmlConfig) {
        self.pml_size = config.inner.size;
        self.pml_size_xyz = config.inner.size_xyz;
        self.pml_inside = config.inner.inside;
        self.pml_alpha_xyz = config.inner.alpha_xyz;
        self.pml_config = Some(config.inner);
    }

    /// Attach a pre-built HelmholtzConfig object.
    ///
    /// Parameters
    /// ----------
    /// config : HelmholtzConfig
    ///     Helmholtz configuration built with
    ///     ``HelmholtzConfig().with_frequency(1e6)``.
    fn set_helmholtz_config(&mut self, config: PyHelmholtzConfig) {
        self.helmholtz_frequency = config.inner.frequency;
        self.helmholtz_config = Some(config.inner);
    }

    /// Attach a pre-built NonlinearConfig object.
    ///
    /// Parameters
    /// ----------
    /// config : NonlinearConfig
    ///     Nonlinear configuration built with
    ///     ``NonlinearConfig().with_enabled().with_alpha_coeff(0.75)``.
    fn set_nonlinear_config(&mut self, config: PyNonlinearConfig) {
        self.enable_nonlinear = config.inner.enabled;
        self.alpha_coeff = config.inner.alpha_coeff;
        self.alpha_power = config.inner.alpha_power;
        self.nonlinear_config = Some(config.inner);
    }

    /// Attach a pre-built ThermalConfig object.
    ///
    /// When set, ``Simulation.run()`` drives the coupled acoustic-thermal
    /// loop via the PSTD solver. The result's ``thermal_temperature`` (°C)
    /// and ``thermal_dose`` (CEM43 min) fields are populated.
    ///
    /// Parameters
    /// ----------
    /// config : ThermalConfig
    ///     Thermal coupling configuration built with
    ///     ``ThermalConfig(center_frequency=1e6).with_bioheat()``.
    fn set_thermal_config(&mut self, config: PyThermalConfig) {
        self.thermal = Some(config.inner);
    }

    /// Attach a pre-built PoroelasticConfig object.
    ///
    /// When set, routes Biot poroelastic material properties (porosity,
    /// permeability, tortuosity, fluid density/bulk-modulus/viscosity)
    /// through the solver config instead of falling back to SSOT defaults
    /// derived from the ``Medium`` trait.
    ///
    /// Parameters
    /// ----------
    /// config : PoroelasticConfig
    ///     Poroelastic material configuration built with
    ///     ``PoroelasticConfig().with_porosity(0.3).with_permeability(1e-9)``.
    fn set_poroelastic_config(&mut self, config: PyPoroelasticConfig) {
        self.poroelastic = Some(config.inner);
    }

    /// Remove poroelastic material configuration, reverting to SSOT defaults.
    pub fn clear_poroelastic(&mut self) {
        self.poroelastic = None;
    }

    /// True if a poroelastic material configuration is attached.
    #[getter]
    pub fn has_poroelastic(&self) -> bool {
        self.poroelastic.is_some()
    }

    // ── Legacy thermal setter (backward-compatible) ───────────────────────

    /// Attach acoustic→thermal coupling to this simulation (legacy API).
    ///
    /// When set, ``Simulation.run()`` with a PSTD solver drives the coupled
    /// time loop: acoustic heat deposition Q = 2α·c·e [W/m³] feeds the Pennes
    /// bioheat / thermal diffusion solver every ``n_acoustic_per_thermal`` steps.
    ///
    /// Prefer ``set_thermal_config(ThermalConfig(...).with_bioheat())`` for
    /// new code — it directly constructs a ``ThermalConfig`` config object.
    #[pyo3(signature = (
        center_frequency,
        n_acoustic_per_thermal = 1,
        thermal_conductivity = DEFAULT_K,
        density = DEFAULT_RHO,
        specific_heat = DEFAULT_CP,
        enable_bioheat = false,
        perfusion_rate = DEFAULT_WB,
        blood_density = DEFAULT_RHO_B,
        blood_specific_heat = DEFAULT_CPB,
        arterial_temperature = DEFAULT_TA_C,
        metabolic_heat = 0.0,
        initial_temperature = DEFAULT_TA_C,
        track_thermal_dose = true,
        dt_thermal = None,
    ))]
    ///
    /// # Errors
    ///
    /// Raises a Python value error when thermal parameters or the coupling
    /// cadence are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn set_thermal(
        &mut self,
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
    ) -> PyResult<()> {
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
        self.thermal = Some(KwaversThermalConfig {
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
        });
        Ok(())
    }

    /// Remove thermal coupling, reverting to acoustic-only simulation.
    pub fn clear_thermal(&mut self) {
        self.thermal = None;
    }

    /// True if thermal coupling is configured.
    #[getter]
    pub fn has_thermal(&self) -> bool {
        self.thermal.is_some()
    }

    // ── Legacy setters (backward-compatible, sync to config builders) ─────
}
