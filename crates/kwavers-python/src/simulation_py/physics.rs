//! Physics switches and coefficients on a `Simulation`: nonlinearity, axisymmetry, absorption, Helmholtz frequency.

use super::*;

#[pymethods]
impl Simulation {
    /// Enable or disable the Westervelt nonlinear acoustic source term.
    fn set_nonlinear(&mut self, enable: bool) {
        self.enable_nonlinear = enable;
        if let Some(ref mut cfg) = self.nonlinear_config {
            cfg.enabled = enable;
        }
    }

    /// Return whether the Westervelt nonlinear term is enabled.
    #[getter]
    fn nonlinear(&self) -> bool {
        self.enable_nonlinear
    }

    /// Enable axisymmetric (CylindricalAS) geometry for 2-D radial simulations.
    fn set_axisymmetric(&mut self, enable: bool) {
        self.axisymmetric = enable;
    }

    /// Return whether axisymmetric geometry is enabled.
    #[getter]
    fn axisymmetric(&self) -> bool {
        self.axisymmetric
    }

    /// Set medium absorption coefficient (k-Wave `medium.alpha_coeff`).
    fn set_alpha_coeff(&mut self, alpha: f64) {
        self.alpha_coeff = alpha;
        if let Some(ref mut cfg) = self.nonlinear_config {
            cfg.alpha_coeff = alpha;
        }
    }

    /// Set medium absorption power law exponent (k-Wave `medium.alpha_power`).
    fn set_alpha_power(&mut self, power: f64) {
        self.alpha_power = power;
        if let Some(ref mut cfg) = self.nonlinear_config {
            cfg.alpha_power = power;
        }
    }

    /// Select causal dispersion or absorption without it.
    ///
    /// ``Causal`` (default) rejects a configured ``alpha_power = 1``, where
    /// `tan(pi*y/2)` is singular; a medium that supplies a degenerate exponent
    /// (``y = 1`` or ``0``) is given the configured ``alpha_power`` instead.
    /// ``Omitted`` is k-Wave's `alpha_mode = 'no_dispersion'`: ``y = 1`` is
    /// used as given, from the medium or the configuration.
    fn set_power_law_dispersion(&mut self, dispersion: PyPowerLawDispersion) {
        self.power_law_dispersion = dispersion.into();
        if let Some(ref mut cfg) = self.nonlinear_config {
            cfg.power_law_dispersion = dispersion.into();
        }
    }

    /// Get the power-law dispersion mode.
    #[getter]
    fn power_law_dispersion(&self) -> PyPowerLawDispersion {
        self.power_law_dispersion.into()
    }

    /// Get medium absorption coefficient [dB/(MHz^y·cm)].
    #[getter]
    fn alpha_coeff(&self) -> f64 {
        self.alpha_coeff
    }

    /// Get medium absorption power law exponent.
    #[getter]
    fn alpha_power(&self) -> f64 {
        self.alpha_power
    }

    /// Set the Helmholtz solver frequency for wavenumber control.
    ///
    /// When a frequency is set, the Helmholtz solver derives the wavenumber
    /// independently from the time step `dt`:
    ///
    /// ```text
    /// k = 2π · frequency / cₘₐₓ
    /// ```
    ///
    /// When no frequency is set (the default), the wavenumber is derived from
    /// `dt` as `k = 2π / (cₘₐₓ · dt)`, which is convenient for quick
    /// prototyping but couples the frequency-domain solve to the time step.
    ///
    /// Parameters
    /// ----------
    /// frequency : float
    ///     Source frequency in Hz (e.g., `1e6` for 1 MHz).
    ///
    /// Examples
    /// --------
    /// >>> sim.set_helmholtz_wavenumber(1e6)  # 1 MHz Helmholtz solve
    fn set_helmholtz_wavenumber(&mut self, frequency: f64) -> PyResult<()> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err(
                "Helmholtz frequency must be positive (Hz)",
            ));
        }
        self.helmholtz_frequency = Some(frequency);
        if let Some(ref mut cfg) = self.helmholtz_config {
            cfg.frequency = Some(frequency);
        }
        Ok(())
    }

    /// Get the currently configured Helmholtz frequency, if any.
    #[getter]
    fn helmholtz_frequency(&self) -> Option<f64> {
        self.helmholtz_frequency
    }

    // ── Run (delegated to kwavers SimulationRunner) ─────────────────────────
}
