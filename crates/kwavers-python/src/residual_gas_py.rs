//! PyO3 binding for the 3-D residual cavitation-gas (lacuna) void-fraction field.
//!
//! Exposes [`kwavers_simulation::multi_physics::residual_gas::ResidualGasField`] —
//! the per-voxel gas void fraction `β(x)` with Epstein–Plesset dissolution kinetics
//! and Wood / Commander–Prosperetti acoustic-property coupling — so a treatment
//! pipeline can evolve a genuine 3-D lacuna field over a sonication schedule and
//! feed the resulting heterogeneous sound-speed and attenuation fields into the
//! PSTD propagation. This is the single source of truth for the residual-gas
//! physics; the Python layer only marshals arrays.

use kwavers_physics::acoustics::bubble_dynamics::{EpsteinPlessetDissolution, GasDiffusionParams};
use kwavers_simulation::multi_physics::residual_gas::ResidualGasField;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// 3-D residual cavitation-gas (lacuna) void-fraction field `β(x)` [-].
///
/// Each therapy pulse `deposit`s fresh gas (resetting the representative residual
/// radius to the nucleation radius); between pulses `dissolve` evolves the cloud by
/// Epstein–Plesset kinetics so `β` decays as `(R(t)/R₀)³`. `sound_speed_field` and
/// `attenuation_field` return the Wood / Commander–Prosperetti acoustic properties
/// of the gas-laden medium for the next pulse.
#[pyclass(name = "ResidualGasField")]
pub struct PyResidualGasField {
    inner: ResidualGasField,
    shape: (usize, usize, usize),
}

#[pymethods]
impl PyResidualGasField {
    /// Create an empty field (`β = 0`) on an `(nx, ny, nz)` grid with freshly
    /// nucleated residual bubbles of equilibrium radius `deposit_radius_m`.
    #[new]
    #[pyo3(signature = (nx, ny, nz, deposit_radius_m = 3.0e-6))]
    fn new(nx: usize, ny: usize, nz: usize, deposit_radius_m: f64) -> Self {
        Self {
            inner: ResidualGasField::new((nx, ny, nz), deposit_radius_m),
            shape: (nx, ny, nz),
        }
    }

    /// Deposit freshly nucleated gas: add the per-voxel volume fraction
    /// `gas_fraction` to `β` (clamped < 1) and reset the representative radius.
    fn deposit(&mut self, gas_fraction: PyReadonlyArray3<'_, f64>) -> PyResult<()> {
        let a = gas_fraction.as_array();
        if a.dim() != self.shape {
            return Err(PyValueError::new_err(format!(
                "gas_fraction shape {:?} != field shape {:?}",
                a.dim(),
                self.shape
            )));
        }
        self.inner.deposit(a);
        Ok(())
    }

    /// Evolve the residual cloud over a rest interval `dt_s` by Epstein–Plesset
    /// dissolution at dissolved-gas saturation `saturation_fraction` (f = C∞/C_s).
    #[pyo3(signature = (dt_s, saturation_fraction = 0.5))]
    fn dissolve(&mut self, dt_s: f64, saturation_fraction: f64) {
        let model =
            EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(saturation_fraction));
        self.inner.dissolve(dt_s, &model);
    }

    /// Current void-fraction field `β(x)` as an (nx, ny, nz) array.
    fn void_fraction(&self, py: Python<'_>) -> Py<PyArray3<f64>> {
        self.inner.void_fraction().to_owned().into_pyarray(py).into()
    }

    /// Wood (1930) effective sound-speed field [m/s] for the gas-laden medium.
    #[pyo3(signature = (c_liquid, rho_liquid, c_gas = 343.0, rho_gas = 1.2))]
    fn sound_speed_field(
        &self,
        py: Python<'_>,
        c_liquid: f64,
        rho_liquid: f64,
        c_gas: f64,
        rho_gas: f64,
    ) -> Py<PyArray3<f64>> {
        self.inner
            .sound_speed_field(c_liquid, rho_liquid, c_gas, rho_gas)
            .into_pyarray(py)
            .into()
    }

    /// Commander–Prosperetti excess-attenuation field [Np/m] at `freq_hz`.
    #[pyo3(signature = (freq_hz, c_liquid, rho_liquid, mu_liquid = 1.0e-3,
                        p0_pa = 101_325.0, polytropic = 1.4))]
    fn attenuation_field(
        &self,
        py: Python<'_>,
        freq_hz: f64,
        c_liquid: f64,
        rho_liquid: f64,
        mu_liquid: f64,
        p0_pa: f64,
        polytropic: f64,
    ) -> Py<PyArray3<f64>> {
        self.inner
            .attenuation_field(freq_hz, c_liquid, rho_liquid, mu_liquid, p0_pa, polytropic)
            .into_pyarray(py)
            .into()
    }

    /// Representative residual-bubble radius [m] (shrinks with dissolution).
    fn representative_radius(&self) -> f64 {
        self.inner.representative_radius()
    }

    /// Peak void fraction anywhere in the field.
    fn peak_void_fraction(&self) -> f64 {
        self.inner.peak_void_fraction()
    }

    /// Total residual gas volume `Σ β·dV` [m³] for voxel volume `dv_m3`.
    fn total_gas_volume(&self, dv_m3: f64) -> f64 {
        self.inner.total_gas_volume(dv_m3)
    }
}
