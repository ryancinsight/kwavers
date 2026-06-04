use kwavers_medium::core::CoreMedium;
use pyo3::prelude::*;

use super::{Medium, MediumInner};

#[pymethods]
impl Medium {
    /// Compressional (P-wave) speed [m/s].
    #[getter]
    fn c_compression(&self) -> f64 {
        let m = self.inner.as_medium();
        let lambda = match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_lambda_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        };
        let mu = match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_mu_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        };
        let rho = m.density(0, 0, 0);
        if rho > 0.0 {
            ((lambda + 2.0 * mu) / rho).sqrt()
        } else {
            0.0
        }
    }

    /// Shear (S-wave) speed [m/s].
    #[getter]
    fn c_shear(&self) -> f64 {
        let m = self.inner.as_medium();
        let mu = match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_mu_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        };
        let rho = m.density(0, 0, 0);
        if rho > 0.0 {
            (mu / rho).sqrt()
        } else {
            0.0
        }
    }

    /// First Lamé parameter λ (Pa).
    #[getter]
    fn lame_lambda(&self) -> f64 {
        match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_lambda_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        }
    }

    /// Second Lamé parameter μ (shear modulus, Pa).
    #[getter]
    fn lame_mu(&self) -> f64 {
        match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_mu_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        }
    }

    /// Sound speed [m/s] (max for heterogeneous).
    #[getter]
    fn sound_speed(&self) -> f64 {
        self.inner.as_medium().max_sound_speed()
    }

    /// Density [kg/m³] (origin voxel for heterogeneous).
    #[getter]
    fn density(&self) -> f64 {
        self.inner.as_medium().density(0, 0, 0)
    }

    /// Whether the medium is homogeneous.
    #[getter]
    fn is_homogeneous(&self) -> bool {
        matches!(self.inner, MediumInner::Homogeneous(_))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            MediumInner::Homogeneous(h) => format!(
                "Medium.homogeneous(sound_speed={:.1}, density={:.1})",
                h.max_sound_speed(),
                h.density(0, 0, 0)
            ),
            MediumInner::Heterogeneous(h) => {
                let shape = h.sound_speed.shape();
                format!(
                    "Medium(heterogeneous, shape=({}, {}, {}), c_max={:.1})",
                    shape[0],
                    shape[1],
                    shape[2],
                    h.max_sound_speed()
                )
            }
        }
    }

    fn __str__(&self) -> String {
        match &self.inner {
            MediumInner::Homogeneous(_) => "Homogeneous Medium".to_string(),
            MediumInner::Heterogeneous(h) => {
                let shape = h.sound_speed.shape();
                format!(
                    "Heterogeneous Medium ({}x{}x{})",
                    shape[0], shape[1], shape[2]
                )
            }
        }
    }
}
