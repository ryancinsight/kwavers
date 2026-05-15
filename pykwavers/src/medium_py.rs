use numpy::PyReadonlyArray3;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::medium::core::CoreMedium;
use kwavers::domain::medium::heterogeneous::{HeterogeneousFactory, HeterogeneousMedium};
use kwavers::domain::medium::traits::Medium as MediumTrait;
use kwavers::domain::medium::HomogeneousMedium;

use crate::grid_py::Grid;

/// Internal enum holding either homogeneous or heterogeneous medium.
#[derive(Clone, Debug)]
pub(crate) enum MediumInner {
    Homogeneous(Box<HomogeneousMedium>),
    Heterogeneous(Box<HeterogeneousMedium>),
}

impl MediumInner {
    /// Get a reference to the inner medium as a trait object.
    pub(crate) fn as_medium(&self) -> &dyn MediumTrait {
        match self {
            MediumInner::Homogeneous(h) => h.as_ref(),
            MediumInner::Heterogeneous(h) => h.as_ref(),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Medium {
    /// Internal medium (homogeneous or heterogeneous)
    pub(crate) inner: MediumInner,
}

#[pymethods]
impl Medium {
    /// Create a heterogeneous (spatially varying) medium from 3D arrays.
    ///
    /// Parameters
    /// ----------
    /// sound_speed : ndarray (3D float64)
    ///     Spatially varying sound speed [m/s].  Shape must match the grid.
    /// density : ndarray (3D float64)
    ///     Spatially varying density [kg/m³].  Shape must match the grid.
    /// absorption : ndarray (3D float64), optional
    ///     Spatially varying absorption [dB/(MHz·cm)] (default: zeros).
    /// nonlinearity : ndarray (3D float64), optional
    ///     Spatially varying B/A parameter (default: zeros).
    ///
    /// Returns
    /// -------
    /// Medium
    ///     Heterogeneous acoustic medium
    ///
    /// Examples
    /// --------
    /// >>> c = np.ones((32, 32, 32)) * 1500.0
    /// >>> c[16:, :, :] = 2000.0
    /// >>> rho = np.ones((32, 32, 32)) * 1000.0
    /// >>> medium = Medium(sound_speed=c, density=rho)
    #[new]
    #[pyo3(signature = (sound_speed, density, absorption=None, alpha_power=None, nonlinearity=None))]
    fn new(
        sound_speed: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        absorption: Option<PyReadonlyArray3<f64>>,
        // Power-law exponent y for absorption: α(f) = α₀·(f/f_ref)^y.
        // Pass a scalar float (broadcast to all voxels) or a 3D array
        // matching the shape of `sound_speed`.  Default: 1.0.
        alpha_power: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
        nonlinearity: Option<PyReadonlyArray3<f64>>,
    ) -> PyResult<Self> {
        let c_arr = sound_speed.as_array().to_owned();
        let rho_arr = density.as_array().to_owned();

        let shape = c_arr.shape().to_vec();
        if shape.len() != 3 {
            return Err(PyValueError::new_err("sound_speed must be a 3D array"));
        }
        if rho_arr.shape() != shape.as_slice() {
            return Err(PyValueError::new_err(
                "density shape must match sound_speed shape",
            ));
        }

        if c_arr.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err(
                "All sound_speed values must be positive",
            ));
        }
        if rho_arr.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err("All density values must be positive"));
        }

        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        let mut het = HeterogeneousMedium::new_acoustic_only(nx, ny, nz, true);
        het.sound_speed = c_arr;
        het.density = rho_arr;

        if let Some(abs) = absorption {
            let abs_arr = abs.as_array().to_owned();
            if abs_arr.shape() != [nx, ny, nz] {
                return Err(PyValueError::new_err(
                    "absorption shape must match sound_speed shape",
                ));
            }
            het.absorption = abs_arr;
        }

        // alpha_power: accept scalar float OR 3D ndarray.
        if let Some(py_ap) = alpha_power {
            use ndarray::Array3 as A3;
            if let Ok(scalar) = py_ap.extract::<f64>() {
                het.alpha_power = A3::from_elem((nx, ny, nz), scalar);
            } else if let Ok(arr) = py_ap.extract::<PyReadonlyArray3<f64>>() {
                let ap_arr = arr.as_array().to_owned();
                if ap_arr.shape() != [nx, ny, nz] {
                    return Err(PyValueError::new_err(
                        "alpha_power shape must match sound_speed shape",
                    ));
                }
                het.alpha_power = ap_arr;
            } else {
                return Err(PyValueError::new_err(
                    "alpha_power must be a float or a 3D ndarray matching sound_speed shape",
                ));
            }
        }

        if let Some(nl) = nonlinearity {
            let nl_arr = nl.as_array().to_owned();
            if nl_arr.shape() != [nx, ny, nz] {
                return Err(PyValueError::new_err(
                    "nonlinearity shape must match sound_speed shape",
                ));
            }
            het.nonlinearity = nl_arr;
        }

        Ok(Medium {
            inner: MediumInner::Heterogeneous(Box::new(het)),
        })
    }

    /// Create a homogeneous medium with uniform properties.
    ///
    /// Parameters
    /// ----------
    /// sound_speed : float
    ///     Sound speed [m/s] (typical: water=1500, tissue=1540, bone=4080)
    /// density : float
    ///     Density [kg/m³] (typical: water=1000, tissue=1060, bone=1850)
    /// absorption : float, optional
    ///     Absorption coefficient [dB/(MHz·cm)] (default: 0.0)
    /// nonlinearity : float, optional
    ///     B/A nonlinearity parameter (default: 0.0, tissue≈6, water≈5)
    /// alpha_power : float, optional
    ///     Power law exponent for absorption (default: 1.0)
    /// grid : Grid, optional
    ///     Grid for material field pre-computation
    ///
    /// Returns
    /// -------
    /// Medium
    ///     Homogeneous acoustic medium
    ///
    /// Examples
    /// --------
    /// >>> # Water at 20°C
    /// >>> medium = Medium.homogeneous(1500.0, 1000.0)
    /// >>> # Soft tissue
    /// >>> medium = Medium.homogeneous(1540.0, 1060.0, absorption=0.5, nonlinearity=6.0)
    #[staticmethod]
    #[pyo3(signature = (sound_speed, density, absorption=0.0, nonlinearity=0.0, alpha_power=1.0, grid=None))]
    fn homogeneous(
        sound_speed: f64,
        density: f64,
        absorption: f64,
        nonlinearity: f64,
        alpha_power: f64,
        grid: Option<&Grid>,
    ) -> PyResult<Self> {
        // Validate inputs
        if sound_speed <= 0.0 {
            return Err(PyValueError::new_err("Sound speed must be positive"));
        }
        if density <= 0.0 {
            return Err(PyValueError::new_err("Density must be positive"));
        }
        if absorption < 0.0 {
            return Err(PyValueError::new_err("Absorption must be non-negative"));
        }

        // Create default grid if not provided
        let default_grid = KwaversGrid::default();
        let grid_ref = grid.map(|g| &g.inner).unwrap_or(&default_grid);

        let mut medium = HomogeneousMedium::new(density, sound_speed, 0.0, 0.0, grid_ref);

        // Wire absorption and nonlinearity if provided
        if absorption > 0.0 || nonlinearity > 0.0 {
            medium
                .set_acoustic_properties(absorption, alpha_power, nonlinearity)
                .map_err(|e| {
                    PyValueError::new_err(format!("Invalid acoustic properties: {}", e))
                })?;
        }

        Ok(Medium {
            inner: MediumInner::Homogeneous(Box::new(medium)),
        })
    }

    /// Create a homogeneous **elastic** medium parameterised by physical wave
    /// speeds.
    ///
    /// This is the natural pykwavers equivalent of k-Wave's
    /// ``medium.sound_speed_compression`` / ``medium.sound_speed_shear``
    /// inputs to ``pstdElastic2D`` / ``pstdElastic3D``. The Lamé parameters
    /// are derived in closed form from the elastic-wave dispersion relations:
    ///
    /// ::
    ///
    ///     μ = ρ · c_s²                     (shear modulus)
    ///     λ = ρ · (c_p² − 2 · c_s²)         (first Lamé parameter)
    ///
    /// Parameters
    /// ----------
    /// c_compression : float
    ///     Compressional (P-wave) speed [m/s]. Must be positive.
    /// c_shear : float
    ///     Shear (S-wave) speed [m/s]. Must be ≥ 0 and satisfy
    ///     ``2·c_shear² ≤ c_compression²`` (thermodynamic stability,
    ///     equivalent to ``ν ≥ 0``).
    /// density : float
    ///     Mass density [kg/m³]. Must be positive.
    /// grid : Grid, optional
    ///     Computational grid; when omitted a default grid is used (the
    ///     elastic medium itself is uniform, so grid only sizes the cached
    ///     property arrays).
    ///
    /// Returns
    /// -------
    /// Medium
    ///     Homogeneous elastic medium with Lamé parameters set from the
    ///     supplied wave speeds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If any input is non-finite, non-positive (where required), or the
    ///     stability bound ``2·c_s² ≤ c_p²`` is violated.
    ///
    /// Examples
    /// --------
    /// >>> # k-Wave example_ewp_layered_medium upper layer (water)
    /// >>> water = pkw.Medium.elastic(1500.0, 0.0, 1000.0)
    /// >>> # k-Wave example_ewp_layered_medium lower layer (bone-like)
    /// >>> bone  = pkw.Medium.elastic(2000.0, 800.0, 1200.0)
    #[staticmethod]
    #[pyo3(signature = (c_compression, c_shear, density, grid=None))]
    fn elastic(
        c_compression: f64,
        c_shear: f64,
        density: f64,
        grid: Option<&Grid>,
    ) -> PyResult<Self> {
        let default_grid = KwaversGrid::default();
        let grid_ref = grid.map(|g| &g.inner).unwrap_or(&default_grid);

        let medium =
            HomogeneousMedium::elastic_homogeneous(density, c_compression, c_shear, grid_ref)
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "Invalid elastic parameters. Requirements: density > 0, \
                 c_compression > 0, c_shear ≥ 0, 2·c_shear² ≤ c_compression². \
                 (Stability bound: ν ≥ 0; recovers fluid medium when c_shear = 0.)",
                    )
                })?;

        Ok(Medium {
            inner: MediumInner::Homogeneous(Box::new(medium)),
        })
    }

    /// Create a heterogeneous elastic medium from per-voxel wave-speed and density arrays.
    ///
    /// Lamé parameters are computed per voxel:
    ///   μ   = ρ · c_s²
    ///   λ   = ρ · (c_p² − 2·c_s²)
    ///
    /// Parameters
    /// ----------
    /// c_compression : ndarray (3D float64)
    ///     P-wave speed [m/s] at every voxel.
    /// c_shear : ndarray (3D float64)
    ///     S-wave speed [m/s] at every voxel; set to 0 for fluid voxels.
    /// density : ndarray (3D float64)
    ///     Density [kg/m³] at every voxel.
    /// reference_frequency : float, optional
    ///     Reference frequency for absorption [Hz] (default 1 MHz).
    ///
    /// Returns
    /// -------
    /// Medium
    ///     Heterogeneous elastic medium with Lamé parameters set from the wave speeds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If array shapes mismatch, any density ≤ 0, any c_compression ≤ 0,
    ///     any c_shear < 0, or stability is violated (2·c_s² > c_p² at any voxel).
    #[staticmethod]
    #[pyo3(signature = (c_compression, c_shear, density, reference_frequency=1.0e6))]
    fn elastic_heterogeneous(
        c_compression: PyReadonlyArray3<f64>,
        c_shear: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        reference_frequency: f64,
    ) -> PyResult<Self> {
        let cp = c_compression.as_array();
        let cs = c_shear.as_array();
        let rho = density.as_array();

        let medium = HeterogeneousFactory::from_elastic_arrays(cp, cs, rho, reference_frequency)
            .map_err(PyValueError::new_err)?;

        Ok(Medium {
            inner: MediumInner::Heterogeneous(Box::new(medium)),
        })
    }

    /// Compressional (P-wave) speed [m/s].
    ///
    /// Computed from the stored Lamé parameters and density via
    /// ``c_p = sqrt((λ + 2μ) / ρ)``. For a fluid medium this collapses to
    /// the acoustic sound speed.
    #[getter]
    fn c_compression(&self) -> f64 {
        let m = self.inner.as_medium();
        // Use the centre voxel; HomogeneousMedium is uniform so any (i,j,k) works.
        // For heterogeneous media this returns the centre value which is a
        // documented limitation of this scalar getter (see also `density`).
        let lambda = match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_lambda_value(),
            MediumInner::Heterogeneous(_) => {
                // Heterogeneous λ access via grid+coords, not exposed here.
                0.0
            }
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
    ///
    /// Computed from the shear modulus and density: ``c_s = sqrt(μ / ρ)``.
    /// Returns 0 for fluid media (μ = 0).
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
    ///
    /// Stored on the homogeneous medium directly. For fluid media this
    /// equals the bulk modulus ``K = ρ · c_p²``.
    #[getter]
    fn lame_lambda(&self) -> f64 {
        match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_lambda_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        }
    }

    /// Second Lamé parameter μ (shear modulus, Pa).
    ///
    /// Zero for fluid media; positive for elastic solids supporting shear waves.
    #[getter]
    fn lame_mu(&self) -> f64 {
        match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_mu_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        }
    }

    /// Sound speed [m/s].
    /// For homogeneous media returns the uniform value.
    /// For heterogeneous media returns the maximum sound speed.
    #[getter]
    fn sound_speed(&self) -> f64 {
        self.inner.as_medium().max_sound_speed()
    }

    /// Density [kg/m³].
    /// For homogeneous media returns the uniform value.
    /// For heterogeneous media returns the density at the origin.
    #[getter]
    fn density(&self) -> f64 {
        self.inner.as_medium().density(0, 0, 0)
    }

    /// Whether the medium is homogeneous.
    #[getter]
    fn is_homogeneous(&self) -> bool {
        matches!(self.inner, MediumInner::Homogeneous(_))
    }

    /// String representation.
    fn __repr__(&self) -> String {
        match &self.inner {
            MediumInner::Homogeneous(h) => {
                format!(
                    "Medium.homogeneous(sound_speed={:.1}, density={:.1})",
                    h.max_sound_speed(),
                    h.density(0, 0, 0)
                )
            }
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

    /// Human-readable string.
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
