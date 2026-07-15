//! pyo3 thin wrappers around `kwavers_physics::field_surrogate`.
//!
//! Exposes [`FocalKernel`] and [`KernelCube`] to Python so treatment
//! planners can build a kernel cube from `.npz`-loaded numpy arrays
//! and query normalized focal envelopes at arbitrary `(f0, pnp)`
//! within the sweep — all interpolation logic lives in kwavers Rust.

use kwavers_physics::field_surrogate::{
    place_kernel_at_focus as kwavers_place_kernel_at_focus, resample_trilinear,
    FocalKernel as KwaversFocalKernel, KernelCube as KwaversKernelCube,
};
use numpy::{PyArray3, PyReadonlyArray3, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::breast_fwi_bindings::complex_compat::{leto3_to_nd3, nd_to_leto3};

/// A cached focal-pressure kernel from a single PSTD pulse.
///
/// Wraps [`kwavers_physics::field_surrogate::FocalKernel`]. Construct
/// from a numpy `(nx, ny, nz)` float64 array of per-voxel peak
/// rarefactional pressure (positive Pa), plus geometry + source
/// metadata.
#[pyclass(name = "FocalKernel", module = "pykwavers")]
#[derive(Clone)]
pub struct FocalKernel {
    pub(crate) inner: KwaversFocalKernel,
}

#[pymethods]
impl FocalKernel {
    /// Construct from numpy + metadata.
    ///
    /// Parameters
    /// ----------
    /// field : ndarray (3D float64)
    ///     Per-voxel peak rarefactional pressure [Pa, positive].
    /// dx_m : float
    ///     Isotropic grid spacing [m].
    /// focus_idx : tuple[int, int, int]
    ///     Grid index of the focal voxel.
    /// f0 : float
    ///     Source centre frequency [Hz].
    /// pnp_realised : float
    ///     Peak rarefactional pressure realised at the focal voxel [Pa].
    /// source_pa : float
    ///     Source drive pressure at the bowl surface [Pa].
    /// fwhm_lat_m : float
    ///     Penttinen 1976 lateral focal FWHM [m].
    /// fwhm_ax_m : float
    ///     Penttinen 1976 axial focal FWHM [m].
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (field, dx_m, focus_idx, f0, pnp_realised, source_pa,
                         fwhm_lat_m, fwhm_ax_m))]
    fn new(
        field: PyReadonlyArray3<f64>,
        dx_m: f64,
        focus_idx: (usize, usize, usize),
        f0: f64,
        pnp_realised: f64,
        source_pa: f64,
        fwhm_lat_m: f64,
        fwhm_ax_m: f64,
    ) -> PyResult<Self> {
        if dx_m <= 0.0 {
            return Err(PyValueError::new_err("dx_m must be positive"));
        }
        if f0 <= 0.0 {
            return Err(PyValueError::new_err("f0 must be positive"));
        }
        let arr = field.as_array().to_owned();
        let (nx, ny, nz) = arr.dim();
        if focus_idx.0 >= nx || focus_idx.1 >= ny || focus_idx.2 >= nz {
            return Err(PyValueError::new_err(format!(
                "focus_idx {focus_idx:?} out of bounds for shape ({nx}, {ny}, {nz})"
            )));
        }
        Ok(FocalKernel {
            inner: KwaversFocalKernel::new(
                nd_to_leto3(arr),
                dx_m,
                focus_idx,
                f0,
                pnp_realised,
                source_pa,
                fwhm_lat_m,
                fwhm_ax_m,
            ),
        })
    }

    #[getter]
    fn dx_m(&self) -> f64 {
        self.inner.dx_m
    }

    #[getter]
    fn focus_idx(&self) -> (usize, usize, usize) {
        self.inner.focus_idx
    }

    #[getter]
    fn f0(&self) -> f64 {
        self.inner.f0
    }

    #[getter]
    fn pnp_realised(&self) -> f64 {
        self.inner.pnp_realised
    }

    #[getter]
    fn source_pa(&self) -> f64 {
        self.inner.source_pa
    }

    #[getter]
    fn fwhm_lat_m(&self) -> f64 {
        self.inner.fwhm_lat_m
    }

    #[getter]
    fn fwhm_ax_m(&self) -> f64 {
        self.inner.fwhm_ax_m
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize) {
        let [nx, ny, nz] = self.inner.shape();
        (nx, ny, nz)
    }

    /// Peak rarefactional pressure at the focal voxel [Pa].
    fn focal_pressure(&self) -> f64 {
        self.inner.focal_pressure()
    }

    /// Return a copy of the field array as a numpy array.
    fn field<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        leto3_to_nd3(self.inner.field.clone()).to_pyarray(py)
    }

    fn __repr__(&self) -> String {
        let [nx, ny, nz] = self.inner.shape();
        format!(
            "FocalKernel(shape=({nx}, {ny}, {nz}), dx_m={:.3e}, f0={:.2e}, pnp={:.2e})",
            self.inner.dx_m, self.inner.f0, self.inner.pnp_realised
        )
    }
}

/// Bilinear interpolator across a sparse `(f0, pnp)` kernel sweep.
///
/// Wraps [`kwavers_physics::field_surrogate::KernelCube`]. Construct
/// from a list of [`FocalKernel`] whose `(f0, pnp_realised)` pairs
/// form a Cartesian grid; query returns a normalized focal envelope
/// (`env.max() == 1`) on the planner grid.
///
/// Physics
/// -------
/// * `pnp` is degenerate in the linear-water regime — accepted for API
///   symmetry but does not drive shape selection.
/// * `f0` is real: linear blend of the two nearest sweep corners after
///   resampling each to the planner grid, then re-normalize.
/// * `f0` outside the sweep clamps to the nearest corner (no
///   extrapolation).
#[pyclass(name = "KernelCube", module = "pykwavers")]
pub struct KernelCube {
    inner: KwaversKernelCube,
}

#[pymethods]
impl KernelCube {
    /// Construct from a list of `FocalKernel` instances. The `(f0,
    /// pnp_realised)` pairs must form a complete Cartesian grid;
    /// raises `ValueError` if a corner is missing.
    #[new]
    fn new(kernels: Vec<FocalKernel>) -> PyResult<Self> {
        let owned: Vec<KwaversFocalKernel> = kernels.into_iter().map(|k| k.inner).collect();
        let inner = KwaversKernelCube::new(owned)
            .map_err(|e| PyValueError::new_err(format!("KernelCube construction failed: {e}")))?;
        Ok(KernelCube { inner })
    }

    /// Sorted unique `f0` axis values [Hz].
    #[getter]
    fn f0_axis(&self) -> Vec<f64> {
        self.inner.f0_axis().to_vec()
    }

    /// Sorted unique `pnp` axis values [Pa].
    #[getter]
    fn pnp_axis(&self) -> Vec<f64> {
        self.inner.pnp_axis().to_vec()
    }

    /// Number of cached kernels.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Query a normalized focal envelope on the planner grid.
    ///
    /// Parameters
    /// ----------
    /// f0 : float
    ///     Source centre frequency of the query [Hz]. Clamped to the
    ///     sweep bounds — no extrapolation.
    /// pnp : float
    ///     Peak rarefactional pressure of the query [Pa]. Accepted for
    ///     API symmetry but not used to select shape (the linear-water
    ///     regime makes envelope shape amplitude-invariant).
    /// target_shape : tuple[int, int, int]
    ///     Target grid shape (nx, ny, nz).
    /// target_focus_idx : tuple[int, int, int]
    ///     Index of the focal voxel on the target grid.
    /// target_dx_m : float
    ///     Target grid spacing [m].
    ///
    /// Returns
    /// -------
    /// ndarray (3D float64)
    ///     Normalized focal envelope, `env.max() == 1`.
    #[pyo3(signature = (f0, pnp, target_shape, target_focus_idx, target_dx_m))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        f0: f64,
        pnp: f64,
        target_shape: (usize, usize, usize),
        target_focus_idx: (usize, usize, usize),
        target_dx_m: f64,
    ) -> PyResult<Bound<'py, PyArray3<f64>>> {
        if target_dx_m <= 0.0 {
            return Err(PyValueError::new_err("target_dx_m must be positive"));
        }
        if target_focus_idx.0 >= target_shape.0
            || target_focus_idx.1 >= target_shape.1
            || target_focus_idx.2 >= target_shape.2
        {
            return Err(PyValueError::new_err(format!(
                "target_focus_idx {target_focus_idx:?} out of bounds for shape {target_shape:?}"
            )));
        }
        let env = py.detach(|| {
            self.inner
                .query(f0, pnp, target_shape, target_focus_idx, target_dx_m)
        });
        Ok(leto3_to_nd3(env).to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "KernelCube(f0_axis={:?}, pnp_axis={:?}, n_kernels={})",
            self.inner.f0_axis(),
            self.inner.pnp_axis(),
            self.inner.len(),
        )
    }
}

/// Place a `FocalKernel` into a target grid centred on the planner's
/// focal voxel. Returns the per-voxel peak rarefactional pressure
/// (Pa) without normalization — the caller-facing absolute amplitude
/// is preserved.
///
/// Wraps `kwavers_physics::field_surrogate::place_kernel_at_focus`.
#[pyfunction]
#[pyo3(signature = (kernel, target_shape, target_focus_idx, target_dx_m=None))]
fn place_kernel_at_focus<'py>(
    py: Python<'py>,
    kernel: &FocalKernel,
    target_shape: (usize, usize, usize),
    target_focus_idx: (usize, usize, usize),
    target_dx_m: Option<f64>,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    if target_focus_idx.0 >= target_shape.0
        || target_focus_idx.1 >= target_shape.1
        || target_focus_idx.2 >= target_shape.2
    {
        return Err(PyValueError::new_err(format!(
            "target_focus_idx {target_focus_idx:?} out of bounds for shape {target_shape:?}"
        )));
    }
    let placed = py.detach(|| {
        let resampled = match target_dx_m {
            Some(dx) if (dx - kernel.inner.dx_m).abs() > 1e-9 => {
                resample_trilinear(&kernel.inner, dx)
            }
            _ => kernel.inner.clone(),
        };
        kwavers_place_kernel_at_focus(&resampled, target_shape, target_focus_idx)
    });
    Ok(leto3_to_nd3(placed).to_pyarray(py))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FocalKernel>()?;
    m.add_class::<KernelCube>()?;
    m.add_function(wrap_pyfunction!(place_kernel_at_focus, m)?)?;
    Ok(())
}
