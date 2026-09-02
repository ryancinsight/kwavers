//! PML geometry and k-space numerics knobs on a `Simulation`.

use super::*;

#[pymethods]
impl Simulation {
    /// Set PML (perfectly matched layer) absorbing boundary thickness.
    ///
    /// Parameters
    /// ----------
    /// size : int
    ///     Number of grid points for PML absorbing boundary on each face.
    ///     Typical values: 10-20 for small grids, 20-40 for large grids.
    ///
    /// Examples
    /// --------
    /// >>> sim.set_pml_size(20)
    fn set_pml_size(&mut self, size: usize) {
        self.pml_size = Some(size);
        if let Some(ref mut cfg) = self.pml_config {
            cfg.size = Some(size);
        }
    }

    /// Set the FDTD k-space correction mode.
    ///
    /// Parameters
    /// ----------
    /// mode : str
    ///     Either `"none"` or `"spectral"`.
    fn set_kspace_correction(&mut self, mode: &str) -> PyResult<()> {
        self.kspace_correction = match mode.to_ascii_lowercase().as_str() {
            "none" => KSpaceCorrectionMode::None,
            "spectral" => KSpaceCorrectionMode::Spectral,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported k-space correction mode: {other}"
                )))
            }
        };
        Ok(())
    }

    /// Get the current FDTD k-space correction mode.
    #[getter]
    fn kspace_correction(&self) -> String {
        match self.kspace_correction {
            KSpaceCorrectionMode::None => "none".to_string(),
            KSpaceCorrectionMode::Spectral => "spectral".to_string(),
        }
    }

    /// Set the PSTD compatibility mode.
    fn set_compatibility_mode(&mut self, mode: &str) -> PyResult<()> {
        self.compatibility_mode = match mode.to_ascii_lowercase().as_str() {
            "optimal" => CompatibilityMode::Optimal,
            "reference" => CompatibilityMode::Reference,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported PSTD compatibility mode: {other}"
                )))
            }
        };
        Ok(())
    }

    /// Get the current PSTD compatibility mode.
    #[getter]
    fn compatibility_mode(&self) -> String {
        match self.compatibility_mode {
            CompatibilityMode::Optimal => "optimal".to_string(),
            CompatibilityMode::Reference => "reference".to_string(),
        }
    }

    /// Get the current PML size, or None if using automatic sizing.
    #[getter]
    fn pml_size(&self) -> Option<usize> {
        self.pml_size
    }

    /// Set per-axis PML absorbing boundary thickness for k-Wave parity.
    fn set_pml_size_xyz(&mut self, x: usize, y: usize, z: usize) {
        self.pml_size_xyz = Some((x, y, z));
        self.pml_size = Some(x.max(y).max(z));
        if let Some(ref mut cfg) = self.pml_config {
            cfg.size_xyz = Some((x, y, z));
            cfg.size = Some(x.max(y).max(z));
        }
    }

    /// Set uniform PML absorption factor (equivalent to k-Wave scalar `pml_alpha`, default 2.0).
    fn set_pml_alpha(&mut self, alpha: f64) {
        self.pml_alpha_xyz = Some((alpha, alpha, alpha));
        if let Some(ref mut cfg) = self.pml_config {
            cfg.alpha_xyz = Some((alpha, alpha, alpha));
        }
    }

    /// Set per-axis PML absorption factors (equivalent to k-Wave vector `pml_alpha`).
    fn set_pml_alpha_xyz(&mut self, ax: f64, ay: f64, az: f64) {
        self.pml_alpha_xyz = Some((ax, ay, az));
        if let Some(ref mut cfg) = self.pml_config {
            cfg.alpha_xyz = Some((ax, ay, az));
        }
    }

    /// Set whether PML is inside the computational domain.
    fn set_pml_inside(&mut self, inside: bool) {
        self.pml_inside = inside;
        if let Some(ref mut cfg) = self.pml_config {
            cfg.inside = inside;
        }
    }

    /// Get the current PML inside setting.
    #[getter]
    fn pml_inside(&self) -> bool {
        self.pml_inside
    }
}
