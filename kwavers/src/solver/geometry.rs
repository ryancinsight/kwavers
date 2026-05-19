//! Solver geometry mode.
//!
//! Both PSTD and FDTD solvers support an optional axisymmetric (cylindrical)
//! coordinate system in addition to the default Cartesian 3-D mode.
//!
//! # Grid convention for `CylindricalAS`
//!
//! The 3-D array layout `(nx, ny, nz)` is mapped as:
//! - `nx` → axial dimension (z in cylindrical notation), spacing `dx`
//! - `ny` = 1 always (no azimuthal dimension)
//! - `nz` → radial dimension (`r`), spacing `dz = dr`, first point at `r = 0`
//!
//! Velocity fields:
//! - `ux` → axial velocity `u_z`
//! - `uz` → radial velocity `u_r`
//! - `uy` unused (zero)
//!
//! Split density components:
//! - `rhox` → axial density split
//! - `rhoz` → radial density split
//! - `rhoy` unused (zero)

use serde::{Deserialize, Serialize};

/// Spatial geometry of the computational domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SolverGeometry {
    /// Standard 3-D Cartesian coordinates (default).
    #[default]
    Cartesian3D,
    /// Axisymmetric (cylindrical) coordinates — WSWA-FFT radial operators.
    ///
    /// The radial boundary condition at `r = 0` is enforced implicitly by
    /// the whole-sample-symmetric / whole-asymmetric (WSWA) 4×-domain
    /// extension and standard FFT, matching `kspaceFirstOrderAS` with the
    /// `RadialSymmetry = 'WSWA-FFT'` default.
    ///
    /// Only supported when `ny = 1`.
    CylindricalAS,
}
