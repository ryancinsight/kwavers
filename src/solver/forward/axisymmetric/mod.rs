//! Axisymmetric k-space pseudospectral solver
//!
//! Implements `kspaceFirstOrderAS` equivalent for efficient simulation of
//! axially symmetric acoustic wave propagation problems.
//!
//! # Overview
//!
//! The axisymmetric solver reduces 3D cylindrically symmetric problems to 2D
//! by exploiting rotational symmetry about the z-axis. This provides significant
//! computational savings for problems involving:
//!
//! - Circular/spherical transducers
//! - HIFU focused ultrasound
//! - Cylindrical phantoms
//! - Axially symmetric tissue geometries
//!
//! # Coordinate System
//!
//! The solver uses cylindrical coordinates (r, z):
//! - **Axial (z)**: Propagation direction (mapped to x-axis in 2D grid)
//! - **Radial (r)**: Distance from symmetry axis (mapped to y-axis in 2D grid)
//!
//! # Mathematical Background
//!
//! The axisymmetric wave equation in cylindrical coordinates:
//!
//! $$
//! \frac{1}{c^2}\frac{\partial^2 p}{\partial t^2} =
//! \frac{\partial^2 p}{\partial z^2} +
//! \frac{1}{r}\frac{\partial}{\partial r}\left(r\frac{\partial p}{\partial r}\right)
//! $$
//!
//! The radial derivative term is handled using discrete Hankel transforms (DHT)
//! which are the cylindrical analog of the Fourier transform.
//!
//! # Grid Topology
//!
//! This module uses the unified `CylindricalTopology` from `domain::grid::topology`
//! for consistent grid handling across the codebase.
//!
//! # References
//!
//! - Treeby, B. E., & Cox, B. T. (2010). "MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2), 021314.
//! - `kspaceFirstOrderAS` documentation

mod config;
mod solver;
mod transforms;

pub use config::AxisymmetricConfig;
#[allow(deprecated)]
pub use config::AxisymmetricMedium;

// Re-export the unified topology from domain
pub use crate::domain::grid::CylindricalTopology;

pub use solver::AxisymmetricSolver;
pub use transforms::DiscreteHankelTransform;
