//! `ConvergentBornSolver` struct definition and construction.

use super::super::BornWorkspace;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::math::fft::{Fft3d, Shape3D};
use ndarray::Array3;
use num_complex::Complex64;

/// Convergent Born Series solver for the Helmholtz equation.
///
/// ## Mathematical Foundation
///
/// CBS iteration: `ψ_{n+1} = ψ_n - G * (k²V ψ_n)`
///
/// where G is the outgoing Green's function satisfying `∇²G + k²G = -δ(r)`.
///
/// ## References
///
/// 1. Stanziola, A., et al. (2025). "Iterative Born Solver for the Acoustic
///    Helmholtz Equation with Heterogeneous Sound Speed and Density"
/// 2. de Hoop, M. V. (1995). "Convergent Born series for acoustic and elastic
///    wave equations"
#[derive(Debug)]
pub struct ConvergentBornSolver {
    /// Solver configuration.
    pub(super) config: super::super::BornConfig,
    /// Computational grid.
    pub(super) grid: Grid,
    /// Green's function in frequency domain (FFT-accelerated).
    pub(super) green_fft: Option<Array3<Complex64>>,
    /// Workspace arrays for iterative computation.
    pub(super) workspace: BornWorkspace,
    /// Incident field ψ₀.
    pub(super) incident_field: Array3<Complex64>,
    /// Current iteration field.
    pub(super) current_field: Array3<Complex64>,
    /// FFT processor for accelerated Green's function application.
    pub(super) fft_processor: Option<Fft3d>,
}

impl ConvergentBornSolver {
    /// Create a new Convergent Born Series solver.
    pub fn new(config: super::super::BornConfig, grid: Grid) -> Self {
        let workspace = BornWorkspace::new(grid.nx, grid.ny, grid.nz);
        let shape = (grid.nx, grid.ny, grid.nz);
        let fft_processor = if config.use_fft_green {
            Some(Fft3d::new(Shape3D {
                nx: grid.nx,
                ny: grid.ny,
                nz: grid.nz,
            }))
        } else {
            None
        };
        Self {
            config,
            grid,
            green_fft: None,
            workspace,
            incident_field: Array3::zeros(shape),
            current_field: Array3::zeros(shape),
            fft_processor,
        }
    }

    /// Precompute FFT-accelerated Green's function.
    pub fn precompute_green_function(&mut self, wavenumber: f64) -> KwaversResult<()> {
        if !self.config.use_fft_green {
            return Ok(());
        }
        let mut green = Array3::<Complex64>::zeros((self.grid.nx, self.grid.ny, self.grid.nz));
        self.compute_green_kspace(&mut green, wavenumber)?;
        self.green_fft = Some(green);
        Ok(())
    }
}
