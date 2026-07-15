use leto::{Array3, ArrayViewMut3};

use crate::traits::{BoundaryCondition, BoundaryDirections, BoundaryFieldType, PeriodicBoundary};
use kwavers_core::error::KwaversResult;
use kwavers_grid::topology::GridTopology;

use super::{PeriodicBoundaryCondition, PeriodicConfig};

impl PeriodicBoundaryCondition {
    /// New.
    /// # Errors
    /// - Propagates any [`kwavers_core::error::KwaversError`] returned by called functions.
    ///
    pub fn new(config: PeriodicConfig) -> KwaversResult<Self> {
        config.validate()?;

        let active_directions = BoundaryDirections {
            x_min: config.periodic_x,
            x_max: config.periodic_x,
            y_min: config.periodic_y,
            y_max: config.periodic_y,
            z_min: config.periodic_z,
            z_max: config.periodic_z,
        };

        Ok(Self {
            config,
            active_directions,
        })
    }

    /// Apply periodic wrapping in x direction.
    ///
    /// Standard: `p[0,j,k] = p[nx-1,j,k]`, Bloch: multiplied by `cos(φ_x)`.
    pub(super) fn wrap_x(&self, mut field: ArrayViewMut3<f64>) {
        if !self.config.periodic_x {
            return;
        }

        let nx = field.shape()[0];
        let phase = self.config.bloch_phase[0];
        let scale = if phase.abs() < 1e-12 {
            1.0
        } else {
            phase.cos()
        };

        for j in 0..field.shape()[1] {
            for k in 0..field.shape()[2] {
                field[[0, j, k]] = field[[nx - 2, j, k]] * scale;
                field[[nx - 1, j, k]] = field[[1, j, k]] * scale;
            }
        }
    }

    pub(super) fn wrap_y(&self, mut field: ArrayViewMut3<f64>) {
        if !self.config.periodic_y {
            return;
        }

        let ny = field.shape()[1];
        let phase = self.config.bloch_phase[1];
        let scale = if phase.abs() < 1e-12 {
            1.0
        } else {
            phase.cos()
        };

        for i in 0..field.shape()[0] {
            for k in 0..field.shape()[2] {
                field[[i, 0, k]] = field[[i, ny - 2, k]] * scale;
                field[[i, ny - 1, k]] = field[[i, 1, k]] * scale;
            }
        }
    }

    pub(super) fn wrap_z(&self, mut field: ArrayViewMut3<f64>) {
        if !self.config.periodic_z {
            return;
        }

        let nz = field.shape()[2];
        let phase = self.config.bloch_phase[2];
        let scale = if phase.abs() < 1e-12 {
            1.0
        } else {
            phase.cos()
        };

        for i in 0..field.shape()[0] {
            for j in 0..field.shape()[1] {
                field[[i, j, 0]] = field[[i, j, nz - 2]] * scale;
                field[[i, j, nz - 1]] = field[[i, j, 1]] * scale;
            }
        }
    }

    #[must_use]
    pub fn is_bloch(&self) -> bool {
        self.config.bloch_phase.iter().any(|&p| p.abs() > 1e-12)
    }

    #[must_use]
    pub fn bloch_phase(&self) -> [f64; 3] {
        self.config.bloch_phase
    }
}

impl BoundaryCondition for PeriodicBoundaryCondition {
    fn name(&self) -> &str {
        if self.is_bloch() {
            "Bloch Periodic Boundary"
        } else {
            "Periodic Boundary"
        }
    }

    fn active_directions(&self) -> BoundaryDirections {
        self.active_directions
    }

    fn apply_scalar_spatial(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        self.wrap_x(field.reborrow());
        self.wrap_y(field.reborrow());
        self.wrap_z(field);
        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut Array3<kwavers_math::fft::Complex64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Periodic boundaries in frequency domain are automatic (FFT assumes periodicity)
        Ok(())
    }

    fn supports_field_type(&self, field_type: BoundaryFieldType) -> bool {
        matches!(
            field_type,
            BoundaryFieldType::Pressure
                | BoundaryFieldType::Velocity
                | BoundaryFieldType::Displacement
                | BoundaryFieldType::Temperature
                | BoundaryFieldType::Electric
                | BoundaryFieldType::Magnetic
        )
    }

    fn reflection_coefficient(
        &self,
        _angle_degrees: f64,
        _frequency: f64,
        _sound_speed: f64,
    ) -> f64 {
        0.0
    }

    fn reset(&mut self) {}

    fn is_stateful(&self) -> bool {
        false
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl PeriodicBoundary for PeriodicBoundaryCondition {
    fn wrap_periodic(
        &mut self,
        field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
    ) -> KwaversResult<()> {
        self.apply_scalar_spatial(field, _grid, 0, 0.0)
    }

    fn phase_shift(&self) -> [f64; 3] {
        self.config.bloch_phase
    }
}
