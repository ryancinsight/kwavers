use ndarray::{s, Array3, ArrayViewMut3};

use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::{
    BoundaryCondition, BoundaryDirections, FieldType, PeriodicBoundary,
};
use crate::domain::grid::topology::GridTopology;

use super::{PeriodicBoundaryCondition, PeriodicConfig};

impl PeriodicBoundaryCondition {
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

        if phase.abs() < 1e-12 {
            let left = field.slice(s![1, .., ..]).to_owned();
            let right = field.slice(s![nx - 2, .., ..]).to_owned();
            field.slice_mut(s![0, .., ..]).assign(&right);
            field.slice_mut(s![nx - 1, .., ..]).assign(&left);
        } else {
            let cos_phase = phase.cos();
            let left = field.slice(s![1, .., ..]).mapv(|v| v * cos_phase);
            let right = field.slice(s![nx - 2, .., ..]).mapv(|v| v * cos_phase);
            field.slice_mut(s![0, .., ..]).assign(&right);
            field.slice_mut(s![nx - 1, .., ..]).assign(&left);
        }
    }

    pub(super) fn wrap_y(&self, mut field: ArrayViewMut3<f64>) {
        if !self.config.periodic_y {
            return;
        }

        let ny = field.shape()[1];
        let phase = self.config.bloch_phase[1];

        if phase.abs() < 1e-12 {
            let left = field.slice(s![.., 1, ..]).to_owned();
            let right = field.slice(s![.., ny - 2, ..]).to_owned();
            field.slice_mut(s![.., 0, ..]).assign(&right);
            field.slice_mut(s![.., ny - 1, ..]).assign(&left);
        } else {
            let cos_phase = phase.cos();
            let left = field.slice(s![.., 1, ..]).mapv(|v| v * cos_phase);
            let right = field.slice(s![.., ny - 2, ..]).mapv(|v| v * cos_phase);
            field.slice_mut(s![.., 0, ..]).assign(&right);
            field.slice_mut(s![.., ny - 1, ..]).assign(&left);
        }
    }

    pub(super) fn wrap_z(&self, mut field: ArrayViewMut3<f64>) {
        if !self.config.periodic_z {
            return;
        }

        let nz = field.shape()[2];
        let phase = self.config.bloch_phase[2];

        if phase.abs() < 1e-12 {
            let left = field.slice(s![.., .., 1]).to_owned();
            let right = field.slice(s![.., .., nz - 2]).to_owned();
            field.slice_mut(s![.., .., 0]).assign(&right);
            field.slice_mut(s![.., .., nz - 1]).assign(&left);
        } else {
            let cos_phase = phase.cos();
            let left = field.slice(s![.., .., 1]).mapv(|v| v * cos_phase);
            let right = field.slice(s![.., .., nz - 2]).mapv(|v| v * cos_phase);
            field.slice_mut(s![.., .., 0]).assign(&right);
            field.slice_mut(s![.., .., nz - 1]).assign(&left);
        }
    }

    pub fn is_bloch(&self) -> bool {
        self.config.bloch_phase.iter().any(|&p| p.abs() > 1e-12)
    }

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
        self.wrap_x(field.view_mut());
        self.wrap_y(field.view_mut());
        self.wrap_z(field.view_mut());
        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut Array3<crate::math::fft::Complex64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Periodic boundaries in frequency domain are automatic (FFT assumes periodicity)
        Ok(())
    }

    fn supports_field_type(&self, field_type: FieldType) -> bool {
        matches!(
            field_type,
            FieldType::Pressure
                | FieldType::Velocity
                | FieldType::Displacement
                | FieldType::Temperature
                | FieldType::Electric
                | FieldType::Magnetic
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
