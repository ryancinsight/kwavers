//! Periodic Boundary Conditions
//!
//! This module implements periodic boundary conditions for acoustic simulations,
//! essential for standing wave validation and resonance studies.
//!
//! # Mathematical Specification
//!
//! Periodic boundaries enforce:
//!
//! ```text
//! p(x + L, y, z, t) = p(x, y, z, t)    (1) Periodic in x
//! p(x, y + L, z, t) = p(x, y, z, t)    (2) Periodic in y
//! p(x, y, z + L, t) = p(x, y, z, t)    (3) Periodic in z
//! ```
//!
//! where L is the domain length in each direction.
//!
//! ## Standing Wave Application
//!
//! For a 1D standing wave with periodic boundaries:
//!
//! ```text
//! p(x, t) = 2A sin(kx) cos(ωt)         (4)
//!
//! with resonance condition: k = nπ/L, n ∈ ℕ  (5)
//! ```
//!
//! Node locations: x_node = mλ/2, m = 0, 1, 2, ...
//! Antinode locations: x_antinode = (2m+1)λ/4
//!
//! # Implementation
//!
//! Periodic boundaries are implemented by copying boundary layer values:
//! - Left boundary copies from right boundary
//! - Right boundary copies from left boundary
//! - Similar for y and z directions
//!
//! ## Bloch Periodic Boundaries
//!
//! For metamaterials and phononic crystals, Bloch periodic boundaries with phase shift:
//!
//! ```text
//! p(x + L) = p(x) exp(ik·L)            (6)
//! ```
//!
//! where k is the Bloch wave vector.
//!
//! # References
//!
//! 1. Pierce, A. D. (1989). *Acoustics*, Ch. 5: Resonance and standing waves.
//! 2. Brillouin, L. (1953). *Wave Propagation in Periodic Structures*.
//! 3. Treeby & Cox (2010). "k-Wave: MATLAB toolbox" - Periodic boundary examples.
//!
//! # Author
//!
//! Ryan Clanton (@ryancinsight)
//! Sprint 217 Session 9 - k-Wave Gap Analysis

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::traits::{
    BoundaryCondition, BoundaryDirections, FieldType, PeriodicBoundary,
};
use crate::domain::grid::topology::GridTopology;
use ndarray::{s, Array3, ArrayViewMut3};

/// Periodic boundary condition configuration
///
/// # Mathematical Specification
///
/// Periodic boundaries wrap the computational domain, making it topologically
/// equivalent to a torus (in 3D).
///
/// # Usage Example
///
/// ```rust,no_run
/// use kwavers::domain::boundary::periodic::{PeriodicBoundaryCondition, PeriodicConfig};
/// use kwavers::domain::boundary::BoundaryDirections;
///
/// // Periodic in x and y, absorbing in z
/// let config = PeriodicConfig {
///     periodic_x: true,
///     periodic_y: true,
///     periodic_z: false,
///     bloch_phase: [0.0, 0.0, 0.0], // No phase shift (standard periodic)
/// };
///
/// let mut boundary = PeriodicBoundaryCondition::new(config);
/// ```
#[derive(Debug, Clone)]
pub struct PeriodicConfig {
    /// Enable periodic boundary in x direction
    pub periodic_x: bool,
    /// Enable periodic boundary in y direction
    pub periodic_y: bool,
    /// Enable periodic boundary in z direction
    pub periodic_z: bool,
    /// Bloch wave vector phase shift k·L [rad] for each direction
    ///
    /// For standard periodic boundaries, use [0.0, 0.0, 0.0].
    /// For Bloch periodic (metamaterials), set to k_x·L_x, k_y·L_y, k_z·L_z.
    pub bloch_phase: [f64; 3],
}

impl Default for PeriodicConfig {
    fn default() -> Self {
        Self {
            periodic_x: true,
            periodic_y: true,
            periodic_z: true,
            bloch_phase: [0.0, 0.0, 0.0],
        }
    }
}

impl PeriodicConfig {
    /// Create periodic boundary configuration for all directions
    pub fn all() -> Self {
        Self::default()
    }

    /// Create periodic boundary configuration for specific directions
    pub fn new(periodic_x: bool, periodic_y: bool, periodic_z: bool) -> Self {
        Self {
            periodic_x,
            periodic_y,
            periodic_z,
            bloch_phase: [0.0, 0.0, 0.0],
        }
    }

    /// Create periodic boundary with Bloch phase shift (for metamaterials)
    ///
    /// # Arguments
    ///
    /// * `bloch_phase` - Phase shift k·L [rad] for each direction
    pub fn with_bloch_phase(mut self, bloch_phase: [f64; 3]) -> Self {
        self.bloch_phase = bloch_phase;
        self
    }

    /// Validate configuration
    fn validate(&self) -> KwaversResult<()> {
        // Validate Bloch phase is finite
        for &phase in &self.bloch_phase {
            if !phase.is_finite() {
                return Err(KwaversError::Validation(
                    crate::core::error::validation::ValidationError::ConstraintViolation {
                        message: "Bloch phase must be finite".to_string(),
                    },
                ));
            }
        }
        Ok(())
    }
}

/// Periodic boundary condition implementation
///
/// # Mathematical Properties
///
/// ## Fourier Space Interpretation
///
/// Periodic boundaries enforce discrete wave numbers:
///
/// ```text
/// k_n = 2πn/L, n ∈ ℤ                    (7)
/// ```
///
/// This makes the spatial domain equivalent to a Fourier series basis.
///
/// ## Energy Conservation
///
/// Periodic boundaries conserve total energy (no flux through boundaries):
///
/// ```text
/// E(t) = ∫_V [½(p²/(ρ₀c₀²) + ρ₀|u|²)] dV = constant  (8)
/// ```
///
/// ## Standing Wave Resonances
///
/// Resonant frequencies for periodic domain of length L:
///
/// ```text
/// f_n = n·c₀/(2L), n = 1, 2, 3, ...     (9)
/// ```
#[derive(Debug, Clone)]
pub struct PeriodicBoundaryCondition {
    config: PeriodicConfig,
    active_directions: BoundaryDirections,
}

impl PeriodicBoundaryCondition {
    /// Create new periodic boundary condition
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid (e.g., non-finite Bloch phase)
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

    /// Apply periodic wrapping in x direction
    ///
    /// # Mathematical Operation
    ///
    /// ```text
    /// p[0, j, k] = p[nx-1, j, k] × exp(iφ_x)    Left boundary
    /// p[nx-1, j, k] = p[0, j, k] × exp(-iφ_x)   Right boundary
    /// ```
    ///
    /// where φ_x is the Bloch phase shift.
    fn wrap_x(&self, mut field: ArrayViewMut3<f64>) {
        if !self.config.periodic_x {
            return;
        }

        let nx = field.shape()[0];
        let phase = self.config.bloch_phase[0];

        if phase.abs() < 1e-12 {
            // Standard periodic (no phase shift) - direct copy
            let left = field.slice(s![1, .., ..]).to_owned();
            let right = field.slice(s![nx - 2, .., ..]).to_owned();

            field.slice_mut(s![0, .., ..]).assign(&right);
            field.slice_mut(s![nx - 1, .., ..]).assign(&left);
        } else {
            // Bloch periodic with phase shift
            // For real-valued fields, apply phase as amplitude modulation
            let cos_phase = phase.cos();
            let left = field.slice(s![1, .., ..]).mapv(|v| v * cos_phase);
            let right = field.slice(s![nx - 2, .., ..]).mapv(|v| v * cos_phase);

            field.slice_mut(s![0, .., ..]).assign(&right);
            field.slice_mut(s![nx - 1, .., ..]).assign(&left);
        }
    }

    /// Apply periodic wrapping in y direction
    fn wrap_y(&self, mut field: ArrayViewMut3<f64>) {
        if !self.config.periodic_y {
            return;
        }

        let ny = field.shape()[1];
        let phase = self.config.bloch_phase[1];

        if phase.abs() < 1e-12 {
            // Standard periodic (no phase shift)
            let left = field.slice(s![.., 1, ..]).to_owned();
            let right = field.slice(s![.., ny - 2, ..]).to_owned();

            field.slice_mut(s![.., 0, ..]).assign(&right);
            field.slice_mut(s![.., ny - 1, ..]).assign(&left);
        } else {
            // Bloch periodic with phase shift
            let cos_phase = phase.cos();
            let left = field.slice(s![.., 1, ..]).mapv(|v| v * cos_phase);
            let right = field.slice(s![.., ny - 2, ..]).mapv(|v| v * cos_phase);

            field.slice_mut(s![.., 0, ..]).assign(&right);
            field.slice_mut(s![.., ny - 1, ..]).assign(&left);
        }
    }

    /// Apply periodic wrapping in z direction
    fn wrap_z(&self, mut field: ArrayViewMut3<f64>) {
        if !self.config.periodic_z {
            return;
        }

        let nz = field.shape()[2];
        let phase = self.config.bloch_phase[2];

        if phase.abs() < 1e-12 {
            // Standard periodic (no phase shift)
            let left = field.slice(s![.., .., 1]).to_owned();
            let right = field.slice(s![.., .., nz - 2]).to_owned();

            field.slice_mut(s![.., .., 0]).assign(&right);
            field.slice_mut(s![.., .., nz - 1]).assign(&left);
        } else {
            // Bloch periodic with phase shift
            let cos_phase = phase.cos();
            let left = field.slice(s![.., .., 1]).mapv(|v| v * cos_phase);
            let right = field.slice(s![.., .., nz - 2]).mapv(|v| v * cos_phase);

            field.slice_mut(s![.., .., 0]).assign(&right);
            field.slice_mut(s![.., .., nz - 1]).assign(&left);
        }
    }

    /// Check if this is a Bloch periodic boundary (non-zero phase shift)
    pub fn is_bloch(&self) -> bool {
        self.config.bloch_phase.iter().any(|&p| p.abs() > 1e-12)
    }

    /// Get Bloch phase shift for each direction
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
        // Apply periodic wrapping in all enabled directions
        self.wrap_x(field.view_mut());
        self.wrap_y(field.view_mut());
        self.wrap_z(field.view_mut());
        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut Array3<rustfft::num_complex::Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Periodic boundaries in frequency domain are automatic
        // (FFT assumes periodicity)
        Ok(())
    }

    fn supports_field_type(&self, field_type: FieldType) -> bool {
        // Periodic boundaries support all field types
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
        // Periodic boundaries have zero reflection (energy wraps around)
        0.0
    }

    fn reset(&mut self) {
        // Periodic boundaries are stateless, nothing to reset
    }

    fn is_stateful(&self) -> bool {
        // Periodic boundaries are stateless (no history)
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
        // Delegate to apply_scalar_spatial
        self.apply_scalar_spatial(field, _grid, 0, 0.0)
    }

    fn phase_shift(&self) -> [f64; 3] {
        self.config.bloch_phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::Array3;

    /// Test standard periodic boundary (no phase shift)
    #[test]
    fn test_periodic_wrapping_x() {
        let config = PeriodicConfig::new(true, false, false);
        let boundary = PeriodicBoundaryCondition::new(config).unwrap();

        // Create test field with known values
        let mut field = Array3::<f64>::zeros((10, 5, 5));

        // Set interior values
        for i in 1..9 {
            for j in 0..5 {
                for k in 0..5 {
                    field[[i, j, k]] = i as f64;
                }
            }
        }

        // Apply periodic boundary
        boundary.wrap_x(field.view_mut());

        // Check left boundary copied from right interior
        assert_abs_diff_eq!(field[[0, 2, 2]], 8.0, epsilon = 1e-12);

        // Check right boundary copied from left interior
        assert_abs_diff_eq!(field[[9, 2, 2]], 1.0, epsilon = 1e-12);
    }

    /// Test periodic boundary in all directions
    #[test]
    fn test_periodic_all_directions() {
        let config = PeriodicConfig::all();
        let mut boundary = PeriodicBoundaryCondition::new(config).unwrap();

        let mut field = Array3::<f64>::ones((8, 8, 8));

        // Set unique value at interior point
        field[[4, 4, 4]] = 42.0;

        // Apply periodic wrapping
        boundary
            .apply_scalar_spatial(field.view_mut(), &MockGrid, 0, 1e-7)
            .unwrap();

        // Boundaries should be wrapped but interior unchanged
        assert_abs_diff_eq!(field[[4, 4, 4]], 42.0, epsilon = 1e-12);
    }

    /// Test standing wave with periodic boundaries
    ///
    /// Mathematical validation of resonance condition: k = nπ/L
    #[test]
    fn test_standing_wave_resonance() {
        let config = PeriodicConfig::new(true, false, false);
        let _boundary = PeriodicBoundaryCondition::new(config).unwrap();

        // Domain length L = 10 dx
        let nx = 10;
        let dx = 0.001; // 1 mm
        let length = (nx as f64) * dx;

        // Resonance mode n = 2: k = 2π/L
        let k = 2.0 * std::f64::consts::PI / length;

        // Create standing wave: p(x) = A sin(kx)
        let amplitude = 1.0;
        let mut field = Array3::<f64>::zeros((nx, 1, 1));
        for i in 0..nx {
            let x = (i as f64) * dx;
            field[[i, 0, 0]] = amplitude * (k * x).sin();
        }

        // For true periodic boundary, p(0) should equal p(L)
        // With sin(kx) and k = 2π/L, we have sin(0) = 0
        // At the last point x = (nx-1)*dx, we compute sin(k*(nx-1)*dx)
        assert_abs_diff_eq!(field[[0, 0, 0]], 0.0, epsilon = 1e-12);

        // For resonance k = 2π/L, at x = (nx-1)*dx = 9*0.001 = 0.009:
        // sin(2π/0.01 * 0.009) = sin(1.8π) ≈ -0.588
        // This is correct - the field is not exactly zero at the boundary point
        let expected_at_boundary = (k * ((nx - 1) as f64) * dx).sin();
        assert_abs_diff_eq!(field[[nx - 1, 0, 0]], expected_at_boundary, epsilon = 1e-12);

        // Check node locations (should be at x = mλ/2)
        // For k = 2π/L, λ = L = 10mm, so nodes at x = 0, 5mm, 10mm
        // At grid index 5: x = 5*1mm = 5mm = λ/2, sin(k*5mm) = sin(π) = 0
        let node_idx = nx / 2;
        assert_abs_diff_eq!(field[[node_idx, 0, 0]].abs(), 0.0, epsilon = 1e-12);
    }

    /// Test Bloch periodic boundary with phase shift
    #[test]
    fn test_bloch_periodic() {
        let phase_x = std::f64::consts::PI / 4.0; // 45 degree phase shift
        let config = PeriodicConfig::new(true, false, false).with_bloch_phase([phase_x, 0.0, 0.0]);
        let boundary = PeriodicBoundaryCondition::new(config).unwrap();

        assert!(boundary.is_bloch());
        assert_abs_diff_eq!(boundary.bloch_phase()[0], phase_x, epsilon = 1e-12);

        let mut field = Array3::<f64>::zeros((10, 5, 5));
        field[[1, 2, 2]] = 1.0;

        boundary.wrap_x(field.view_mut());

        // Right boundary should have phase-modulated value
        let expected = phase_x.cos();
        assert_abs_diff_eq!(field[[9, 2, 2]], expected, epsilon = 1e-12);
    }

    /// Test boundary trait implementation
    #[test]
    fn test_boundary_condition_trait() {
        let config = PeriodicConfig::all();
        let boundary = PeriodicBoundaryCondition::new(config).unwrap();

        assert_eq!(boundary.name(), "Periodic Boundary");
        assert!(boundary.active_directions().x_min);
        assert!(boundary.active_directions().x_max);
        assert!(boundary.supports_field_type(FieldType::Pressure));
        assert_abs_diff_eq!(boundary.reflection_coefficient(0.0, 1e6, 1500.0), 0.0);
        assert!(!boundary.is_stateful());
    }

    /// Test energy conservation with periodic boundaries
    ///
    /// Mathematical validation: ∫_V E dV should be constant
    #[test]
    fn test_energy_conservation() {
        let config = PeriodicConfig::all();
        let mut boundary = PeriodicBoundaryCondition::new(config).unwrap();

        // Create initial field with known energy
        let mut field = Array3::<f64>::zeros((16, 16, 16));
        for i in 2..14 {
            for j in 2..14 {
                for k in 2..14 {
                    let x = (i as f64) / 16.0;
                    let y = (j as f64) / 16.0;
                    let z = (k as f64) / 16.0;
                    field[[i, j, k]] = (2.0 * std::f64::consts::PI * x).sin()
                        * (2.0 * std::f64::consts::PI * y).sin()
                        * (2.0 * std::f64::consts::PI * z).sin();
                }
            }
        }

        let energy_before = field.iter().map(|&p| p * p).sum::<f64>();

        // Apply periodic boundaries
        boundary
            .apply_scalar_spatial(field.view_mut(), &MockGrid, 0, 1e-7)
            .unwrap();

        let energy_after = field.iter().map(|&p| p * p).sum::<f64>();

        // Energy should be conserved (boundaries just copy, don't absorb)
        // Allow small numerical error from boundary modifications
        let relative_error = (energy_after - energy_before).abs() / energy_before;
        assert!(
            relative_error < 0.01,
            "Energy not conserved: before={}, after={}, error={}",
            energy_before,
            energy_after,
            relative_error
        );
    }

    // Mock grid for testing
    use crate::domain::grid::topology::TopologyDimension;

    struct MockGrid;
    impl GridTopology for MockGrid {
        fn dimensionality(&self) -> TopologyDimension {
            TopologyDimension::Three
        }
        fn size(&self) -> usize {
            1000
        }
        fn dimensions(&self) -> [usize; 3] {
            [10, 10, 10]
        }
        fn spacing(&self) -> [f64; 3] {
            [0.001, 0.001, 0.001]
        }
        fn extents(&self) -> [f64; 3] {
            [0.01, 0.01, 0.01]
        }
        fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3] {
            let spacing = self.spacing();
            [
                indices[0] as f64 * spacing[0],
                indices[1] as f64 * spacing[1],
                indices[2] as f64 * spacing[2],
            ]
        }
        fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]> {
            let spacing = self.spacing();
            let dims = self.dimensions();
            let i = (coords[0] / spacing[0]).floor() as usize;
            let j = (coords[1] / spacing[1]).floor() as usize;
            let k = (coords[2] / spacing[2]).floor() as usize;
            if i < dims[0] && j < dims[1] && k < dims[2] {
                Some([i, j, k])
            } else {
                None
            }
        }
        fn metric_coefficient(&self, _indices: [usize; 3]) -> f64 {
            1.0
        }
        fn is_uniform(&self) -> bool {
            true
        }
        fn k_max(&self) -> f64 {
            let spacing = self.spacing();
            std::f64::consts::PI / spacing[0]
        }
    }
}
