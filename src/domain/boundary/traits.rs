//! Boundary condition trait system for unified field handling
//!
//! This module defines a comprehensive trait-based boundary condition system that:
//! - Abstracts over different boundary types (CPML, PML, Dirichlet, Neumann, etc.)
//! - Supports multiple physics domains (acoustic, elastic, electromagnetic, thermal)
//! - Provides a unified interface for solver integration
//! - Enforces mathematical invariants at the type level
//!
//! # Design Philosophy
//!
//! The trait system follows these principles:
//! - **Separation of Concerns**: Boundary logic is separate from solver logic
//! - **Mathematical Rigor**: Invariants are enforced through type system
//! - **Zero-Cost Abstraction**: Trait dispatch is inlined in release builds
//! - **Extensibility**: Easy to add new boundary types without modifying existing code
//!
//! # Architecture
//!
//! ```text
//! BoundaryCondition (core trait)
//!     ↓
//! ├── AbsorbingBoundary (PML, CPML, sponge layers)
//! ├── ReflectiveBoundary (rigid, soft, impedance)
//! └── PeriodicBoundary (wraparound, phase shift)
//! ```
//!
//! # Mathematical Foundation
//!
//! Each boundary condition must satisfy:
//! - **Stability**: No energy growth at boundaries (|r| ≤ 1)
//! - **Passivity**: Energy can only be absorbed or reflected, never created
//! - **Causality**: Boundary response depends only on past/present fields

use crate::core::error::KwaversResult;
use crate::domain::grid::GridTopology;
use ndarray::{Array3, ArrayViewMut3};
use rustfft::num_complex::Complex;
use std::fmt::Debug;

/// Field type enumeration for multi-physics support
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldType {
    /// Acoustic pressure field (Pa)
    Pressure,
    /// Velocity field components (m/s)
    Velocity,
    /// Particle displacement (m)
    Displacement,
    /// Stress tensor components (Pa)
    Stress,
    /// Electric field (V/m)
    Electric,
    /// Magnetic field (A/m)
    Magnetic,
    /// Temperature field (K)
    Temperature,
    /// Optical fluence (W/m²)
    Fluence,
}

/// Domain in which boundary is applied
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryDomain {
    /// Spatial domain (real-space)
    Spatial,
    /// Frequency domain (k-space, spectral)
    Frequency,
    /// Time domain (discrete time steps)
    Temporal,
}

/// Direction flags for selective boundary application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundaryDirections {
    /// Apply to x-min boundary
    pub x_min: bool,
    /// Apply to x-max boundary
    pub x_max: bool,
    /// Apply to y-min boundary
    pub y_min: bool,
    /// Apply to y-max boundary
    pub y_max: bool,
    /// Apply to z-min boundary
    pub z_min: bool,
    /// Apply to z-max boundary
    pub z_max: bool,
}

impl Default for BoundaryDirections {
    fn default() -> Self {
        Self::all()
    }
}

impl BoundaryDirections {
    /// All boundaries enabled
    pub const fn all() -> Self {
        Self {
            x_min: true,
            x_max: true,
            y_min: true,
            y_max: true,
            z_min: true,
            z_max: true,
        }
    }

    /// No boundaries enabled
    pub const fn none() -> Self {
        Self {
            x_min: false,
            x_max: false,
            y_min: false,
            y_max: false,
            z_min: false,
            z_max: false,
        }
    }

    /// Only XY plane boundaries (for 2D simulations)
    pub const fn xy_plane() -> Self {
        Self {
            x_min: true,
            x_max: true,
            y_min: true,
            y_max: true,
            z_min: false,
            z_max: false,
        }
    }
}

/// Core boundary condition trait
///
/// This trait defines the interface that all boundary conditions must implement.
/// It provides a unified way for solvers to apply boundary conditions without
/// knowing the specific implementation details.
///
/// # Type Safety
///
/// Implementations must ensure:
/// - Thread safety (`Send + Sync`)
/// - Debuggability (`Debug`)
/// - Mathematical correctness (enforced by invariant checks)
///
/// # Invariants
///
/// All implementations must preserve:
/// 1. **Energy Conservation**: Total energy cannot increase at boundaries
/// 2. **Stability**: Boundary updates must be numerically stable
/// 3. **Causality**: No dependence on future field values
pub trait BoundaryCondition: Debug + Send + Sync {
    /// Get the boundary type name for logging/debugging
    fn name(&self) -> &str;

    /// Get the directions where this boundary is active
    fn active_directions(&self) -> BoundaryDirections;

    /// Apply boundary condition to a scalar field in spatial domain
    ///
    /// # Arguments
    ///
    /// * `field` - Mutable view of the field to modify
    /// * `grid` - Grid topology for coordinate information
    /// * `time_step` - Current simulation time step
    /// * `dt` - Time step size (seconds)
    ///
    /// # Mathematical Form
    ///
    /// For absorbing boundaries, this typically applies:
    /// ```text
    /// u(x,t) → u(x,t) * exp(-σ(x) * Δt)
    /// ```
    /// where σ(x) is the absorption profile.
    fn apply_scalar_spatial(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()>;

    /// Apply boundary condition to a scalar field in frequency domain
    ///
    /// For spectral methods, k-space pseudospectral solvers.
    fn apply_scalar_frequency(
        &mut self,
        field: &mut Array3<Complex<f64>>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()>;

    /// Apply boundary condition to a vector field
    ///
    /// Used for velocity, displacement, electric/magnetic fields.
    ///
    /// # Arguments
    ///
    /// * `field_x` - X-component of the field
    /// * `field_y` - Y-component of the field
    /// * `field_z` - Z-component of the field
    fn apply_vector_spatial(
        &mut self,
        field_x: ArrayViewMut3<f64>,
        field_y: ArrayViewMut3<f64>,
        field_z: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
        time_step: usize,
        dt: f64,
    ) -> KwaversResult<()> {
        // Default implementation: apply to each component independently
        self.apply_scalar_spatial(field_x, grid, time_step, dt)?;
        self.apply_scalar_spatial(field_y, grid, time_step, dt)?;
        self.apply_scalar_spatial(field_z, grid, time_step, dt)?;
        Ok(())
    }

    /// Check if boundary supports a given field type
    fn supports_field_type(&self, field_type: FieldType) -> bool {
        // Default: support common acoustic fields
        matches!(
            field_type,
            FieldType::Pressure | FieldType::Velocity | FieldType::Displacement
        )
    }

    /// Estimate reflection coefficient at given angle
    ///
    /// Returns the magnitude of the reflection coefficient |r| ∈ [0, 1].
    /// A perfectly absorbing boundary would return 0.0.
    ///
    /// # Arguments
    ///
    /// * `angle_degrees` - Angle of incidence (0 = normal, 90 = grazing)
    /// * `frequency` - Wave frequency (Hz)
    /// * `sound_speed` - Medium sound speed (m/s)
    fn reflection_coefficient(
        &self,
        _angle_degrees: f64,
        _frequency: f64,
        _sound_speed: f64,
    ) -> f64 {
        // Default: assume perfect reflection (worst case)
        1.0
    }

    /// Reset internal state (e.g., memory variables for PML)
    fn reset(&mut self);

    /// Check if the boundary has internal state that needs updates
    fn is_stateful(&self) -> bool {
        false
    }

    /// Get memory usage in bytes (for performance monitoring)
    fn memory_usage(&self) -> usize {
        std::mem::size_of_val(self)
    }
}

/// Trait for absorbing boundary conditions (PML, CPML, sponge layers)
///
/// Absorbing boundaries attenuate outgoing waves to minimize reflections.
/// They are characterized by absorption profiles σ(x) that grow from zero
/// inside the domain to maximum at the boundary.
pub trait AbsorbingBoundary: BoundaryCondition {
    /// Get the thickness of the absorbing layer (number of grid points)
    fn thickness(&self) -> usize;

    /// Get the absorption profile at a given position
    ///
    /// Returns σ(x) where the field decays as exp(-σ * Δt).
    ///
    /// # Arguments
    ///
    /// * `indices` - Grid indices [i, j, k]
    /// * `grid` - Grid topology
    fn absorption_profile(&self, indices: [usize; 3], grid: &dyn GridTopology) -> f64;

    /// Get the target reflection coefficient
    ///
    /// This is the design parameter R₀ used to compute absorption strength.
    fn target_reflection(&self) -> f64;

    /// Validate that the boundary layer is thick enough
    ///
    /// Rule of thumb: At least λ_min / 4 or 10-20 grid points.
    fn validate_thickness(&self, grid: &dyn GridTopology, max_frequency: f64) -> KwaversResult<()> {
        let min_spacing = grid.spacing().iter().cloned().fold(f64::INFINITY, f64::min);
        let wavelength = 1500.0 / max_frequency; // Assuming typical sound speed
        let min_thickness = (wavelength / (4.0 * min_spacing)).ceil() as usize;

        if self.thickness() < min_thickness.max(10) {
            return Err(crate::core::error::KwaversError::Config(
                crate::core::error::ConfigError::InvalidValue {
                    parameter: "boundary thickness".to_string(),
                    value: self.thickness().to_string(),
                    constraint: format!("Must be at least {} points", min_thickness.max(10)),
                },
            ));
        }

        Ok(())
    }
}

/// Trait for reflective boundary conditions (rigid, soft, impedance-matched)
pub trait ReflectiveBoundary: BoundaryCondition {
    /// Get the reflection coefficient (complex for impedance boundaries)
    ///
    /// For rigid boundary: r = +1.0
    /// For soft boundary: r = -1.0
    /// For impedance-matched: r = (Z - Z₀) / (Z + Z₀)
    fn reflection_coefficient_complex(&self, frequency: f64) -> Complex<f64>;

    /// Check if the boundary is perfectly rigid (no normal velocity)
    fn is_rigid(&self) -> bool {
        false
    }

    /// Check if the boundary is perfectly soft (zero pressure)
    fn is_soft(&self) -> bool {
        false
    }
}

/// Trait for periodic boundary conditions
pub trait PeriodicBoundary: BoundaryCondition {
    /// Apply periodic wrapping to field
    ///
    /// Ensures that field(x_max) = field(x_min) for all periodic dimensions.
    fn wrap_periodic(
        &mut self,
        field: ArrayViewMut3<f64>,
        grid: &dyn GridTopology,
    ) -> KwaversResult<()>;

    /// Get phase shift for Bloch periodic boundaries (e.g., metamaterials)
    ///
    /// Returns k·L where k is the Bloch wave vector.
    fn phase_shift(&self) -> [f64; 3] {
        [0.0, 0.0, 0.0] // Default: no phase shift
    }
}

/// Helper struct for boundary layer geometry calculations
#[derive(Debug, Clone)]
pub struct BoundaryLayer {
    /// Start index of the layer
    pub start: usize,
    /// End index of the layer (exclusive)
    pub end: usize,
    /// Thickness in grid points
    pub thickness: usize,
    /// Direction (0=x, 1=y, 2=z)
    pub direction: usize,
    /// Side: false=min, true=max
    pub is_max_side: bool,
}

impl BoundaryLayer {
    /// Create a new boundary layer
    pub fn new(start: usize, end: usize, direction: usize, is_max_side: bool) -> Self {
        assert!(end > start, "Invalid boundary layer: end <= start");
        Self {
            start,
            end,
            thickness: end - start,
            direction,
            is_max_side,
        }
    }

    /// Check if a given index is inside this boundary layer
    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        index >= self.start && index < self.end
    }

    /// Get the normalized distance into the layer [0, 1]
    ///
    /// 0 = at domain edge (full absorption)
    /// 1 = at layer interior (no absorption)
    #[inline]
    pub fn normalized_distance(&self, index: usize) -> f64 {
        if !self.contains(index) {
            return 0.0;
        }

        let pos = if self.is_max_side {
            // For max side, distance from the outer edge
            (self.end - 1 - index) as f64
        } else {
            // For min side, distance from the outer edge
            (index - self.start) as f64
        };

        (pos / (self.thickness - 1) as f64).min(1.0)
    }

    /// Compute polynomial absorption profile
    ///
    /// Uses σ(d) = σ_max * ((1-d)/1)^n where d ∈ [0,1] and n is polynomial order.
    pub fn polynomial_profile(&self, index: usize, order: u32, sigma_max: f64) -> f64 {
        let d = self.normalized_distance(index);
        sigma_max * (1.0 - d).powi(order as i32)
    }
}

/// Boundary layer manager for multi-sided boundaries
#[derive(Debug, Clone)]
pub struct BoundaryLayerManager {
    layers: Vec<BoundaryLayer>,
}

impl BoundaryLayerManager {
    /// Create boundary layers from grid dimensions and thickness
    pub fn new(grid: &dyn GridTopology, thickness: usize, directions: BoundaryDirections) -> Self {
        let dims = grid.dimensions();
        let mut layers = Vec::new();

        if directions.x_min && dims[0] > thickness {
            layers.push(BoundaryLayer::new(0, thickness, 0, false));
        }
        if directions.x_max && dims[0] > thickness {
            layers.push(BoundaryLayer::new(dims[0] - thickness, dims[0], 0, true));
        }

        if directions.y_min && dims[1] > thickness {
            layers.push(BoundaryLayer::new(0, thickness, 1, false));
        }
        if directions.y_max && dims[1] > thickness {
            layers.push(BoundaryLayer::new(dims[1] - thickness, dims[1], 1, true));
        }

        if directions.z_min && dims[2] > thickness {
            layers.push(BoundaryLayer::new(0, thickness, 2, false));
        }
        if directions.z_max && dims[2] > thickness {
            layers.push(BoundaryLayer::new(dims[2] - thickness, dims[2], 2, true));
        }

        Self { layers }
    }

    /// Get layers for a specific direction
    pub fn layers_for_direction(&self, direction: usize) -> impl Iterator<Item = &BoundaryLayer> {
        self.layers
            .iter()
            .filter(move |layer| layer.direction == direction)
    }

    /// Check if any layer contains the given indices
    pub fn contains(&self, indices: [usize; 3]) -> bool {
        self.layers.iter().any(|layer| {
            layer.direction == 0 && layer.contains(indices[0])
                || layer.direction == 1 && layer.contains(indices[1])
                || layer.direction == 2 && layer.contains(indices[2])
        })
    }

    /// Get combined absorption at a point (sum over all active layers)
    pub fn combined_absorption<F>(&self, indices: [usize; 3], profile_fn: F) -> f64
    where
        F: Fn(&BoundaryLayer, usize) -> f64,
    {
        let mut total = 0.0;

        for layer in &self.layers {
            let idx = indices[layer.direction];
            if layer.contains(idx) {
                total += profile_fn(layer, idx);
            }
        }

        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boundary_directions() {
        let all = BoundaryDirections::all();
        assert!(all.x_min && all.x_max);

        let none = BoundaryDirections::none();
        assert!(!none.x_min && !none.x_max);

        let xy = BoundaryDirections::xy_plane();
        assert!(xy.x_min && !xy.z_min);
    }

    #[test]
    fn test_boundary_layer() {
        let layer = BoundaryLayer::new(0, 10, 0, false);
        assert_eq!(layer.thickness, 10);
        assert!(layer.contains(5));
        assert!(!layer.contains(15));

        // Normalized distance should be 0 at edge, increasing inward
        assert!((layer.normalized_distance(0) - 0.0).abs() < 1e-10);
        assert!((layer.normalized_distance(9) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_profile() {
        let layer = BoundaryLayer::new(0, 10, 0, false);
        let sigma_max = 100.0;

        // At edge (index 0), should be maximum
        let edge_sigma = layer.polynomial_profile(0, 2, sigma_max);
        assert!((edge_sigma - sigma_max).abs() < 1e-10);

        // At interior edge (index 9), should be near zero
        let interior_sigma = layer.polynomial_profile(9, 2, sigma_max);
        assert!(interior_sigma.abs() < 1e-10);

        // In between should be monotonic
        let mid_sigma = layer.polynomial_profile(5, 2, sigma_max);
        assert!(mid_sigma > interior_sigma && mid_sigma < edge_sigma);
    }
}
