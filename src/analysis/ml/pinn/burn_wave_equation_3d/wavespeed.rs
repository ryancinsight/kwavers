//! Wave speed function wrapper with Burn Module trait support
//!
//! This module provides a wrapper for wave speed functions c(x, y, z) that integrates
//! with Burn's Module system. It supports both CPU-based closures and device-resident
//! grids for efficient heterogeneous media modeling.
//!
//! ## Wave Speed Representations
//!
//! - **CPU Closure**: Function `c(x, y, z) → f32` evaluated on-demand
//! - **Device Grid**: Pre-computed tensor of wave speeds for fast lookup
//!
//! ## Module Integration
//!
//! Implements `Module`, `AutodiffModule`, and `ModuleDisplay` traits to enable:
//! - Device migration (CPU ↔ GPU)
//! - Automatic differentiation compatibility
//! - Module hierarchy inspection

use burn::module::{AutodiffModule, Devices, Module};
use burn::tensor::{backend::AutodiffBackend, backend::Backend, Tensor};
use std::sync::Arc;

/// Wave speed function wrapper supporting CPU closures and device grids
///
/// This struct wraps a wave speed function c(x, y, z) and integrates it with
/// Burn's Module system for device management and autodiff compatibility.
///
/// # Type Parameters
///
/// * `B` - Backend type (e.g., NdArray, WGPU)
///
/// # Fields
///
/// * `func` - CPU closure for wave speed evaluation
/// * `grid` - Optional device-resident tensor for fast lookup
#[derive(Module, Clone)]
pub struct WaveSpeedFn3D<B: Backend> {
    /// CPU function: c(x, y, z) → wave_speed
    pub func: Arc<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>,
    /// Optional device-resident grid of wave speeds [nx, ny, nz]
    pub grid: Option<Tensor<B, 3>>,
}

impl<B: Backend> WaveSpeedFn3D<B> {
    /// Create a new wave speed function from a CPU closure
    ///
    /// # Arguments
    ///
    /// * `func` - Function returning wave speed at coordinates (x, y, z)
    ///
    /// # Returns
    ///
    /// A new `WaveSpeedFn3D` instance wrapping the closure
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use std::sync::Arc;
    /// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::WaveSpeedFn3D;
    ///
    /// // Constant wave speed
    /// let wave_speed = WaveSpeedFn3D::new(Arc::new(|_x, _y, _z| 1500.0));
    ///
    /// // Layered medium
    /// let layered = WaveSpeedFn3D::new(Arc::new(|_x, _y, z| {
    ///     if z < 0.5 { 1500.0 } else { 3000.0 }
    /// }));
    /// ```
    pub fn new(func: Arc<dyn Fn(f32, f32, f32) -> f32 + Send + Sync>) -> Self {
        Self { func, grid: None }
    }

    /// Create a new wave speed function from a device-resident grid
    ///
    /// # Arguments
    ///
    /// * `grid` - Tensor of wave speeds [nx, ny, nz]
    ///
    /// # Returns
    ///
    /// A new `WaveSpeedFn3D` instance with device-resident grid
    ///
    /// # Notes
    ///
    /// The CPU function will return 0.0 when using grid-based lookup.
    /// Grid indexing and interpolation should be implemented in the solver.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use burn::tensor::Tensor;
    /// use kwavers::analysis::ml::pinn::burn_wave_equation_3d::WaveSpeedFn3D;
    ///
    /// let grid = Tensor::<Backend, 3>::ones([64, 64, 64], &device).mul_scalar(1500.0);
    /// let wave_speed = WaveSpeedFn3D::from_grid(grid);
    /// ```
    pub fn from_grid(grid: Tensor<B, 3>) -> Self {
        Self {
            func: Arc::new(|_, _, _| 0.0), // Dummy function for grid-based mode
            grid: Some(grid),
        }
    }

    /// Evaluate wave speed at coordinates (x, y, z)
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinate (meters)
    /// * `y` - Y-coordinate (meters)
    /// * `z` - Z-coordinate (meters)
    ///
    /// # Returns
    ///
    /// Wave speed c(x, y, z) in m/s
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let speed = wave_speed_fn.get(0.5, 0.5, 0.5);
    /// assert_eq!(speed, 1500.0);
    /// ```
    pub fn get(&self, x: f32, y: f32, z: f32) -> f32 {
        (self.func)(x, y, z)
    }
}

impl<B: Backend> std::fmt::Debug for WaveSpeedFn3D<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WaveSpeedFn3D")
            .field("has_grid", &self.grid.is_some())
            .finish()
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for WaveSpeedFn3D<B> {
    type InnerModule = WaveSpeedFn3D<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        WaveSpeedFn3D {
            func: self.func.clone(),
            grid: self.grid.as_ref().map(|g| g.clone().inner()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_wavespeed_creation_closure() {
        let wave_speed = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));

        assert!(wave_speed.grid.is_none());
        assert_eq!(wave_speed.get(0.0, 0.0, 0.0), 1500.0);
    }

    #[test]
    fn test_wavespeed_creation_grid() {
        let device = Default::default();
        let grid = Tensor::<TestBackend, 3>::ones([10, 10, 10], &device).mul_scalar(3000.0);

        let wave_speed = WaveSpeedFn3D::<TestBackend>::from_grid(grid);

        assert!(wave_speed.grid.is_some());
    }

    #[test]
    fn test_wavespeed_evaluation() {
        // Constant speed
        let constant = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
        assert_eq!(constant.get(0.5, 0.5, 0.5), 1500.0);
        assert_eq!(constant.get(1.0, 1.0, 1.0), 1500.0);

        // Spatially varying (layered)
        let layered = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, z| {
            if z < 0.5 {
                1500.0 // Water layer
            } else {
                3000.0 // Tissue layer
            }
        }));
        assert_eq!(layered.get(0.5, 0.5, 0.3), 1500.0);
        assert_eq!(layered.get(0.5, 0.5, 0.7), 3000.0);
    }

    #[test]
    fn test_wavespeed_radial_variation() {
        // Radially varying speed (spherical inclusion)
        let radial = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|x, y, z| {
            let r = (x * x + y * y + z * z).sqrt();
            if r < 0.5 {
                2000.0 // Inclusion
            } else {
                1500.0 // Background
            }
        }));

        assert_eq!(radial.get(0.0, 0.0, 0.0), 2000.0); // Center
        assert_eq!(radial.get(1.0, 0.0, 0.0), 1500.0); // Outside
    }

    #[test]
    fn test_wavespeed_device_migration() {
        let device1 = Default::default();
        let grid1 = Tensor::<TestBackend, 3>::ones([5, 5, 5], &device1);
        let wave_speed1 = WaveSpeedFn3D::<TestBackend>::from_grid(grid1);

        // Migrate to same device (no-op for NdArray)
        let device2 = Default::default();
        let wave_speed2 = wave_speed1.to_device(&device2);

        assert!(wave_speed2.grid.is_some());
    }

    #[test]
    fn test_wavespeed_clone() {
        let original = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
        let cloned = original.clone();

        assert_eq!(original.get(0.5, 0.5, 0.5), cloned.get(0.5, 0.5, 0.5));
    }

    #[test]
    fn test_wavespeed_debug_format() {
        let wave_speed = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|_x, _y, _z| 1500.0));
        let debug_str = format!("{:?}", wave_speed);

        assert!(debug_str.contains("WaveSpeedFn3D"));
        assert!(debug_str.contains("has_grid"));
    }

    #[test]
    fn test_wavespeed_grid_shape() {
        let device = Default::default();
        let grid = Tensor::<TestBackend, 3>::zeros([32, 64, 128], &device);
        let wave_speed = WaveSpeedFn3D::<TestBackend>::from_grid(grid);

        let grid_ref = wave_speed.grid.as_ref().unwrap();
        assert_eq!(grid_ref.shape().dims, [32, 64, 128]);
    }

    #[test]
    fn test_wavespeed_complex_heterogeneity() {
        // Multi-region medium with three materials
        let complex = WaveSpeedFn3D::<TestBackend>::new(Arc::new(|x, y, z| {
            if z < 0.3 {
                1500.0 // Region 1: Water
            } else if z < 0.7 && x < 0.5 {
                2500.0 // Region 2: Soft tissue (left)
            } else if z < 0.7 {
                3500.0 // Region 3: Bone (right)
            } else {
                1000.0 // Region 4: Fat
            }
        }));

        assert_eq!(complex.get(0.3, 0.5, 0.2), 1500.0); // Water
        assert_eq!(complex.get(0.3, 0.5, 0.5), 2500.0); // Soft tissue
        assert_eq!(complex.get(0.7, 0.5, 0.5), 3500.0); // Bone
        assert_eq!(complex.get(0.5, 0.5, 0.8), 1000.0); // Fat
    }
}
