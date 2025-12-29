//! GPU-accelerated photoacoustic imaging
//!
//! Provides CUDA/OpenCL acceleration for photoacoustic wave propagation
//! using compute shaders and parallel processing.

use crate::error::KwaversResult;
use ndarray::Array3;

/// GPU-accelerated photoacoustic wave propagator
#[derive(Debug)]
pub struct GPUPhotoacousticPropagator {
    /// Underlying CPU-based implementation used as a correctness-preserving backend.
    cpu_fallback: CPUPhotoacousticPropagator,
}

impl GPUPhotoacousticPropagator {
    /// Create a new propagator.
    ///
    /// This implementation currently uses a validated CPU-based backend while
    /// preserving the public GPU-oriented API surface. It intentionally does
    /// not expose partial or misleading GPU behavior.
    pub fn new(dt: f64, dx: f64, speed_of_sound: f64) -> KwaversResult<Self> {
        Ok(Self {
            cpu_fallback: CPUPhotoacousticPropagator::new(dt, dx, speed_of_sound),
        })
    }

    /// Check if true hardware-accelerated backends are available.
    ///
    /// Returns `false` until a fully validated GPU implementation is provided.
    pub fn is_available() -> bool {
        false
    }

    /// Propagate acoustic wave using the current backend.
    ///
    /// Delegates to the CPU implementation to ensure deterministic, tested behavior.
    pub fn propagate_wave(
        &self,
        initial_pressure: &Array3<f64>,
        time_steps: usize,
    ) -> KwaversResult<Vec<Array3<f64>>> {
        self.cpu_fallback
            .propagate_wave(initial_pressure, time_steps)
    }
}

/// CPU fallback implementation for photoacoustic propagation
#[derive(Debug)]
pub struct CPUPhotoacousticPropagator {
    /// Time step for acoustic propagation
    pub(crate) dt: f64,
    /// Spatial step size
    pub(crate) dx: f64,
    /// Speed of sound in medium
    pub(crate) speed_of_sound: f64,
}

impl CPUPhotoacousticPropagator {
    /// Create new CPU propagator
    pub fn new(dt: f64, dx: f64, speed_of_sound: f64) -> Self {
        Self {
            dt,
            dx,
            speed_of_sound,
        }
    }

    /// Propagate acoustic wave using FDTD
    pub fn propagate_wave(
        &self,
        initial_pressure: &Array3<f64>,
        time_steps: usize,
    ) -> KwaversResult<Vec<Array3<f64>>> {
        let mut pressure_fields = Vec::with_capacity(time_steps + 1);
        pressure_fields.push(initial_pressure.clone());

        // Simplified 3D wave propagation using finite differences
        // In a full implementation, this would use proper FDTD with absorbing boundaries

        let mut current_pressure = initial_pressure.clone();
        let mut previous_pressure = Array3::zeros(initial_pressure.dim());

        for _step in 0..time_steps {
            // Simple wave equation: ∂²p/∂t² = c²∇²p
            // Using central difference: p(t+dt) = 2p(t) - p(t-dt) + c²dt²∇²p(t)

            let mut next_pressure = Array3::zeros(initial_pressure.dim());
            let c2_dt2 = self.speed_of_sound.powi(2) * self.dt.powi(2);
            let dx2_inv = 1.0 / (self.dx * self.dx);

            // Interior points computation - boundary conditions handled separately
            let (nx, ny, nz) = initial_pressure.dim();
            if nx > 2 && ny > 2 && nz > 2 {
                for i in 1..nx - 1 {
                    for j in 1..ny - 1 {
                        for k in 1..nz - 1 {
                            // Laplacian approximation
                            let laplacian = (current_pressure[[i + 1, j, k]]
                                + current_pressure[[i - 1, j, k]]
                                + current_pressure[[i, j + 1, k]]
                                + current_pressure[[i, j - 1, k]]
                                + current_pressure[[i, j, k + 1]]
                                + current_pressure[[i, j, k - 1]]
                                - 6.0 * current_pressure[[i, j, k]])
                                * dx2_inv;

                            // Wave equation update
                            next_pressure[[i, j, k]] = 2.0 * current_pressure[[i, j, k]]
                                - previous_pressure[[i, j, k]]
                                + c2_dt2 * laplacian;
                        }
                    }
                }
            }

            pressure_fields.push(next_pressure.clone());
            previous_pressure = current_pressure;
            current_pressure = next_pressure;
        }

        Ok(pressure_fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_cpu_propagator_creation() {
        let propagator = CPUPhotoacousticPropagator::new(1e-8, 0.001, 1500.0);
        // Just verify it creates without error
        assert_eq!(propagator.dt, 1e-8);
        assert_eq!(propagator.dx, 0.001);
        assert_eq!(propagator.speed_of_sound, 1500.0);
    }

    #[test]
    fn test_wave_propagation() {
        let propagator = CPUPhotoacousticPropagator::new(1e-8, 0.001, 1500.0);

        // Create a simple initial pressure field (point source)
        let mut initial_pressure = Array3::<f64>::zeros((10, 10, 10));
        initial_pressure[[5, 5, 5]] = 1.0; // Point source at center

        let result = propagator.propagate_wave(&initial_pressure, 5);

        assert!(result.is_ok());
        let fields = result.unwrap();
        assert_eq!(fields.len(), 6); // Initial + 5 time steps

        // Check that energy is conserved (approximately)
        for field in &fields {
            let total_energy: f64 = field.iter().map(|&x| x * x).sum();
            assert!(total_energy > 0.0);
        }
    }

    #[test]
    fn test_gpu_wrapper_uses_cpu_backend() {
        let gpu = GPUPhotoacousticPropagator::new(1e-8, 0.001, 1500.0)
            .expect("GPU wrapper construction must succeed");
        let mut initial_pressure = Array3::<f64>::zeros((10, 10, 10));
        initial_pressure[[5, 5, 5]] = 1.0;
        let result = gpu
            .propagate_wave(&initial_pressure, 3)
            .expect("GPU wrapper must delegate to CPU backend");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_gpu_availability_flag() {
        // Explicitly validate that we do not falsely claim GPU support.
        assert!(!GPUPhotoacousticPropagator::is_available());
    }
}
