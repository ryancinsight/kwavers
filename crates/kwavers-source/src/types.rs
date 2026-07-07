//! Source domain primitive
//!
//! Defines the core Source trait for wave generation.

// GridSource moved to parent module

use kwavers_grid::Grid;
use kwavers_signal::Signal;
use ndarray::Array3;
use std::fmt::Debug;

// GridSource re-exported by parent mod

use crate::parallel::zip_mut_ref;
use serde::{Deserialize, Serialize};

/// Type of source injection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SourceField {
    #[default]
    Pressure,
    VelocityX,
    VelocityY,
    VelocityZ,
}

/// Electromagnetic polarization state
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourcePolarization {
    /// Linear polarization along x-axis
    LinearX,
    /// Linear polarization along y-axis
    LinearY,
    /// Linear polarization along z-axis
    LinearZ,
    /// Right circular polarization
    RightCircular,
    /// Left circular polarization
    LeftCircular,
    /// Elliptical polarization (ratio, phase difference)
    Elliptical { ratio: f64, phase_diff: f64 },
}

/// Electromagnetic wave type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SourceEMWaveType {
    /// Transverse electromagnetic (no longitudinal components)
    TEM,
    /// Transverse electric (E_z = 0)
    TE,
    /// Transverse magnetic (H_z = 0)
    TM,
    /// Hybrid mode (both E_z, H_z nonzero)
    Hybrid,
}

pub type SourceType = SourceField;

/// Focal properties for focused wave sources
///
/// These properties characterize the focusing behavior of sources like
/// focused transducers, Gaussian beams, and phased arrays.
#[derive(Debug, Clone, Copy)]
pub struct SourceFocalProperties {
    /// Focal point position (m)
    pub focal_point: (f64, f64, f64),

    /// Focal depth/length: distance from source center to focal point (m)
    pub focal_depth: f64,

    /// Spot size at focus: beam waist or FWHM (m)
    pub spot_size: f64,

    /// F-number: focal_length / aperture_diameter (dimensionless)
    pub f_number: Option<f64>,

    /// Rayleigh range: depth of focus (m)
    pub rayleigh_range: Option<f64>,

    /// Numerical aperture: sin(half_angle) (dimensionless)
    pub numerical_aperture: Option<f64>,

    /// Focal gain: intensity amplification at focus (dimensionless)
    pub focal_gain: Option<f64>,
}

/// Efficient source trait using mask-based approach.
///
// dyn: used as `dyn Source` (open, cross-crate-extensible implementor set —
// 6 impls in kwavers-source, 8+ in kwavers-transducer — and stored in
// heterogeneous `Vec<Box<dyn Source>>` collections). Per the zero-cost policy
// (ADR 012) this is a sanctioned dynamic-dispatch boundary: the concrete type is
// genuinely unknown at the storage site and dispatch is O(num_sources)/step
// (the scalar `amplitude(t)`), never per-cell. Trait methods stay object-safe.
pub trait Source: Debug + Sync + Send {
    /// Create a source mask on the grid (1.0 at source locations, 0.0 elsewhere)
    /// This is called once during initialization for optimal performance
    fn create_mask(&self, grid: &Grid) -> Array3<f64>;

    /// Add this source's spatial mask into a caller-owned mask buffer.
    ///
    /// # Contract
    /// `mask.dim()` must equal `(grid.nx, grid.ny, grid.nz)`. The default
    /// implementation derives the elementwise result from [`Source::create_mask`].
    /// Implementations override this method when their mask algebra is exactly
    /// additive without extra collision state.
    fn add_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        let source_mask = self.create_mask(grid);
        zip_mut_ref(mask.view_mut(), source_mask.view(), |dst, &src| {
            *dst += src;
        });
    }

    /// Write this source's spatial mask into a caller-owned buffer.
    ///
    /// # Theorem
    /// For every source mask `M_s`, `create_mask_into(grid, out)` computes
    /// `out = 0 + M_s`, which is elementwise identical to assigning
    /// `create_mask(grid)` into `out`. This lets solvers reuse one mask buffer
    /// across timesteps without changing source superposition semantics.
    fn create_mask_into(&self, grid: &Grid, mask: &mut Array3<f64>) {
        debug_assert_eq!(mask.dim(), (grid.nx, grid.ny, grid.nz));
        mask.fill(0.0);
        self.add_mask_into(grid, mask);
    }

    /// Get the signal amplitude at time t
    /// This is called once per time step, not per grid point
    fn amplitude(&self, t: f64) -> f64;

    /// Get source positions for visualization/analysis
    fn positions(&self) -> Vec<(f64, f64, f64)>;

    /// Get the underlying signal
    fn signal(&self) -> &dyn Signal;

    /// Get the type of source
    fn source_type(&self) -> SourceField {
        SourceField::Pressure
    }

    /// Get the initial amplitude (for p0/u0)
    /// If non-zero, this source contributes to the initial conditions
    fn initial_amplitude(&self) -> f64 {
        0.0
    }

    /// Get source term at a specific position and time
    /// Uses `create_mask()` and `amplitude()` internally for compatibility
    fn get_source_term(&self, t: f64, x: f64, y: f64, z: f64, grid: &Grid) -> f64 {
        // Fallback implementation - inefficient but maintains compatibility
        if let Some((i, j, k)) = grid.position_to_indices(x, y, z) {
            let mask = self.create_mask(grid);
            mask.get((i, j, k)).copied().unwrap_or(0.0) * self.amplitude(t)
        } else {
            0.0
        }
    }

    // ==================== Focal Properties API ====================
    // These methods enable PINN adapters and analysis tools to extract
    // focal properties from sources without tight coupling to concrete types.

    /// Get focal point position (m)
    ///
    /// Returns the position where the wave field converges to maximum intensity.
    /// Returns `None` for unfocused sources (plane waves, point sources, etc.)
    fn focal_point(&self) -> Option<(f64, f64, f64)> {
        None
    }

    /// Get focal depth (m)
    ///
    /// Distance from the source center/aperture to the focal point.
    /// Also known as focal length for geometric focusing.
    fn focal_depth(&self) -> Option<f64> {
        None
    }

    /// Get spot size at focus (m)
    ///
    /// Minimum transverse beam dimension at the focal point.
    /// - For Gaussian beams: beam waist w0
    /// - For focused transducers: FWHM of main lobe
    /// - For diffraction-limited systems: ~λ * f_number
    fn spot_size(&self) -> Option<f64> {
        None
    }

    /// Get F-number (dimensionless)
    ///
    /// Ratio of focal length to aperture diameter: F = f / D
    /// Characterizes the focusing strength and depth of field:
    /// - Small F# (<1): Strong focusing, shallow depth
    /// - Large F# (>3): Weak focusing, large depth
    fn f_number(&self) -> Option<f64> {
        None
    }

    /// Get Rayleigh range (m)
    ///
    /// Depth of focus: distance over which beam radius ≤ √2 * spot_size.
    /// For Gaussian beams: z_R = π * w0² / λ
    /// For focused transducers: approximately λ * F#²
    fn rayleigh_range(&self) -> Option<f64> {
        None
    }

    /// Get numerical aperture (dimensionless)
    ///
    /// NA = sin(θ) where θ is the half-angle of the convergence cone.
    /// Related to F-number by: NA ≈ 1 / (2 * F#)
    /// Higher NA → stronger focusing, better resolution
    fn numerical_aperture(&self) -> Option<f64> {
        None
    }

    /// Get focal gain (dimensionless)
    ///
    /// Intensity amplification factor at focus compared to source surface.
    /// Accounts for geometric focusing and diffraction effects.
    /// For ideal focusing: gain ≈ (aperture_area) / (spot_size²)
    fn focal_gain(&self) -> Option<f64> {
        None
    }

    /// Get comprehensive focal properties
    ///
    /// Convenience method that collects all focal properties into a single struct.
    /// Returns `None` if the source is not focused.
    fn get_focal_properties(&self) -> Option<SourceFocalProperties> {
        // Only return properties if source has a focal point
        let focal_point = self.focal_point()?;
        let focal_depth = self.focal_depth()?;
        let spot_size = self.spot_size()?;

        Some(SourceFocalProperties {
            focal_point,
            focal_depth,
            spot_size,
            f_number: self.f_number(),
            rayleigh_range: self.rayleigh_range(),
            numerical_aperture: self.numerical_aperture(),
            focal_gain: self.focal_gain(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_signal::{NullSignal, Signal};
    use ndarray::Array3;

    #[derive(Debug)]
    struct DefaultMaskSource {
        mask: Array3<f64>,
        signal: NullSignal,
    }

    impl DefaultMaskSource {
        fn new(mask: Array3<f64>) -> Self {
            Self {
                mask,
                signal: NullSignal::new(),
            }
        }
    }

    impl Source for DefaultMaskSource {
        fn create_mask(&self, _grid: &Grid) -> Array3<f64> {
            self.mask.clone()
        }

        fn amplitude(&self, _t: f64) -> f64 {
            0.0
        }

        fn positions(&self) -> Vec<(f64, f64, f64)> {
            Vec::new()
        }

        fn signal(&self) -> &dyn Signal {
            &self.signal
        }
    }

    #[test]
    fn default_add_mask_into_accumulates_created_mask() {
        let grid = Grid::new(3, 2, 2, 1.0, 1.0, 1.0).unwrap();
        let source = DefaultMaskSource::new(
            Array3::from_shape_vec(
                (grid.nx, grid.ny, grid.nz),
                vec![
                    0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0, 55.0, 89.0, 144.0,
                ],
            )
            .unwrap(),
        );
        let mut target = Array3::from_elem((grid.nx, grid.ny, grid.nz), 10.0);
        let expected = source.create_mask(&grid).mapv(|value| value + 10.0);

        source.add_mask_into(&grid, &mut target);

        assert_eq!(target, expected);
    }
}
