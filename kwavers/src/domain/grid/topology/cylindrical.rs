use super::{GridTopology, TopologyDimension};
use crate::core::error::{ConfigError, KwaversError, KwaversResult};
use ndarray::{Array1, Array2, Array3};
use std::f64::consts::PI;
use crate::core::constants::numerical::{TWO_PI};

/// Cylindrical grid topology for axisymmetric simulations
///
/// Maps 2D indices (i, j) to cylindrical coordinates (z, r).
///
/// # Coordinate Convention
///
/// - Index i → axial position z
/// - Index j → radial position r
/// - Implicit azimuthal symmetry (no φ dependence)
///
/// # Mathematical Properties
///
/// - Metric coefficient: r * dr * dz (area), 2π * r * dr * dz (volume)
/// - Singularity at r = 0 requires special handling
/// - Hankel transform for radial spectral methods
#[derive(Debug, Clone)]
pub struct CylindricalTopology {
    /// Number of grid points in axial (z) direction
    pub nz: usize,
    /// Number of grid points in radial (r) direction
    pub nr: usize,
    /// Grid spacing in axial direction (m)
    pub dz: f64,
    /// Grid spacing in radial direction (m)
    pub dr: f64,
    /// Axial origin offset (m)
    pub z0: f64,
    /// Radial origin (typically 0 for axis of symmetry)
    pub r0: f64,
    z_coords: Array1<f64>,
    r_coords: Array1<f64>,
    kz: Array1<f64>,
    kr: Array1<f64>,
}

impl CylindricalTopology {
    /// Create a new cylindrical topology
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(nz: usize, nr: usize, dz: f64, dr: f64) -> KwaversResult<Self> {
        Self::with_origin(nz, nr, dz, dr, 0.0, 0.0)
    }

    /// Create a cylindrical topology with custom origin
    /// # Errors
    /// - Returns [`KwaversError::Config`] if the precondition for a Config-class constraint is violated.
    ///
    pub fn with_origin(
        nz: usize,
        nr: usize,
        dz: f64,
        dr: f64,
        z0: f64,
        r0: f64,
    ) -> KwaversResult<Self> {
        if nz == 0 || nr == 0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "cylindrical grid dimensions".to_owned(),
                value: format!("nz={}, nr={}", nz, nr),
                constraint: "Must be positive".to_owned(),
            }));
        }

        if dz <= 0.0 || dr <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "cylindrical grid spacing".to_owned(),
                value: format!("dz={}, dr={}", dz, dr),
                constraint: "Must be positive".to_owned(),
            }));
        }

        if !dz.is_finite() || !dr.is_finite() {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "cylindrical grid spacing".to_owned(),
                value: format!("dz={}, dr={}", dz, dr),
                constraint: "Must be finite".to_owned(),
            }));
        }

        let z_coords = Array1::from_shape_fn(nz, |i| (i as f64).mul_add(dz, z0));
        let r_coords = Array1::from_shape_fn(nr, |j| (j as f64).mul_add(dr, r0));
        let kz = Self::compute_fft_wavenumbers(nz, dz);
        let kr = Self::compute_hankel_wavenumbers(nr, dr, r0);

        Ok(Self {
            nz,
            nr,
            dz,
            dr,
            z0,
            r0,
            z_coords,
            r_coords,
            kz,
            kr,
        })
    }

    /// Compute standard FFT wavenumbers
    ///
    /// k[i] = 2π * freq[i] following FFT convention:
    /// [0, 1, 2, ..., N/2, -N/2+1, ..., -1] / (N * d)
    fn compute_fft_wavenumbers(n: usize, d: f64) -> Array1<f64> {
        let dk = TWO_PI / (n as f64 * d);
        Array1::from_shape_fn(n, |i| {
            if i <= n / 2 {
                i as f64 * dk
            } else {
                (i as f64 - n as f64) * dk
            }
        })
    }

    /// Compute radial wavenumbers for discrete Hankel transform
    ///
    /// Uses zeros of J₀ Bessel function: j₀ₘ ≈ (m - 0.25) * π for large m.
    /// Wavenumbers are k_m = j₀ₘ / r_max.
    fn compute_hankel_wavenumbers(nr: usize, dr: f64, r0: f64) -> Array1<f64> {
        let r_max = (nr as f64).mul_add(dr, r0);
        Array1::from_shape_fn(nr, |m| {
            if m == 0 {
                0.0
            } else {
                let j0m = (m as f64 - 0.25) * PI;
                j0m / r_max
            }
        })
    }

    /// Get axial coordinates array
    #[must_use]
    pub fn z_coordinates(&self) -> &Array1<f64> {
        &self.z_coords
    }

    /// Get radial coordinates array
    #[must_use]
    pub fn r_coordinates(&self) -> &Array1<f64> {
        &self.r_coords
    }

    /// Get axial wavenumbers
    #[must_use]
    pub fn kz_wavenumbers(&self) -> &Array1<f64> {
        &self.kz
    }

    /// Get radial wavenumbers (Hankel transform)
    #[must_use]
    pub fn kr_wavenumbers(&self) -> &Array1<f64> {
        &self.kr
    }

    #[inline]
    #[must_use]
    pub fn z_at(&self, i: usize) -> f64 {
        (i as f64).mul_add(self.dz, self.z0)
    }

    #[inline]
    #[must_use]
    pub fn r_at(&self, j: usize) -> f64 {
        (j as f64).mul_add(self.dr, self.r0)
    }

    #[inline]
    #[must_use]
    pub fn iz_for(&self, z: f64) -> usize {
        let rel = z - self.z0;
        ((rel / self.dz).round() as usize).min(self.nz - 1)
    }

    #[inline]
    #[must_use]
    pub fn ir_for(&self, r: f64) -> usize {
        let rel = r - self.r0;
        ((rel / self.dr).round() as usize).min(self.nr - 1)
    }

    #[inline]
    #[must_use]
    pub fn z_max(&self) -> f64 {
        ((self.nz - 1) as f64).mul_add(self.dz, self.z0)
    }

    #[inline]
    #[must_use]
    pub fn r_max(&self) -> f64 {
        ((self.nr - 1) as f64).mul_add(self.dr, self.r0)
    }

    /// Create 2D meshgrid of z coordinates
    #[must_use]
    pub fn z_mesh(&self) -> Array2<f64> {
        let mut mesh = Array2::zeros((self.nz, self.nr));
        for i in 0..self.nz {
            let z = self.z_coords[i];
            for j in 0..self.nr {
                mesh[[i, j]] = z;
            }
        }
        mesh
    }

    /// Create 2D meshgrid of r coordinates
    #[must_use]
    pub fn r_mesh(&self) -> Array2<f64> {
        let mut mesh = Array2::zeros((self.nz, self.nr));
        for i in 0..self.nz {
            for j in 0..self.nr {
                mesh[[i, j]] = self.r_coords[j];
            }
        }
        mesh
    }

    /// Calculate area element for integration: r * dr * dz
    ///
    /// At r = 0, uses half-cell width to avoid singularity: 0.5 * dr² * dz
    #[must_use]
    pub fn area_element(&self, j: usize) -> f64 {
        let r = self.r_coords[j];
        if j == 0 && self.r0.abs() < 1e-15 {
            0.5 * self.dr * self.dr * self.dz
        } else {
            r * self.dr * self.dz
        }
    }

    /// Calculate volume of rotation for full 3D: 2π * r * dr * dz
    #[must_use]
    pub fn volume_element(&self, j: usize) -> f64 {
        TWO_PI * self.area_element(j)
    }
}

impl GridTopology for CylindricalTopology {
    fn dimensionality(&self) -> TopologyDimension {
        TopologyDimension::Two
    }

    fn size(&self) -> usize {
        self.nz * self.nr
    }

    fn dimensions(&self) -> [usize; 3] {
        [self.nz, self.nr, 1]
    }

    fn spacing(&self) -> [f64; 3] {
        [self.dz, self.dr, 0.0]
    }

    fn extents(&self) -> [f64; 3] {
        [
            (self.nz - 1) as f64 * self.dz,
            (self.nr - 1) as f64 * self.dr,
            0.0,
        ]
    }

    fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3] {
        let z = (indices[0] as f64).mul_add(self.dz, self.z0);
        let r = (indices[1] as f64).mul_add(self.dr, self.r0);
        [z, r, 0.0]
    }

    fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]> {
        let z = coords[0];
        let r = coords[1];

        if z < self.z0 || r < self.r0 {
            return None;
        }

        let z_rel = z - self.z0;
        let r_rel = r - self.r0;

        let z_max = (self.nz - 1) as f64 * self.dz;
        let r_max = (self.nr - 1) as f64 * self.dr;

        if z_rel > z_max || r_rel > r_max {
            return None;
        }

        let i = (z_rel / self.dz).round().min((self.nz - 1) as f64) as usize;
        let j = (r_rel / self.dr).round().min((self.nr - 1) as f64) as usize;

        Some([i, j, 0])
    }

    fn metric_coefficient(&self, indices: [usize; 3]) -> f64 {
        self.area_element(indices[1])
    }

    fn is_uniform(&self) -> bool {
        const EPSILON: f64 = 1e-10;
        (self.dz - self.dr).abs() < EPSILON
    }

    fn k_max(&self) -> f64 {
        let min_spacing = self.dz.min(self.dr);
        PI / min_spacing
    }

    fn create_field(&self) -> Array3<f64> {
        Array3::zeros((self.nz, self.nr, 1))
    }
}
