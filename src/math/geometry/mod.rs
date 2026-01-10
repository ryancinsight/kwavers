//! Geometry Helper Functions
//!
//! Geometry creation functions for transducer masks,
//! region-of-interest definitions, and spatial configurations.
//!
//! This module provides SSOT (Single Source of Truth) for geometric primitives
//! used in acoustic simulations, matching common MATLAB toolbox ergonomics.
//!
//! # MATLAB Toolbox Migration
//!
//! | Toolbox Function | Kwavers Equivalent | Description |
//! |-----------------|-------------------|-------------|
//! | `makeDisc` | [`make_disc`] | 2D circular mask |
//! | `makeBall` | [`make_ball`] | 3D spherical mask |
//! | `makeSphere` | [`make_sphere`] | Alias for [`make_ball`] |
//! | `makeLine` | [`make_line`] | Linear mask connecting two points |
//!
//! # References
//!
//! - Treeby, B. E., & Cox, B. T. (2010). "MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2), 021314.

use crate::{core::error::KwaversResult, domain::grid::Grid};
use ndarray::Array3;

/// Create a 2D circular disc mask
///
/// Generates a binary mask with `true` inside disc, `false` outside (`makeDisc`).
///
/// # Arguments
///
/// * `grid` - Grid defining spatial discretization
/// * `center` - Center point \[x, y, z\] in meters
/// * `radius` - Disc radius in meters
///
/// # Mathematical Definition
///
/// For each grid point $(x_i, y_j, z_k)$:
///
/// $$
/// \text{mask}(i,j,k) = \begin{cases}
/// \text{true} & \text{if } \sqrt{(x_i - x_c)^2 + (y_j - y_c)^2} \leq r \\\\
/// \text{false} & \text{otherwise}
/// \end{cases}
/// $$
///
/// where $(x_c, y_c, z_c)$ is the center and $r$ is the radius.
///
/// # Examples
///
/// ```rust
/// use kwavers::{math::geometry::make_disc, Grid};
///
/// let grid = Grid::new(64, 64, 1, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
/// let center = [3.2e-3, 3.2e-3, 0.0]; // Center in meters
/// let radius = 1.0e-3; // 1mm radius
///
/// let mask = make_disc(&grid, center, radius).unwrap();
/// assert_eq!(mask.dim(), (64, 64, 1));
/// ```
///
/// # MATLAB Compatibility
///
/// Equivalent to MATLAB:
/// ```matlab
/// disc = makeDisc(Nx, Ny, cx, cy, radius);
/// ```
pub fn make_disc(grid: &Grid, center: [f64; 3], radius: f64) -> KwaversResult<Array3<bool>> {
    // Validate inputs
    if radius <= 0.0 {
        return Err(crate::core::error::KwaversError::Config(
            crate::core::error::ConfigError::InvalidValue {
                parameter: "radius".to_string(),
                value: radius.to_string(),
                constraint: "Radius must be positive".to_string(),
            },
        ));
    }

    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, _dz) = (grid.dx, grid.dy, grid.dz);

    let mut mask = Array3::from_elem((nx, ny, nz), false);

    let radius_sq = radius * radius;

    // Iterate over grid points
    for i in 0..nx {
        let x = i as f64 * dx;
        let dx_sq = (x - center[0]).powi(2);

        for j in 0..ny {
            let y = j as f64 * dy;
            let dy_sq = (y - center[1]).powi(2);

            // Check if point is inside disc (2D distance)
            let dist_sq = dx_sq + dy_sq;

            if dist_sq <= radius_sq {
                // Set all z layers (2D disc extends through depth)
                for k in 0..nz {
                    mask[[i, j, k]] = true;
                }
            }
        }
    }

    Ok(mask)
}

/// Create a 3D spherical ball mask
///
/// Generates a binary mask with value `true` inside a spherical region
/// and `false` outside, matching the `makeBall` function.
///
/// # Arguments
///
/// * `grid` - Grid defining spatial discretization
/// * `center` - Center point \[x, y, z\] in meters
/// * `radius` - Sphere radius in meters
///
/// # Returns
///
/// Binary mask with `true` inside sphere, `false` outside
///
/// # Mathematical Definition
///
/// For each grid point $(x_i, y_j, z_k)$:
///
/// $$
/// \text{mask}(i,j,k) = \begin{cases}
/// \text{true} & \text{if } \sqrt{(x_i - x_c)^2 + (y_j - y_c)^2 + (z_k - z_c)^2} \leq r \\\\
/// \text{false} & \text{otherwise}
/// \end{cases}
/// $$
///
/// where $(x_c, y_c, z_c)$ is the center and $r$ is the radius.
///
/// # Examples
///
/// ```rust
/// use kwavers::{math::geometry::make_ball, Grid};
///
/// let grid = Grid::new(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
/// let center = [3.2e-3, 3.2e-3, 3.2e-3]; // Center in meters
/// let radius = 1.0e-3; // 1mm radius
///
/// let mask = make_ball(&grid, center, radius).unwrap();
/// assert_eq!(mask.dim(), (64, 64, 64));
/// ```
///
/// # MATLAB Compatibility
///
/// Equivalent to MATLAB:
/// ```matlab
/// ball = makeBall(Nx, Ny, Nz, cx, cy, cz, radius);
/// ```
pub fn make_ball(grid: &Grid, center: [f64; 3], radius: f64) -> KwaversResult<Array3<bool>> {
    // Validate inputs
    if radius <= 0.0 {
        return Err(crate::core::error::KwaversError::Config(
            crate::core::error::ConfigError::InvalidValue {
                parameter: "radius".to_string(),
                value: radius.to_string(),
                constraint: "Radius must be positive".to_string(),
            },
        ));
    }

    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

    let mut mask = Array3::from_elem((nx, ny, nz), false);

    let radius_sq = radius * radius;

    // Iterate over grid points
    for i in 0..nx {
        let x = i as f64 * dx;
        let dx_sq = (x - center[0]).powi(2);

        for j in 0..ny {
            let y = j as f64 * dy;
            let dy_sq = (y - center[1]).powi(2);

            for k in 0..nz {
                let z = k as f64 * dz;
                let dz_sq = (z - center[2]).powi(2);

                // Check if point is inside sphere (3D distance)
                let dist_sq = dx_sq + dy_sq + dz_sq;

                if dist_sq <= radius_sq {
                    mask[[i, j, k]] = true;
                }
            }
        }
    }

    Ok(mask)
}

/// Create a 3D spherical mask (alias for [`make_ball`])
///
/// Provided for MATLAB API compatibility. Functionally identical to [`make_ball`].
///
/// # Arguments
///
/// * `grid` - Grid defining spatial discretization
/// * `center` - Center point \[x, y, z\] in meters
/// * `radius` - Sphere radius in meters
///
/// # Returns
///
/// Binary mask with `true` inside sphere, `false` outside
///
/// # Examples
///
/// ```rust
/// use kwavers::{math::geometry::make_sphere, Grid};
///
/// let grid = Grid::new(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
/// let center = [3.2e-3, 3.2e-3, 3.2e-3];
/// let radius = 1.0e-3;
///
/// let mask = make_sphere(&grid, center, radius).unwrap();
/// ```
///
/// # MATLAB Compatibility
///
/// Equivalent to MATLAB:
/// ```matlab
/// sphere = makeSphere(Nx, Ny, Nz, cx, cy, cz, radius);
/// ```
#[inline]
pub fn make_sphere(grid: &Grid, center: [f64; 3], radius: f64) -> KwaversResult<Array3<bool>> {
    make_ball(grid, center, radius)
}

/// Create a line mask connecting two points
///
/// Generates a binary mask with value `true` along a line connecting
/// two points using Bresenham-like 3D line algorithm.
///
/// # Arguments
///
/// * `grid` - Grid defining spatial discretization
/// * `start` - Starting point \[x, y, z\] in meters
/// * `end` - Ending point \[x, y, z\] in meters
///
/// # Returns
///
/// Binary mask with `true` along line, `false` elsewhere
///
/// # Mathematical Definition
///
/// The line is parameterized as:
///
/// $$
/// \mathbf{p}(t) = \mathbf{p}_0 + t(\mathbf{p}_1 - \mathbf{p}_0), \quad t \in \[0, 1\]
/// $$
///
/// where $\mathbf{p}_0$ is the start point and $\mathbf{p}_1$ is the end point.
///
/// # Examples
///
/// ```rust
/// use kwavers::{math::geometry::make_line, Grid};
///
/// let grid = Grid::new(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
/// let start = [1.0e-3, 1.0e-3, 1.0e-3];
/// let end = [5.0e-3, 5.0e-3, 5.0e-3];
///
/// let mask = make_line(&grid, start, end).unwrap();
/// ```
///
/// # MATLAB Compatibility
///
/// Equivalent to MATLAB:
/// ```matlab
/// line = makeLine(Nx, Ny, Nz, start_x, start_y, start_z, end_x, end_y, end_z);
/// ```
pub fn make_line(grid: &Grid, start: [f64; 3], end: [f64; 3]) -> KwaversResult<Array3<bool>> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
    let (dx, dy, dz) = (grid.dx, grid.dy, grid.dz);

    let mut mask = Array3::from_elem((nx, ny, nz), false);

    // Convert physical coordinates to grid indices
    let start_i = (start[0] / dx).round() as isize;
    let start_j = (start[1] / dy).round() as isize;
    let start_k = (start[2] / dz).round() as isize;

    let end_i = (end[0] / dx).round() as isize;
    let end_j = (end[1] / dy).round() as isize;
    let end_k = (end[2] / dz).round() as isize;

    // 3D Bresenham-like algorithm
    let di = (end_i - start_i).abs();
    let dj = (end_j - start_j).abs();
    let dk = (end_k - start_k).abs();

    let si = if end_i > start_i { 1 } else { -1 };
    let sj = if end_j > start_j { 1 } else { -1 };
    let sk = if end_k > start_k { 1 } else { -1 };

    let dm = di.max(dj).max(dk);

    let mut i = start_i;
    let mut j = start_j;
    let mut k = start_k;

    let mut ei = dm / 2;
    let mut ej = dm / 2;
    let mut ek = dm / 2;

    for _ in 0..=dm {
        // Set mask if within bounds
        if i >= 0 && i < nx as isize && j >= 0 && j < ny as isize && k >= 0 && k < nz as isize {
            mask[[i as usize, j as usize, k as usize]] = true;
        }

        ei -= di;
        if ei < 0 {
            i += si;
            ei += dm;
        }

        ej -= dj;
        if ej < 0 {
            j += sj;
            ej += dm;
        }

        ek -= dk;
        if ek < 0 {
            k += sk;
            ek += dm;
        }
    }

    Ok(mask)
}

pub(crate) fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let mag_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if !mag_sq.is_finite() || mag_sq <= 0.0 {
        panic!("normalize3: vector must be finite and non-zero");
    }
    let mag = mag_sq.sqrt();
    [v[0] / mag, v[1] / mag, v[2] / mag]
}

pub(crate) fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

pub(crate) fn orthogonal_basis_from_normal3(normal: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let n = normalize3(normal);

    let v = if n[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };

    let u = normalize3(cross3(v, n));
    let v = cross3(n, u);

    (u, v)
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_disc_basic() {
        let grid = Grid::new(32, 32, 1, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let center = [1.6e-3, 1.6e-3, 0.0]; // Center of 32x32 grid
        let radius = 0.5e-3;

        let mask = make_disc(&grid, center, radius).unwrap();

        // Check dimensions
        assert_eq!(mask.dim(), (32, 32, 1));

        // Center should be true
        assert!(mask[[16, 16, 0]]);

        // Point at radius should be approximately on boundary
        // Count true values (should be roughly πr²/dx²)
        let count = mask.iter().filter(|&&x| x).count();
        let expected_area = std::f64::consts::PI * radius * radius;
        let cell_area = grid.dx * grid.dy;
        let expected_count = (expected_area / cell_area).round() as usize;

        // Allow 20% tolerance due to discretization
        assert!((count as f64 - expected_count as f64).abs() / (expected_count as f64) < 0.2);
    }

    #[test]
    fn test_make_disc_invalid_radius() {
        let grid = Grid::new(32, 32, 1, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let center = [1.6e-3, 1.6e-3, 0.0];

        let result = make_disc(&grid, center, -1.0);
        assert!(result.is_err());

        let result = make_disc(&grid, center, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_ball_basic() {
        let grid = Grid::new(32, 32, 32, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let center = [1.6e-3, 1.6e-3, 1.6e-3]; // Center of 32x32x32 grid
        let radius = 0.5e-3;

        let mask = make_ball(&grid, center, radius).unwrap();

        // Check dimensions
        assert_eq!(mask.dim(), (32, 32, 32));

        // Center should be true
        assert!(mask[[16, 16, 16]]);

        // Count true values (should be roughly (4/3)πr³/dx³)
        let count = mask.iter().filter(|&&x| x).count();
        let expected_volume = (4.0 / 3.0) * std::f64::consts::PI * radius.powi(3);
        let cell_volume = grid.dx * grid.dy * grid.dz;
        let expected_count = (expected_volume / cell_volume).round() as usize;

        // Allow 25% tolerance due to discretization
        assert!((count as f64 - expected_count as f64).abs() / (expected_count as f64) < 0.25);
    }

    #[test]
    fn test_make_ball_invalid_radius() {
        let grid = Grid::new(32, 32, 32, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let center = [1.6e-3, 1.6e-3, 1.6e-3];

        let result = make_ball(&grid, center, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_make_sphere_alias() {
        let grid = Grid::new(32, 32, 32, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let center = [1.6e-3, 1.6e-3, 1.6e-3];
        let radius = 0.5e-3;

        let ball = make_ball(&grid, center, radius).unwrap();
        let sphere = make_sphere(&grid, center, radius).unwrap();

        // Should be identical
        assert_eq!(ball, sphere);
    }

    #[test]
    fn test_make_line_diagonal() {
        let grid = Grid::new(32, 32, 32, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let start = [0.0, 0.0, 0.0];
        let end = [3.1e-3, 3.1e-3, 3.1e-3]; // Diagonal line

        let mask = make_line(&grid, start, end).unwrap();

        // Check dimensions
        assert_eq!(mask.dim(), (32, 32, 32));

        // Start and end should be true
        assert!(mask[[0, 0, 0]]);
        assert!(mask[[31, 31, 31]]);

        // Count should be approximately sqrt(3) * 31 (diagonal distance)
        let count = mask.iter().filter(|&&x| x).count();
        assert!(count > 31 && count < 60); // Reasonable range for diagonal
    }

    #[test]
    fn test_make_line_axis_aligned() {
        let grid = Grid::new(32, 32, 1, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let start = [0.0, 1.5e-3, 0.0];
        let end = [3.1e-3, 1.5e-3, 0.0]; // Horizontal line

        let mask = make_line(&grid, start, end).unwrap();

        // All points along y=15 should be true
        for i in 0..32 {
            assert!(mask[[i, 15, 0]]);
        }
    }

    #[test]
    fn test_disc_symmetry() {
        let grid = Grid::new(64, 64, 1, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let center = [3.2e-3, 3.2e-3, 0.0]; // Center of grid
        let radius = 1.0e-3;

        let mask = make_disc(&grid, center, radius).unwrap();

        // Check symmetry: mask[32+r, 32] == mask[32-r, 32] for small r
        for r in 1..5 {
            assert_eq!(mask[[32 + r, 32, 0]], mask[[32 - r, 32, 0]]);
            assert_eq!(mask[[32, 32 + r, 0]], mask[[32, 32 - r, 0]]);
        }
    }

    #[test]
    fn test_ball_symmetry() {
        let grid = Grid::new(64, 64, 64, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let center = [3.2e-3, 3.2e-3, 3.2e-3]; // Center of grid
        let radius = 1.0e-3;

        let mask = make_ball(&grid, center, radius).unwrap();

        // Check symmetry in all three dimensions
        for r in 1..5 {
            assert_eq!(mask[[32 + r, 32, 32]], mask[[32 - r, 32, 32]]);
            assert_eq!(mask[[32, 32 + r, 32]], mask[[32, 32 - r, 32]]);
            assert_eq!(mask[[32, 32, 32 + r]], mask[[32, 32, 32 - r]]);
        }
    }
}
