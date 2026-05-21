//! CBS grid and bandlimited point projection.

use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::ElementPosition;
use std::f64::consts::PI;

/// Default BLI tolerance used by the canonical k-Wave-compatible source path.
pub const DEFAULT_BLI_TOLERANCE: f64 = 0.05;

/// Uniform Cartesian volume grid for CBS kernels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSpec {
    pub dimensions: (usize, usize, usize),
    pub spacing_m: f64,
}

impl GridSpec {
    /// Create a centered uniform grid spec.
    ///
    /// # Errors
    /// Returns an error if dimensions or spacing are invalid.
    pub fn new(dimensions: (usize, usize, usize), spacing_m: f64) -> KwaversResult<Self> {
        let (nx, ny, nz) = dimensions;
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(KwaversError::InvalidInput(format!(
                "CBS grid dimensions must be nonzero, got {dimensions:?}"
            )));
        }
        if !spacing_m.is_finite() || spacing_m <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CBS grid spacing must be positive and finite, got {spacing_m}"
            )));
        }
        Ok(Self {
            dimensions,
            spacing_m,
        })
    }

    #[must_use]
    pub fn len(self) -> usize {
        let (nx, ny, nz) = self.dimensions;
        nx * ny * nz
    }

    #[must_use]
    pub fn cell_volume_m3(self) -> f64 {
        self.spacing_m.powi(3)
    }

    #[must_use]
    pub fn linear_index(self, ix: usize, iy: usize, iz: usize) -> usize {
        let (_, ny, nz) = self.dimensions;
        ix * ny * nz + iy * nz + iz
    }

    #[must_use]
    pub fn center_at(self, ix: usize, iy: usize, iz: usize) -> ElementPosition {
        let (nx, ny, nz) = self.dimensions;
        let cx = 0.5 * nx as f64;
        let cy = 0.5 * ny as f64;
        let cz = 0.5 * nz as f64;
        ElementPosition {
            x_m: (ix as f64 + 0.5 - cx) * self.spacing_m,
            y_m: (iy as f64 + 0.5 - cy) * self.spacing_m,
            z_m: (iz as f64 + 0.5 - cz) * self.spacing_m,
        }
    }

    #[must_use]
    pub fn centers(self) -> Vec<(usize, ElementPosition)> {
        let (nx, ny, nz) = self.dimensions;
        let mut centers = Vec::with_capacity(self.len());
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    centers.push((self.linear_index(ix, iy, iz), self.center_at(ix, iy, iz)));
                }
            }
        }
        centers
    }

    #[must_use]
    pub fn min_distance_m(self) -> f64 {
        0.5 * self.spacing_m
    }
}

/// BLI projection configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BliConfig {
    pub tolerance: f64,
}

impl Default for BliConfig {
    fn default() -> Self {
        Self {
            tolerance: DEFAULT_BLI_TOLERANCE,
        }
    }
}

impl BliConfig {
    /// Stencil half-width in grid cells.
    ///
    /// # Errors
    /// Returns an error if tolerance is invalid.
    pub fn half_width(self) -> KwaversResult<isize> {
        if !self.tolerance.is_finite() || self.tolerance <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "BLI tolerance must be positive and finite, got {}",
                self.tolerance
            )));
        }
        Ok((1.0 / (PI * self.tolerance)).ceil() as isize)
    }
}

/// Weighted grid contribution for point injection or recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridWeight {
    pub linear_index: usize,
    pub weight: f64,
}

/// Compute BLI weights for a continuous point on a centered grid.
///
/// # Errors
/// Returns an error if the grid/config are invalid.
pub fn bli_weights(
    grid: GridSpec,
    point: ElementPosition,
    config: BliConfig,
) -> KwaversResult<Vec<GridWeight>> {
    let half_width = config.half_width()?;
    if !within_stencil_support(grid, point, half_width) {
        return Ok(Vec::new());
    }
    let (nx, ny, nz) = grid.dimensions;
    let nearest = nearest_indices(grid, point);
    let on_grid = on_grid_axes(grid, point, nearest);
    let mut weights = Vec::new();

    for di in -half_width..=half_width {
        if on_grid[0] && di != 0 {
            continue;
        }
        for dj in -half_width..=half_width {
            if on_grid[1] && dj != 0 {
                continue;
            }
            for dk in -half_width..=half_width {
                if on_grid[2] && dk != 0 {
                    continue;
                }
                let ix = nearest[0] as isize + di;
                let iy = nearest[1] as isize + dj;
                let iz = nearest[2] as isize + dk;
                if ix < 0
                    || iy < 0
                    || iz < 0
                    || ix >= nx as isize
                    || iy >= ny as isize
                    || iz >= nz as isize
                {
                    continue;
                }
                let ix = ix as usize;
                let iy = iy as usize;
                let iz = iz as usize;
                let center = grid.center_at(ix, iy, iz);
                let weight = sinc(PI * (center.x_m - point.x_m) / grid.spacing_m)
                    * sinc(PI * (center.y_m - point.y_m) / grid.spacing_m)
                    * sinc(PI * (center.z_m - point.z_m) / grid.spacing_m);
                if weight != 0.0 {
                    weights.push(GridWeight {
                        linear_index: grid.linear_index(ix, iy, iz),
                        weight,
                    });
                }
            }
        }
    }
    Ok(weights)
}

fn nearest_indices(grid: GridSpec, point: ElementPosition) -> [usize; 3] {
    let (nx, ny, nz) = grid.dimensions;
    [
        nearest_axis(nx, grid.spacing_m, point.x_m),
        nearest_axis(ny, grid.spacing_m, point.y_m),
        nearest_axis(nz, grid.spacing_m, point.z_m),
    ]
}

fn within_stencil_support(grid: GridSpec, point: ElementPosition, half_width: isize) -> bool {
    let (nx, ny, nz) = grid.dimensions;
    axis_within_support(nx, grid.spacing_m, point.x_m, half_width)
        && axis_within_support(ny, grid.spacing_m, point.y_m, half_width)
        && axis_within_support(nz, grid.spacing_m, point.z_m, half_width)
}

fn axis_within_support(n: usize, spacing_m: f64, value_m: f64, half_width: isize) -> bool {
    let min_center = (0.5 - 0.5 * n as f64) * spacing_m;
    let max_center = (n as f64 - 0.5 - 0.5 * n as f64) * spacing_m;
    let support = half_width as f64 * spacing_m;
    value_m >= min_center - support && value_m <= max_center + support
}

fn nearest_axis(n: usize, spacing_m: f64, value_m: f64) -> usize {
    let center = 0.5 * n as f64;
    let raw = value_m / spacing_m + center - 0.5;
    raw.round().clamp(0.0, (n - 1) as f64) as usize
}

fn on_grid_axes(grid: GridSpec, point: ElementPosition, nearest: [usize; 3]) -> [bool; 3] {
    let center = grid.center_at(nearest[0], nearest[1], nearest[2]);
    let threshold = grid.spacing_m * 1.0e-3;
    [
        (center.x_m - point.x_m).abs() < threshold,
        (center.y_m - point.y_m).abs() < threshold,
        (center.z_m - point.z_m).abs() < threshold,
    ]
}

#[inline]
fn sinc(x: f64) -> f64 {
    if x.abs() <= f64::EPSILON {
        1.0
    } else {
        x.sin() / x
    }
}
