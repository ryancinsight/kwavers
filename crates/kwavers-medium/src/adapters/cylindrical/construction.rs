//! `CylindricalMediumProjection::new` — samples the 3D medium at the
//! `θ = 0` plane and caches the resulting 2D property arrays.

use ndarray::Array2;

use super::CylindricalMediumProjection;
use kwavers_core::error::{ConfigError, KwaversError, KwaversResult};
use kwavers_grid::{CylindricalTopology, Grid};
use crate::Medium;

impl<'a, M: Medium> CylindricalMediumProjection<'a, M> {
    /// Create a new cylindrical projection of the 3D medium
    ///
    /// This constructor samples the medium at the θ = 0 plane and caches
    /// the resulting 2D property arrays.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Grid dimensions don't match medium array dimensions
    /// - Cylindrical topology coordinates exceed grid bounds
    /// - Medium validation fails
    pub fn new(
        medium: &'a M,
        grid: &'a Grid,
        topology: &'a CylindricalTopology,
    ) -> KwaversResult<Self> {
        // Validate medium against grid
        medium.validate(grid)?;

        // Check if medium is homogeneous (fast path)
        let is_homogeneous = medium.is_homogeneous();

        // Allocate 2D arrays
        let shape = (topology.nz, topology.nr);
        let mut sound_speed_2d = Array2::zeros(shape);
        let mut density_2d = Array2::zeros(shape);
        let mut absorption_2d = Array2::zeros(shape);

        // Track min/max sound speed during projection
        let mut max_c = f64::NEG_INFINITY;
        let mut min_c = f64::INFINITY;

        // Project medium properties onto cylindrical grid
        //
        // For each point (iz, ir) in the cylindrical grid:
        // 1. Compute physical coordinates (r, z)
        // 2. Map to Cartesian (x, y, z) with x = r, y = 0 (θ = 0 plane)
        // 3. Convert to grid indices (i, j, k)
        // 4. Sample medium properties
        for iz in 0..topology.nz {
            let z = topology.z_at(iz);

            for ir in 0..topology.nr {
                let r = topology.r_at(ir);

                let x = r;
                let y = 0.0;

                let (i, j, k) = grid.coordinates_to_indices(x, y, z).ok_or_else(|| {
                    KwaversError::Config(ConfigError::InvalidValue {
                        parameter: "cylindrical topology".to_owned(),
                        value: format!(
                            "iz={}, ir={} -> x={}, y={}, z={} (out of bounds)",
                            iz, ir, x, y, z
                        ),
                        constraint: format!(
                            "Coordinates must be within grid bounds: ({}, {}, {})",
                            grid.nx as f64 * grid.dx,
                            grid.ny as f64 * grid.dy,
                            grid.nz as f64 * grid.dz
                        ),
                    })
                })?;

                let c = medium.sound_speed(i, j, k);
                let rho = medium.density(i, j, k);
                let alpha = medium.absorption(i, j, k);

                sound_speed_2d[[iz, ir]] = c;
                density_2d[[iz, ir]] = rho;
                absorption_2d[[iz, ir]] = alpha;

                max_c = max_c.max(c);
                min_c = min_c.min(c);

                if c <= 0.0 || !c.is_finite() {
                    return Err(KwaversError::Config(ConfigError::InvalidValue {
                        parameter: "sound_speed".to_owned(),
                        value: c.to_string(),
                        constraint: "Must be positive and finite".to_owned(),
                    }));
                }

                if rho <= 0.0 || !rho.is_finite() {
                    return Err(KwaversError::Config(ConfigError::InvalidValue {
                        parameter: "density".to_owned(),
                        value: rho.to_string(),
                        constraint: "Must be positive and finite".to_owned(),
                    }));
                }

                if alpha < 0.0 || !alpha.is_finite() {
                    return Err(KwaversError::Config(ConfigError::InvalidValue {
                        parameter: "absorption".to_owned(),
                        value: alpha.to_string(),
                        constraint: "Must be non-negative and finite".to_owned(),
                    }));
                }
            }
        }

        // Project nonlinearity if medium has it.
        // Homogeneous medium: constant sample at origin; heterogeneous: per-cell projection.
        let nonlinearity_2d = if is_homogeneous {
            let b_over_a = medium.nonlinearity(0, 0, 0);
            if b_over_a != 0.0 {
                Some(Array2::from_elem(shape, b_over_a))
            } else {
                None
            }
        } else {
            let mut nonlin_2d = Array2::zeros(shape);
            let mut has_nonlinearity = false;

            for iz in 0..topology.nz {
                let z = topology.z_at(iz);

                for ir in 0..topology.nr {
                    let r = topology.r_at(ir);
                    let x = r;
                    let y = 0.0;

                    if let Some((i, j, k)) = grid.coordinates_to_indices(x, y, z) {
                        let b_over_a = medium.nonlinearity(i, j, k);
                        nonlin_2d[[iz, ir]] = b_over_a;

                        if b_over_a != 0.0 {
                            has_nonlinearity = true;
                        }
                    }
                }
            }

            if has_nonlinearity {
                Some(nonlin_2d)
            } else {
                None
            }
        };

        Ok(Self {
            medium,
            grid,
            topology,
            sound_speed_2d,
            density_2d,
            absorption_2d,
            nonlinearity_2d,
            max_sound_speed: max_c,
            min_sound_speed: min_c,
            is_homogeneous,
        })
    }
}
