//! Absorbing boundary masks for spectral CBS operators.
//!
//! # Contract
//!
//! The periodic spectral Green operator wraps waves at the grid edge. A
//! polynomial sponge applies a real diagonal taper `W` before and after the
//! spectral inverse:
//!
//! ```text
//! G_abs = W G_periodic W
//! ```
//!
//! Because `W = W^H`, the adjoint operator is exactly
//! `G_abs^H = W G_periodic^H W`.

use super::grid::GridSpec;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Absorbing boundary policy for spectral CBS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AbsorbingBoundary {
    /// Use the raw periodic spectral Green operator.
    Disabled,
    /// Polynomial sponge with `exp(-strength * eta^order)` attenuation per axis.
    Polynomial {
        /// Number of boundary cells participating in the sponge.
        thickness_cells: usize,
        /// Edge attenuation strength in nepers.
        strength_nepers: f64,
        /// Polynomial order used for normalized boundary depth.
        order: u32,
    },
}

impl AbsorbingBoundary {
    #[must_use]
    pub const fn disabled() -> Self {
        Self::Disabled
    }

    /// Construct a polynomial sponge policy.
    ///
    /// # Errors
    /// Returns an error when thickness, strength, or order are outside the
    /// finite absorbing-boundary domain.
    pub fn polynomial(
        thickness_cells: usize,
        strength_nepers: f64,
        order: u32,
    ) -> KwaversResult<Self> {
        let boundary = Self::Polynomial {
            thickness_cells,
            strength_nepers,
            order,
        };
        boundary.validate()?;
        Ok(boundary)
    }

    /// Validate scalar policy parameters independent of a concrete grid.
    ///
    /// # Errors
    /// Returns an error when the policy contains invalid scalar values.
    pub fn validate(self) -> KwaversResult<()> {
        match self {
            Self::Disabled => Ok(()),
            Self::Polynomial {
                thickness_cells,
                strength_nepers,
                order,
            } => {
                if thickness_cells == 0 {
                    return Err(KwaversError::InvalidInput(
                        "absorbing boundary thickness must be nonzero".to_owned(),
                    ));
                }
                if !strength_nepers.is_finite() || strength_nepers < 0.0 {
                    return Err(KwaversError::InvalidInput(format!(
                        "absorbing boundary strength must be finite and nonnegative, got {strength_nepers}"
                    )));
                }
                if order == 0 {
                    return Err(KwaversError::InvalidInput(
                        "absorbing boundary polynomial order must be nonzero".to_owned(),
                    ));
                }
                Ok(())
            }
        }
    }

    /// Validate scalar policy parameters and grid support.
    ///
    /// # Errors
    /// Returns an error when the policy cannot leave an undamped interior on
    /// the supplied grid.
    pub fn validate_for_grid(self, grid: GridSpec) -> KwaversResult<()> {
        self.validate()?;
        if let Self::Polynomial {
            thickness_cells, ..
        } = self
        {
            validate_grid_support(grid, thickness_cells)?;
        }
        Ok(())
    }
}

pub(super) fn absorbing_weights(
    grid: GridSpec,
    boundary: AbsorbingBoundary,
) -> KwaversResult<Vec<f64>> {
    boundary.validate_for_grid(grid)?;
    match boundary {
        AbsorbingBoundary::Disabled => Ok(vec![1.0; (grid.shape()[0] * grid.shape()[1] * grid.shape()[2])]),
        AbsorbingBoundary::Polynomial {
            thickness_cells,
            strength_nepers,
            order,
        } => polynomial_weights(grid, thickness_cells, strength_nepers, order),
    }
}

fn polynomial_weights(
    grid: GridSpec,
    thickness_cells: usize,
    strength_nepers: f64,
    order: u32,
) -> KwaversResult<Vec<f64>> {
    let (nx, ny, nz) = grid.dimensions;
    validate_grid_support(grid, thickness_cells)?;
    let mut weights = Vec::with_capacity((grid.shape()[0] * grid.shape()[1] * grid.shape()[2]));
    for ix in 0..nx {
        let ax = axis_depth(ix, nx, thickness_cells, order);
        for iy in 0..ny {
            let ay = axis_depth(iy, ny, thickness_cells, order);
            for iz in 0..nz {
                let az = axis_depth(iz, nz, thickness_cells, order);
                weights.push((-strength_nepers * (ax + ay + az)).exp());
            }
        }
    }
    Ok(weights)
}

fn validate_grid_support(grid: GridSpec, thickness_cells: usize) -> KwaversResult<()> {
    let min_dimension = [grid.dimensions.0, grid.dimensions.1, grid.dimensions.2]
        .into_iter()
        .min()
        .expect("grid dimensions are nonempty");
    if thickness_cells * 2 >= min_dimension {
        return Err(KwaversError::InvalidInput(format!(
            "absorbing boundary thickness {thickness_cells} leaves no interior cells for grid {:?}",
            grid.dimensions
        )));
    }
    Ok(())
}

fn axis_depth(index: usize, count: usize, thickness_cells: usize, order: u32) -> f64 {
    let lower_distance = index;
    let upper_distance = count - 1 - index;
    let boundary_distance = lower_distance.min(upper_distance);
    if boundary_distance >= thickness_cells {
        0.0
    } else {
        let normalized = (thickness_cells - boundary_distance) as f64 / thickness_cells as f64;
        normalized.powi(order as i32)
    }
}
