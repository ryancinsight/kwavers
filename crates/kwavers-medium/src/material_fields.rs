//! Per-cell sampling of a [`crate::Medium`]'s acoustic properties into dense fields.
//!
//! A wave solver reads medium properties once, at construction, and then runs
//! thousands of steps against the sampled arrays — the trait's per-point
//! accessors are far too slow for a hot loop. This is the single place that
//! sampling happens, so every solver sees the same property set and the same
//! unit conventions.
//!
//! The set is what a nonlinear absorbing acoustic medium is fully described by:
//! density, sound speed, power-law absorption (coefficient *and* exponent), and
//! the nonlinearity parameter `B/A`. Before this existed each solver sampled
//! the two properties it happened to need and recomputed the rest inline, which
//! is why the FDTD path had no absorption available to it at all.

use kwavers_grid::Grid;
use leto::Array3;

use crate::{acoustic::AcousticProperties, CoreMedium};

/// Container for the acoustic properties a wave solver samples from a medium.
#[derive(Debug, Clone)]
pub struct GenericMaterialFields<T> {
    /// Ambient density `ρ₀` \[kg·m⁻³].
    pub rho0: T,
    /// Ambient sound speed `c₀` \[m·s⁻¹].
    pub c0: T,
    /// Power-law absorption prefactor `α₀` \[dB/(MHz^y·cm)] — the k-Wave
    /// convention the medium trait reports. Convert with
    /// `kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_m`
    /// (time-domain, Np·m⁻¹ at a frequency) or `…_to_np_omega_m` (spectral).
    pub alpha0_db: T,
    /// Power-law absorption exponent `y` (dimensionless).
    pub alpha_power: T,
    /// Nonlinearity parameter `B/A` (dimensionless).
    pub nonlinearity: T,
}

pub type MaterialFields = GenericMaterialFields<Array3<f64>>;

impl MaterialFields {
    /// Create new zero-initialized material fields with given shape.
    ///
    /// `alpha_power` is initialized to zero like the rest; a zero exponent is
    /// not a physically meaningful medium, so callers either [`Self::sample`]
    /// a real medium or fill every field themselves.
    #[must_use]
    pub fn new(shape: (usize, usize, usize)) -> Self {
        Self {
            rho0: Array3::zeros(shape),
            c0: Array3::zeros(shape),
            alpha0_db: Array3::zeros(shape),
            alpha_power: Array3::zeros(shape),
            nonlinearity: Array3::zeros(shape),
        }
    }

    /// Sample every property from `medium` over `grid`, in one traversal.
    ///
    /// Point accessors are called once per cell per property. That is the cost
    /// of the trait boundary and it is paid once at construction; the
    /// alternative — each solver looping separately for the subset it wants —
    /// is what this replaces.
    ///
    /// A medium that reports [`CoreMedium::is_homogeneous`] is sampled once and
    /// broadcast, which turns `5·nx·ny·nz` virtual calls into five. On a 256³
    /// grid that is the difference between ~84 million dynamic dispatches at
    /// construction and none.
    #[must_use]
    pub fn sample<M>(medium: &M, grid: &Grid) -> Self
    where
        M: CoreMedium + AcousticProperties + ?Sized,
    {
        let shape = (grid.nx, grid.ny, grid.nz);
        if medium.is_homogeneous() {
            let (x, y, z) = grid.indices_to_coordinates(0, 0, 0);
            return Self {
                rho0: Array3::from_elem(shape, medium.density(0, 0, 0)),
                c0: Array3::from_elem(shape, medium.sound_speed(0, 0, 0)),
                alpha0_db: Array3::from_elem(shape, medium.alpha_coefficient(x, y, z, grid)),
                alpha_power: Array3::from_elem(shape, medium.alpha_power(x, y, z, grid)),
                nonlinearity: Array3::from_elem(shape, medium.nonlinearity(0, 0, 0)),
            };
        }

        let mut fields = Self::new(shape);
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let index = [i, j, k];
                    fields.rho0[index] = medium.density(i, j, k);
                    fields.c0[index] = medium.sound_speed(i, j, k);
                    fields.alpha0_db[index] = medium.alpha_coefficient(x, y, z, grid);
                    fields.alpha_power[index] = medium.alpha_power(x, y, z, grid);
                    fields.nonlinearity[index] = medium.nonlinearity(i, j, k);
                }
            }
        }
        fields
    }

    /// `ρ₀c₀²` \[Pa] — the lossless bulk modulus, per cell.
    #[must_use]
    pub fn bulk_modulus(&self) -> Array3<f64> {
        self.rho0.zip_map(&self.c0, |rho, c| rho * c * c)
    }

    /// `true` when every cell has a zero absorption prefactor, i.e. the medium
    /// is lossless and an absorbing solver path would be wasted work.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        self.alpha0_db.iter().all(|&a| a == 0.0)
    }
}

#[cfg(test)]
mod tests;
