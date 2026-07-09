//! Kirchhoff (diffraction-stack) migration.
//!
//! Kirchhoff migration forms an image by summing recorded data along
//! diffraction surfaces. For an image point `x`, the energy scattered from `x`
//! and recorded by a source–receiver pair `(s, r)` arrives at two-way traveltime
//! `T = T_s(x) + T_r(x)`. Summing each trace at its predicted arrival time over
//! all pairs constructively reinforces true scatterers and destructively cancels
//! elsewhere:
//!
//! ```text
//! I(x) = Σ_{(s,r)}  d_{s,r}( T_s(x) + T_r(x) ).
//! ```
//!
//! Traveltime tables `T_s`, `T_r` come from the [`super::eikonal`] solver, so the
//! method handles heterogeneous velocity, unlike straight-ray migration.
//!
//! # Reference
//! - Schneider, W. A. (1978). "Integral formulation for migration in two and
//!   three dimensions." *Geophysics*, 43(1), 49–76.
//! - Bleistein, N., Cohen, J. K., & Stockwell, J. W. (2001). *Mathematics of
//!   Multidimensional Seismic Imaging, Migration, and Inversion*. Springer.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

/// A recorded trace for one source–receiver pair.
#[derive(Debug, Clone)]
pub struct Trace {
    /// Index into the source traveltime-table list.
    pub source: usize,
    /// Index into the receiver traveltime-table list.
    pub receiver: usize,
    /// Time samples, uniformly spaced by `dt`.
    pub samples: Vec<f64>,
}

/// Linear interpolation of a uniformly-sampled trace at continuous time `t`.
/// Returns `0.0` outside the recorded window.
fn sample_at(samples: &[f64], t: f64, dt: f64) -> f64 {
    if t < 0.0 || samples.len() < 2 {
        return 0.0;
    }
    let x = t / dt;
    let i = x.floor() as usize;
    if i + 1 >= samples.len() {
        return 0.0;
    }
    let frac = x - i as f64;
    samples[i] * (1.0 - frac) + samples[i + 1] * frac
}

/// Kirchhoff diffraction-stack migrator.
#[derive(Debug, Clone, Copy)]
pub struct KirchhoffMigrator {
    /// Trace sample interval `dt` [s].
    dt: f64,
}

impl KirchhoffMigrator {
    /// Create a migrator with trace sampling interval `dt` [s].
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when `dt ≤ 0`.
    pub fn new(dt: f64) -> KwaversResult<Self> {
        if dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Kirchhoff dt must be positive".to_owned(),
            ));
        }
        Ok(Self { dt })
    }

    /// Migrate `traces` into an image using per-source and per-receiver
    /// traveltime tables (each `[nx, ny, nz]`, e.g. from
    /// [`super::eikonal::EikonalSolver`]).
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when the traveltime-table lists are
    /// empty, their shapes disagree, or a trace references a missing table.
    pub fn migrate(
        &self,
        traces: &[Trace],
        source_tt: &[Array3<f64>],
        receiver_tt: &[Array3<f64>],
    ) -> KwaversResult<Array3<f64>> {
        let dim = source_tt
            .first()
            .or_else(|| receiver_tt.first())
            .ok_or_else(|| KwaversError::InvalidInput("no traveltime tables provided".to_owned()))?
            .dim();
        for tt in source_tt.iter().chain(receiver_tt.iter()) {
            if tt.dim() != dim {
                return Err(KwaversError::InvalidInput(
                    "traveltime tables have mismatched shapes".to_owned(),
                ));
            }
        }
        let (nx, ny, nz) = dim;
        let mut image = Array3::zeros((nx, ny, nz));
        for tr in traces {
            let st = source_tt.get(tr.source).ok_or_else(|| {
                KwaversError::InvalidInput("trace references missing source table".to_owned())
            })?;
            let rt = receiver_tt.get(tr.receiver).ok_or_else(|| {
                KwaversError::InvalidInput("trace references missing receiver table".to_owned())
            })?;
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        let t = st[[i, j, k]] + rt[[i, j, k]];
                        image[[i, j, k]] += sample_at(&tr.samples, t, self.dt);
                    }
                }
            }
        }
        Ok(image)
    }
}
