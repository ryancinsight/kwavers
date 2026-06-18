//! Cross-beam vector flow imaging.
//!
//! A single Doppler measurement recovers only the velocity *component along the
//! beam*, `v_i = v · d̂_i`, where `d̂_i` is the unit beam direction. Insonifying
//! the same sample volume from two or more directions (multi-angle / crossed
//! beams) over-determines the full velocity vector, which is recovered by least
//! squares. This removes the angle ambiguity that forces conventional Doppler to
//! assume a known beam-to-flow angle.
//!
//! # Method
//!
//! Given measurements `{v_i}` along unit directions `{d̂_i}` (i = 1..N, N ≥ 2),
//! the velocity vector `v` minimizes `Σ_i (d̂_i · v − v_i)²`. The normal
//! equations are
//!
//! ```text
//! ( Σ_i d̂_i d̂_iᵀ ) v = Σ_i v_i d̂_i,
//! ```
//!
//! a 2×2 (or 3×3) symmetric positive-definite system provided the beam
//! directions are not collinear. For two non-collinear beams the fit is exact.
//!
//! # References
//! - Dunmire, B., Beach, K. W., Labs, K.-H., Plett, M., & Strandness, D. E.
//!   (2000). "Cross-beam vector Doppler ultrasound for angle-independent
//!   velocity measurements." *Ultrasound in Medicine & Biology*, 26(8),
//!   1213–1235.
//! - Jensen, J. A., et al. (2016). "Recent advances in blood flow vector
//!   velocity imaging." *IEEE TUFFC*, 63(11), 1985–2000.

use kwavers_core::error::{KwaversError, KwaversResult};

/// A reconstructed 2-D flow velocity vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VectorVelocity {
    /// Lateral velocity component `v_x` [m/s].
    pub vx: f64,
    /// Axial velocity component `v_z` [m/s].
    pub vz: f64,
}

impl VectorVelocity {
    /// Flow-speed magnitude `‖v‖` [m/s].
    #[must_use]
    pub fn magnitude(&self) -> f64 {
        self.vz.hypot(self.vx)
    }

    /// Flow direction `atan2(v_x, v_z)` [rad], measured from the axial (z) axis.
    #[must_use]
    pub fn angle(&self) -> f64 {
        self.vx.atan2(self.vz)
    }
}

/// Cross-beam vector-flow estimator over a fixed set of beam directions.
#[derive(Debug, Clone)]
pub struct VectorFlowEstimator {
    /// Unit beam directions `(d_x, d_z)`; the inverse-projection geometry.
    directions: Vec<[f64; 2]>,
    /// Precomputed normal matrix `M = Σ d̂ d̂ᵀ` (symmetric 2×2: m00, m01, m11).
    m00: f64,
    m01: f64,
    m11: f64,
    /// `det(M)`; non-zero iff the directions span the plane.
    det: f64,
}

impl VectorFlowEstimator {
    /// Build an estimator from beam directions.
    ///
    /// Directions are normalized internally. Requires at least two beams that
    /// are not collinear.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when fewer than two beams are
    /// given, any beam has zero length, or all beams are collinear (singular
    /// normal matrix).
    pub fn new(beam_directions: &[[f64; 2]]) -> KwaversResult<Self> {
        if beam_directions.len() < 2 {
            return Err(KwaversError::InvalidInput(
                "vector flow requires at least two beam directions".to_owned(),
            ));
        }
        let mut directions = Vec::with_capacity(beam_directions.len());
        let (mut m00, mut m01, mut m11) = (0.0, 0.0, 0.0);
        for d in beam_directions {
            let norm = d[1].hypot(d[0]);
            if norm < f64::EPSILON {
                return Err(KwaversError::InvalidInput(
                    "beam direction has zero length".to_owned(),
                ));
            }
            let u = [d[0] / norm, d[1] / norm];
            m00 += u[0] * u[0];
            m01 += u[0] * u[1];
            m11 += u[1] * u[1];
            directions.push(u);
        }
        let det = m00 * m11 - m01 * m01;
        if det.abs() < 1e-12 {
            return Err(KwaversError::InvalidInput(
                "beam directions are collinear; cannot resolve the velocity vector".to_owned(),
            ));
        }
        Ok(Self {
            directions,
            m00,
            m01,
            m11,
            det,
        })
    }

    /// Number of configured beams.
    #[must_use]
    pub fn beam_count(&self) -> usize {
        self.directions.len()
    }

    /// Recover the velocity vector from beam-projected Doppler velocities.
    ///
    /// `projected[i]` is the Doppler-measured velocity component along beam `i`
    /// (`v · d̂_i`), in the same order as the configured directions.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when `projected.len()` does not
    /// match the number of beams.
    pub fn estimate(&self, projected: &[f64]) -> KwaversResult<VectorVelocity> {
        if projected.len() != self.directions.len() {
            return Err(KwaversError::InvalidInput(format!(
                "expected {} projected velocities, got {}",
                self.directions.len(),
                projected.len()
            )));
        }
        // Right-hand side b = Σ v_i d̂_i.
        let (mut bx, mut bz) = (0.0, 0.0);
        for (u, &v) in self.directions.iter().zip(projected.iter()) {
            bx += v * u[0];
            bz += v * u[1];
        }
        // Solve the 2×2 SPD system M v = b by Cramer's rule.
        let vx = (self.m11 * bx - self.m01 * bz) / self.det;
        let vz = (self.m00 * bz - self.m01 * bx) / self.det;
        Ok(VectorVelocity { vx, vz })
    }
}
