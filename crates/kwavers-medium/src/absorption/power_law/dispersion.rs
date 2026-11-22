//! Dispersion of the amplitude power law at nonnegative frequency.
use super::{PowerLawAbsorption, TWO_PI};
use aequitas::systems::si::quantities::{Dimensionless, Frequency, Velocity};
use kwavers_core::error::{KwaversError, KwaversResult};

/// Compute tan(πy/2) for finite nonnegative y. Modulo two is exact in
/// binary. Reflecting about the nearest integer bounds each argument by
/// π/4 and retains the sign on the adjacent representable values at a pole.
fn dispersion_tangent(exponent: Dimensionless) -> KwaversResult<f64> {
    let phase = exponent.into_base() % 2.0;
    let half_pi = std::f64::consts::FRAC_PI_2;
    Ok(match phase {
        0.0 => 0.0,
        0.5 => 1.0,
        1.0 => {
            return Err(KwaversError::InvalidInput(
                "Odd exponents are singular in the power-law dispersion model".into(),
            ))
        }
        1.5 => -1.0,
        x if x < 0.5 => (half_pi * x).tan(),
        x if x < 1.5 => -(half_pi * (x - 1.0)).tan().recip(),
        x => (half_pi * (x - 2.0)).tan(),
    })
}

impl PowerLawAbsorption {
    /// Phase velocity ω/k. At DC the right-hand limit is zero for a
    /// dispersive sublinear exponent and the reference speed otherwise.
    ///
    /// # Errors
    /// Rejects invalid frequency or speed, odd-exponent dispersion poles,
    /// and a nonphysical or unrepresentable positive-frequency phase branch.
    pub fn phase_velocity(&self, frequency: Frequency, c0: Velocity) -> KwaversResult<Velocity> {
        self.dispersive_velocity(frequency, c0, Dimensionless::from_base(1.0))
    }

    /// Signed group velocity dω/dk on the positive-phase branch.
    /// For k = ω(1+q)/c₀ and q proportional to ω^(y−1),
    /// dk/dω = (1+yq)/c₀. This derivative requires one law evaluation.
    ///
    /// # Errors
    /// In addition to the phase-velocity domain checks, rejects zero group
    /// slowness and velocities outside the scalar range.
    pub fn group_velocity(&self, frequency: Frequency, c0: Velocity) -> KwaversResult<Velocity> {
        self.dispersive_velocity(frequency, c0, self.exponent)
    }

    fn dispersive_velocity(
        &self,
        frequency: Frequency,
        c0: Velocity,
        weight: Dimensionless,
    ) -> KwaversResult<Velocity> {
        if !c0.into_base().is_finite() || c0.into_base() <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Sound speed must be finite and positive".into(),
            ));
        }
        if !frequency.into_base().is_finite() || frequency.into_base() < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Frequency must be finite and nonnegative".into(),
            ));
        }
        if !self.dispersion_correction || self.reference_loss.into_base() == 0.0 {
            return Ok(c0);
        }
        let tangent = dispersion_tangent(self.exponent)?;
        if tangent == 0.0 {
            return Ok(c0);
        }
        let y = self.exponent.into_base();
        if frequency.into_base() == 0.0 {
            return Ok(if y < 1.0 {
                Velocity::from_base(0.0)
            } else {
                c0
            });
        }
        let ratio = frequency / self.reference_frequency;
        let scale = (c0 * (self.reference_loss / self.reference_frequency)) / TWO_PI;
        let mut q = (scale * (tangent * ratio.into_base().powf(y - 1.0))).into_base();
        // When an intermediate leaves binary64 range, combine its logarithm
        // with the others before exponentiating. Keep near-unity ratios in
        // quotient form to retain their small relative frequency change.
        if !q.is_normal() || !(weight.into_base() * q).is_finite() {
            let r = ratio.into_base();
            let log_ratio = if r.is_normal() {
                r.ln()
            } else {
                frequency.into_base().ln() - self.reference_frequency.into_base().ln()
            };
            let log_scale = c0.into_base().ln() + self.reference_loss.into_base().ln()
                - self.reference_frequency.into_base().ln()
                - TWO_PI.ln()
                + tangent.abs().ln();
            let log_q = (y - 1.0).mul_add(log_ratio, log_scale);
            let log_weighted = weight.into_base().ln() + log_q;
            // Group slowness uses wq. A sublinear exponent can make
            // wq representable even if q alone would overflow.
            if tangent > 0.0 && log_q >= f64::MAX.ln() && log_weighted < f64::MAX.ln() {
                return finite_velocity(c0 / (1.0 + log_weighted.exp()));
            }
            if tangent > 0.0 && log_weighted >= f64::MAX.ln() {
                // Adding one to wq at this scale is below half an ulp.
                let speed = (c0.into_base().ln() - log_weighted).exp();
                return finite_velocity(Velocity::from_base(speed));
            }
            q = log_q.exp().copysign(tangent);
        }
        if 1.0 + q <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Nonpositive power-law phase slowness".into(),
            ));
        }
        finite_velocity(c0 / weight.into_base().mul_add(q, 1.0))
    }
}

fn finite_velocity(velocity: Velocity) -> KwaversResult<Velocity> {
    if !velocity.into_base().is_finite() || velocity.into_base() == 0.0 {
        return Err(KwaversError::InvalidInput(
            "Power-law velocity is singular or outside scalar range".into(),
        ));
    }
    Ok(velocity)
}
