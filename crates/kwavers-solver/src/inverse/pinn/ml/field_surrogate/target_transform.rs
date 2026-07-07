//! Per-channel target transforms for the field-surrogate PINN.
//!
//! The network always emits values in `[-1, 1]`. Mapping the physical
//! pressure target into that range is a free design choice that
//! controls how the loss is distributed across the dynamic range:
//!
//! * [`TargetTransform::Linear`] divides by a constant scale. The MSE
//!   loss is then proportional to `(Δp)²`, so a 1 % relative error at
//!   the focal peak contributes the same as a 1 % relative error in
//!   the noise floor — biased toward fitting the volumetric "mostly
//!   zero" rim and under-fitting the peak.
//!
//! * [`TargetTransform::SignedLog1p`] applies a sign-preserving
//!   logarithmic compression `T(p) = sign(p)·log1p(|p|/p_ε) / T_max`.
//!   This puts equal MSE weight on every order of magnitude, so the
//!   peak and the deep-rim get balanced gradient pressure. Empirically
//!   this closes the residual focal-peak underprediction the C-3..C-7
//!   sweep left at ~75 % of target.
//!
//! Both transforms are **bijections on R**: forward maps Pa → `[-1, 1]`
//! and inverse maps back exactly. Round-trip error is bounded by f32
//! precision and the inverse-`expm1` near-zero behaviour.
//!
//! ## Numerical contract
//!
//! For `SignedLog1p { p_eps_pa, t_max }` parameterised against a per-
//! channel maximum `|p|_max`, `T_max = log1p(|p|_max / p_ε)` so that
//! `T(±|p|_max) = ±1`. The inverse uses `expm1` to recover sub-`p_ε`
//! pressures without catastrophic cancellation.
//!
//! `p_eps_pa` must be strictly positive; a typical choice is
//! `p_eps = |p|_max · 10⁻⁶` which gives `T_max ≈ ln(10⁶+1) ≈ 13.8` and
//! distributes loss approximately uniformly across six orders of
//! magnitude — adequate for histotripsy fields whose useful dynamic
//! range is ~`30 MPa → 30 Pa`.

// Used only by the test module (reference pressure-scale assertions).
#[cfg(test)]
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Per-channel forward/inverse mapping between physical Pa and the
/// network's `[-1, 1]` output space.
#[derive(Debug, Clone, Copy)]
pub enum TargetTransform {
    /// `T(p) = (p / scale_pa).clamp(-1, 1)`; `T⁻¹(t) = t · scale_pa`.
    /// Reproduces the pre-C-8 behaviour: simple division by a per-
    /// channel maximum.
    Linear { scale_pa: f32 },
    /// `T(p) = sign(p) · log1p(|p| / p_eps_pa) / t_max`;
    /// `T⁻¹(t) = sign(t) · p_eps_pa · expm1(|t| · t_max)`.
    /// `t_max` is precomputed so that `T(p_max_pa) = 1`.
    SignedLog1p { p_eps_pa: f32, t_max: f32 },
}

impl TargetTransform {
    /// Build a linear transform.
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when `scale_pa <= 0`.
    pub fn linear(scale_pa: f32) -> KwaversResult<Self> {
        // Reject non-positive *and* NaN inputs: `partial_cmp` returns `None`
        // for NaN, which is correctly excluded by the `!= Some(Greater)` test.
        if scale_pa.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) {
            return Err(KwaversError::InvalidInput(
                "TargetTransform::linear requires scale_pa > 0".into(),
            ));
        }
        Ok(Self::Linear { scale_pa })
    }

    /// Build a signed-log1p transform from a per-channel maximum
    /// pressure and a positive ε floor.
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when either input is
    /// non-positive.
    pub fn signed_log1p(p_max_pa: f32, p_eps_pa: f32) -> KwaversResult<Self> {
        // Reject non-positive *and* NaN inputs: `partial_cmp` returns `None`
        // for NaN, which is correctly excluded by the `!= Some(Greater)` test.
        if p_max_pa.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater)
            || p_eps_pa.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater)
        {
            return Err(KwaversError::InvalidInput(
                "TargetTransform::signed_log1p requires both p_max_pa and p_eps_pa > 0".into(),
            ));
        }
        let t_max = (1.0 + p_max_pa / p_eps_pa).ln();
        Ok(Self::SignedLog1p { p_eps_pa, t_max })
    }

    /// Map a physical Pa value into the network's `[-1, 1]` output
    /// space. Inputs outside the calibrated range are clamped.
    #[inline]
    #[must_use]
    pub fn forward(&self, p_pa: f32) -> f32 {
        match *self {
            Self::Linear { scale_pa } => (p_pa / scale_pa).clamp(-1.0, 1.0),
            Self::SignedLog1p { p_eps_pa, t_max } => {
                let s = p_pa.signum();
                let mag = p_pa.abs();
                let t = (mag / p_eps_pa).ln_1p() / t_max;
                (s * t).clamp(-1.0, 1.0)
            }
        }
    }

    /// Map a `[-1, 1]` network output back to physical Pa.
    ///
    /// `t_norm` is clamped to `[-1, 1]` before inversion: the network's
    /// output layer is a plain affine map with no bounding activation
    /// (see [`super::network`]), so an untrained or adversarial network
    /// can emit magnitudes far outside the calibrated domain, which
    /// would otherwise overflow `expm1` to `±inf`.
    #[inline]
    #[must_use]
    pub fn inverse(&self, t_norm: f32) -> f32 {
        let t_norm = t_norm.clamp(-1.0, 1.0);
        match *self {
            Self::Linear { scale_pa } => t_norm * scale_pa,
            Self::SignedLog1p { p_eps_pa, t_max } => {
                let s = t_norm.signum();
                let mag = t_norm.abs();
                s * p_eps_pa * (mag * t_max).exp_m1()
            }
        }
    }
}

/// Per-channel transforms for the `(p_min, p_max, p_rms)` output tuple.
#[derive(Debug, Clone, Copy)]
pub struct OutputTransforms {
    pub p_min: TargetTransform,
    pub p_max: TargetTransform,
    pub p_rms: TargetTransform,
}

impl OutputTransforms {
    /// Build a linear transform for every channel. Mirrors the pre-C-8
    /// behaviour where each channel is divided by its own Pa scale.
    /// # Errors
    /// Propagates [`TargetTransform::linear`] errors.
    pub fn linear(p_min_pa: f32, p_max_pa: f32, p_rms_pa: f32) -> KwaversResult<Self> {
        Ok(Self {
            p_min: TargetTransform::linear(p_min_pa)?,
            p_max: TargetTransform::linear(p_max_pa)?,
            p_rms: TargetTransform::linear(p_rms_pa)?,
        })
    }

    /// Build a signed-log1p transform for every channel from per-
    /// channel maxima and a shared ε ratio so each channel computes
    /// its own `p_eps_pa = p_max_pa · eps_ratio`.
    /// # Errors
    /// Propagates [`TargetTransform::signed_log1p`] errors.
    pub fn signed_log1p(
        p_min_pa: f32,
        p_max_pa: f32,
        p_rms_pa: f32,
        eps_ratio: f32,
    ) -> KwaversResult<Self> {
        if !(eps_ratio > 0.0 && eps_ratio < 1.0) {
            return Err(KwaversError::InvalidInput(
                "OutputTransforms::signed_log1p requires 0 < eps_ratio < 1".into(),
            ));
        }
        Ok(Self {
            p_min: TargetTransform::signed_log1p(p_min_pa, p_min_pa * eps_ratio)?,
            p_max: TargetTransform::signed_log1p(p_max_pa, p_max_pa * eps_ratio)?,
            p_rms: TargetTransform::signed_log1p(p_rms_pa, p_rms_pa * eps_ratio)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_round_trips_within_clamp_range() {
        let scale = 30.0 * MPA_TO_PA as f32;
        let t = TargetTransform::linear(scale).unwrap();
        for &p in &[
            -30.0 * MPA_TO_PA as f32,
            -MPA_TO_PA as f32,
            0.0,
            MPA_TO_PA as f32,
            30.0 * MPA_TO_PA as f32,
        ] {
            let n = t.forward(p);
            assert!(n.abs() <= 1.0 + 1e-6);
            let back = t.inverse(n);
            assert!(
                (back - p).abs() < 1.0,
                "linear round-trip mismatch: {p} → {n} → {back}"
            );
        }
    }

    #[test]
    fn signed_log1p_endpoints_map_to_unit_interval() {
        let p_max = 30.0 * MPA_TO_PA as f32;
        let t = TargetTransform::signed_log1p(p_max, 30.0).unwrap();
        assert!((t.forward(p_max) - 1.0).abs() < 1e-5);
        assert!((t.forward(-p_max) + 1.0).abs() < 1e-5);
        assert!(t.forward(0.0).abs() < 1e-12);
    }

    #[test]
    fn signed_log1p_round_trip_preserves_pressure() {
        let p_max = 30.0 * MPA_TO_PA as f32;
        let t = TargetTransform::signed_log1p(p_max, 30.0).unwrap();
        // Six orders of magnitude in |p|.
        for &p in &[
            -3.0e7_f32, -3.0e6, -3.0e5, -3.0e4, -3.0e3, -3.0e2, -30.0, 0.0, 30.0, 3.0e2, 3.0e3,
            3.0e4, 3.0e5, 3.0e6, 3.0e7,
        ] {
            let n = t.forward(p);
            let back = t.inverse(n);
            let rel = if p.abs() > 0.0 {
                (back - p).abs() / p.abs()
            } else {
                back.abs()
            };
            assert!(
                rel < 1.0e-4,
                "signed-log1p round-trip violated for p={p}: n={n}, back={back}, rel={rel}"
            );
        }
    }

    #[test]
    fn signed_log1p_is_monotonic() {
        let p_max = 30.0 * MPA_TO_PA as f32;
        let t = TargetTransform::signed_log1p(p_max, 30.0).unwrap();
        let mut prev = t.forward(-1.0e8);
        for k in -10..=10 {
            let p = (k as f32) * MPA_TO_PA as f32;
            let n = t.forward(p);
            assert!(
                n >= prev - 1e-7,
                "non-monotonic at p={p}: prev={prev}, n={n}"
            );
            prev = n;
        }
    }

    #[test]
    fn signed_log1p_compresses_dynamic_range() {
        // At |p| = p_eps, the linear transform yields ≈ p_eps/p_max
        // (tiny); the log1p transform yields ≈ ln(2)/t_max — a
        // substantial fraction of unity, demonstrating dynamic-range
        // compression.
        let p_max = 30.0 * MPA_TO_PA as f32;
        let p_eps = 30.0_f32;
        let lin = TargetTransform::linear(p_max).unwrap();
        let log = TargetTransform::signed_log1p(p_max, p_eps).unwrap();
        let n_lin = lin.forward(p_eps);
        let n_log = log.forward(p_eps);
        assert!(
            n_lin < 1.0e-5,
            "linear should produce near-zero norm at p_eps: {n_lin}"
        );
        assert!(
            n_log > 0.01,
            "log1p should lift p_eps to a non-trivial norm: {n_log}"
        );
    }

    #[test]
    fn output_transforms_signed_log1p_rejects_invalid_eps_ratio() {
        let p30 = 30.0 * MPA_TO_PA as f32;
        let p21 = 21.0 * MPA_TO_PA as f32;
        assert!(OutputTransforms::signed_log1p(p30, p30, p21, 0.0).is_err());
        assert!(OutputTransforms::signed_log1p(p30, p30, p21, 1.0).is_err());
        assert!(OutputTransforms::signed_log1p(p30, p30, p21, -0.1).is_err());
        assert!(OutputTransforms::signed_log1p(p30, p30, p21, 1.0e-6).is_ok());
    }

    #[test]
    fn linear_rejects_non_positive_scale() {
        assert!(TargetTransform::linear(0.0).is_err());
        assert!(TargetTransform::linear(-1.0).is_err());
    }
}
