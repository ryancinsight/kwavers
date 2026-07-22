//! Transient spatial impulse response (SIR) of flat pistons (circular & rectangular).
//!
//! The spatial impulse response `h(r, z, t)` is the velocity-potential response
//! at a field point to an impulsive uniform normal velocity of the aperture — the
//! transient (broadband) analogue of the Fast Nearfield Method's continuous-wave
//! field ([`crate`] has the CW path; this is the impulse/pulse-echo path). It is
//! the diffraction kernel of the Field II model: the radiated pressure is
//! `p(t) = ρ ∂/∂t [ v(t) ⊛ h(t) ]`, and the pulse-echo response convolves the
//! transmit and receive SIRs with the excitation.
//!
//! # Closed form (flat circular piston, radius `a`)
//!
//! For a field point at axial distance `z > 0` and lateral offset `r ≥ 0` from the
//! axis, with `ρ = c·t` the radius of the expanding spherical wavefront's
//! intersection circle (`ρ² = (ct)² − z²`):
//!
//! ```text
//!            ⎧ 0                                              ct < d_min
//!            ⎪ c                                  d_min ≤ ct < d_plateau   (only r < a)
//! h(r,z,t) = ⎨ (c/π)·arccos[ ((ct)²−z²+r²−a²) / (2 r √((ct)²−z²)) ]
//!            ⎪                                d_plateau ≤ ct < d_max
//!            ⎩ 0                                              ct ≥ d_max
//! ```
//!
//! with `d_min = z` if `r ≤ a` else `√(z²+(r−a)²)`,
//! `d_plateau = √(z²+(a−r)²)`, and `d_max = √(z²+(a+r)²)`.
//! On axis (`r = 0`) this reduces to the rectangular pulse `h = c` over
//! `z/c ≤ t < √(z²+a²)/c`.
//!
//! # Closed form (flat rectangular piston, half-widths `wx`, `wy`)
//!
//! For a field point `(x, y, z)` whose projection onto the aperture plane is
//! `(x, y)`, the SIR is `h = (c/2π)·Φ(ρ)`, where `ρ = √((ct)²−z²)` is the radius
//! of the wavefront's intersection circle (centered at the projection) and `Φ` is
//! the **angular measure** of that circle lying within the rectangle
//! `[−wx,wx]×[−wy,wy]`. A point on the circle, `(x+ρcosθ, y+ρsinθ)`, is inside iff
//! `cosθ ∈ [(−wx−x)/ρ, (wx−x)/ρ]` and `sinθ ∈ [(−wy−y)/ρ, (wy−y)/ρ]`; `Φ` is the
//! measure of the `θ` satisfying both bands, evaluated exactly from the
//! `arccos`/`arcsin` breakpoints (Lockwood & Willette 1973). Circle fully inside
//! ⇒ `Φ = 2π ⇒ h = c` (the near-field plateau).
//!
//! # References
//! - Stepanishen, P. R. (1971). "Transient radiation from pistons in an infinite
//!   planar baffle." *J. Acoust. Soc. Am.* 49(5B), 1629–1638.
//! - Lockwood, J. C., & Willette, J. G. (1973). "High-speed method for computing
//!   the exact solution for the pressure variations in the nearfield of a baffled
//!   piston." *J. Acoust. Soc. Am.* 53(3), 735–741. — rectangular piston SIR.
//! - Jensen, J. A. (1999). "A new calculation procedure for spatial impulse
//!   responses in ultrasound." *J. Acoust. Soc. Am.* 105(6), 3266–3274.

use kwavers_core::error::{KwaversError, KwaversResult};
use std::f64::consts::PI;

/// A flat circular piston in an infinite rigid baffle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CircularPistonSir {
    radius: f64,
    sound_speed: f64,
}

impl CircularPistonSir {
    /// Create a circular-piston SIR model.
    ///
    /// # Errors
    /// - `KwaversError::InvalidInput` if `radius` or `sound_speed` is
    ///   non-finite or `≤ 0`.
    pub fn new(radius: f64, sound_speed: f64) -> KwaversResult<Self> {
        if !radius.is_finite() || radius <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CircularPistonSir requires radius > 0, got {radius}"
            )));
        }
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CircularPistonSir requires sound_speed > 0, got {sound_speed}"
            )));
        }
        Ok(Self {
            radius,
            sound_speed,
        })
    }

    /// First-arrival time `s` at field point `(r, z)` — the nearest aperture point.
    #[must_use]
    pub fn first_arrival_time(&self, r: f64, z: f64) -> f64 {
        let d_min = if r <= self.radius {
            z.abs()
        } else {
            (z * z + (r - self.radius).powi(2)).sqrt()
        };
        d_min / self.sound_speed
    }

    /// Last-arrival time `s` at field point `(r, z)` — the farthest rim point.
    #[must_use]
    pub fn last_arrival_time(&self, r: f64, z: f64) -> f64 {
        (z * z + (r + self.radius).powi(2)).sqrt() / self.sound_speed
    }

    /// Spatial impulse response `h(r, z, t)` [m/s] at a field point.
    ///
    /// `r ≥ 0` is the lateral offset from the axis, `z > 0` the axial distance,
    /// `t` the time `s`. Returns `0` outside the support `[d_min/c, d_max/c]`.
    #[must_use]
    pub fn evaluate(&self, r: f64, z: f64, t: f64) -> f64 {
        let a = self.radius;
        let c = self.sound_speed;
        let z = z.abs();
        let ct = c * t;

        // Wavefront intersection-circle radius ρ with the aperture plane.
        let rho_sq = ct * ct - z * z;
        if rho_sq <= 0.0 {
            return 0.0; // wavefront has not reached the aperture plane
        }

        let d_min = self.first_arrival_time(r, z) * c;
        let d_max = self.last_arrival_time(r, z) * c;
        if ct < d_min || ct >= d_max {
            return 0.0;
        }

        // On-axis: full-circle plateau over the whole support.
        if r == 0.0 {
            return c;
        }

        // Off-axis: plateau (full circle inside the piston) only when r < a and
        // the wavefront circle still lies entirely within the aperture.
        let d_plateau = (z * z + (a - r).powi(2)).sqrt();
        if r < a && ct < d_plateau {
            return c;
        }

        // Partial-arc region: the wavefront circle intersects the piston rim.
        // arg = (ρ² + r² − a²) / (2 r ρ), clamped for FP safety.
        let rho = rho_sq.sqrt();
        let arg = ((rho_sq + r * r - a * a) / (2.0 * r * rho)).clamp(-1.0, 1.0);
        (c / PI) * arg.acos()
    }

    /// Two-way (monostatic pulse-echo) diffraction kernel `(h ⊛ h)(t)` at field
    /// point `(r, z)`, sampled at step `dt` over `[0, n_samples·dt)`.
    ///
    /// For an element that both transmits and receives, the pulse-echo spatial
    /// response is the convolution of the transmit and receive SIRs; with a single
    /// aperture `h_tx = h_rx = h`, so the diffraction part is `h ⊛ h` (Jensen 1991).
    /// Convolving this with the electrical excitation gives the Field II echo — the
    /// finite-aperture refinement of the point-element `1/r²` model. The one-way
    /// SIR is sampled at bin midpoints and discretely auto-convolved.
    ///
    /// To capture the full kernel choose `n_samples ≥ ⌈2·last_arrival_time/dt⌉`
    /// (the two-way support ends at `2·d_max/c`). The convolution integral
    /// factorizes, `∫(h⊛h)dt = (∫h dt)²`, and on-axis `∫h dt = √(z²+a²) − z`, so
    /// `Σ_k out`K`·dt = (√(z²+a²) − z)²` — the exact normalization.
    ///
    /// # Panics
    /// Panics if `dt ≤ 0` (a non-positive sample step is a caller bug).
    #[must_use]
    pub fn round_trip_response(&self, r: f64, z: f64, dt: f64, n_samples: usize) -> Vec<f64> {
        assert!(dt > 0.0, "round_trip_response requires dt > 0, got {dt}");
        // One-way SIR sampled at bin midpoints on the dt grid.
        let h: Vec<f64> = (0..n_samples)
            .map(|k| self.evaluate(r, z, (k as f64 + 0.5) * dt))
            .collect();
        // Discrete auto-convolution (h ⊛ h)(k·dt) ≈ Σ_i h[i]·h[k−i]·dt, truncated.
        let mut out = vec![0.0_f64; n_samples];
        for (i, &hi) in h.iter().enumerate() {
            if hi == 0.0 {
                continue;
            }
            for (j, &hj) in h.iter().enumerate().take(n_samples - i) {
                out[i + j] += hi * hj * dt;
            }
        }
        out
    }
}

/// A flat rectangular piston in an infinite rigid baffle, centered on the axis
/// with half-widths `wx`, `wy` (full aperture `2·wx × 2·wy`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RectangularPistonSir {
    half_width_x: f64,
    half_width_y: f64,
    sound_speed: f64,
}

impl RectangularPistonSir {
    /// Create a rectangular-piston SIR model from the **half**-widths.
    ///
    /// # Errors
    /// - `KwaversError::InvalidInput` if any half-width or `sound_speed` is
    ///   non-finite or `≤ 0`.
    pub fn new(half_width_x: f64, half_width_y: f64, sound_speed: f64) -> KwaversResult<Self> {
        for (name, v) in [
            ("half_width_x", half_width_x),
            ("half_width_y", half_width_y),
            ("sound_speed", sound_speed),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "RectangularPistonSir requires {name} > 0, got {v}"
                )));
            }
        }
        Ok(Self {
            half_width_x,
            half_width_y,
            sound_speed,
        })
    }

    /// First-arrival time `s` at `(x, y, z)` — the nearest aperture point (the
    /// projection clamped into the rectangle).
    #[must_use]
    pub fn first_arrival_time(&self, x: f64, y: f64, z: f64) -> f64 {
        let cx = x.clamp(-self.half_width_x, self.half_width_x);
        let cy = y.clamp(-self.half_width_y, self.half_width_y);
        (z * z + (x - cx).powi(2) + (y - cy).powi(2)).sqrt() / self.sound_speed
    }

    /// Last-arrival time `s` at `(x, y, z)` — the farthest aperture corner.
    #[must_use]
    pub fn last_arrival_time(&self, x: f64, y: f64, z: f64) -> f64 {
        let dx = (x - self.half_width_x)
            .abs()
            .max((x + self.half_width_x).abs());
        let dy = (y - self.half_width_y)
            .abs()
            .max((y + self.half_width_y).abs());
        (z * z + dx * dx + dy * dy).sqrt() / self.sound_speed
    }

    /// Spatial impulse response `h(x, y, z, t)` [m/s] at a field point.
    ///
    /// `(x, y)` is the lateral position (projection onto the aperture plane), `z`
    /// the axial distance, `t` the time `s`. Returns `0` outside the support.
    #[must_use]
    pub fn evaluate(&self, x: f64, y: f64, z: f64, t: f64) -> f64 {
        let c = self.sound_speed;
        let z = z.abs();
        let ct = c * t;
        let rho_sq = ct * ct - z * z;
        if rho_sq <= 0.0 {
            return 0.0; // wavefront has not reached the aperture plane
        }
        let rho = rho_sq.sqrt();
        (c / (2.0 * PI)) * self.arc_angle_inside(x, y, rho)
    }

    /// Angular measure `Φ ∈ [0, 2π]` of the circle of radius `rho` centered at
    /// `(x, y)` that lies within the rectangle `[−wx,wx]×[−wy,wy]`.
    ///
    /// A point `(x+ρcosθ, y+ρsinθ)` is inside iff `cosθ ∈ [cxlo, cxhi]` and
    /// `sinθ ∈ [sylo, syhi]`. The membership can only change at the `θ` where one
    /// of these four bounds is hit (exact `arccos`/`arcsin` roots); between
    /// consecutive breakpoints membership is constant, so `Φ` is the sum of the
    /// breakpoint intervals whose midpoint is inside both bands.
    fn arc_angle_inside(&self, x: f64, y: f64, rho: f64) -> f64 {
        let (wx, wy) = (self.half_width_x, self.half_width_y);
        // Band bounds (may fall outside [-1,1] ⇒ that side is unconstrained).
        let cx_lo = (-wx - x) / rho;
        let cx_hi = (wx - x) / rho;
        let sy_lo = (-wy - y) / rho;
        let sy_hi = (wy - y) / rho;

        // Critical angles where membership can change, plus the [0, 2π] ends.
        let mut breakpoints = vec![0.0_f64, 2.0 * PI];
        for &v in &[cx_lo, cx_hi] {
            if (-1.0..=1.0).contains(&v) {
                let a = v.acos(); // cosθ = v ⇒ θ = ±a (mod 2π)
                breakpoints.push(a);
                breakpoints.push(2.0 * PI - a);
            }
        }
        for &w in &[sy_lo, sy_hi] {
            if (-1.0..=1.0).contains(&w) {
                let a = w.asin(); // sinθ = w ⇒ θ = a (mod 2π) or π − a
                breakpoints.push(a.rem_euclid(2.0 * PI));
                breakpoints.push((PI - a).rem_euclid(2.0 * PI));
            }
        }
        breakpoints.sort_by(|p, q| p.partial_cmp(q).expect("finite breakpoints"));

        let inside = |theta: f64| {
            let (s, cth) = theta.sin_cos();
            cth >= cx_lo && cth <= cx_hi && s >= sy_lo && s <= sy_hi
        };

        let mut phi = 0.0_f64;
        for pair in breakpoints.windows(2) {
            let (lo, hi) = (pair[0], pair[1]);
            if hi - lo <= 0.0 {
                continue;
            }
            if inside(0.5 * (lo + hi)) {
                phi += hi - lo;
            }
        }
        phi
    }
}

#[cfg(test)]
mod tests;