//! Far-field patch model of a rectangular piston, evaluated by sparse delta
//! integration (SDI).
//!
//! [`RectangularPistonSir`](super::RectangularPistonSir) is the exact
//! Lockwood–Willette spatial impulse response (SIR). This module is the model
//! Field II uses by default for rectangular elements and the one Rivera,
//! Demené & Tanter (2026) accelerate: the element is tiled into `n_x × n_y`
//! rectangular patches, each small enough that the wavefront reaching it is
//! planar, and each patch's SIR is then a trapezoid with a closed form.
//!
//! # Closed form (one patch)
//!
//! For a patch of full widths `w_x × w_y` centred at `c_m` and a field point
//! `p` at distance `l = |p − c_m|` along the unit vector `u = (p − c_m)/l`
//! with in-plane components `(u_x, u_y)`:
//!
//! ```text
//! Δt₁ = min(w_x|u_x|, w_y|u_y|)/c        Δt₂ = max(w_x|u_x|, w_y|u_y|)/c
//! t₁ = l/c − (Δt₁ + Δt₂)/2   t₂ = t₁ + Δt₁   t₃ = t₁ + Δt₂   t₄ = t₁ + Δt₁ + Δt₂
//! A = w_x·w_y/(2π·l)          h_max = A/Δt₂          s = h_max/Δt₁
//!
//!        ⎧ s·(t − t₁)      t₁ ≤ t < t₂
//! h(t) = ⎨ h_max           t₂ ≤ t < t₃
//!        ⎪ s·(t₄ − t)      t₃ ≤ t < t₄
//!        ⎩ 0               otherwise
//! ```
//!
//! (Rivera, Demené & Tanter 2026, Eqs. 5–14; Jensen & Svendsen 1992.) The
//! trapezoid is the exact SIR of the patch once the spherical wavefront's
//! trace on the aperture plane is replaced by a straight line. The distance
//! term that replacement drops is bounded by the sagitta `(w_x² + w_y²)/(8l)`,
//! and the model is valid when the far-field number `w²·f/(4·l·c)` is small
//! (Eq. 22 of the paper) — see [`FarFieldRectangleSir::far_field_number`].
//!
//! # Sparse delta integration
//!
//! The trapezoid's second time derivative is four deltas,
//! `h'' = s·[δ(t−t₁) − δ(t−t₂) − δ(t−t₃) + δ(t−t₄)]`, so the element's SIR is
//! the double integral of the `4·n_x·n_y` deltas of all its patches: each
//! delta is split between the two grid nodes bracketing it with linear
//! weights, the train is summed cumulatively twice, and the cost per patch is
//! eight additions whatever the trapezoid's width in samples (Eqs. 19–21 and
//! 28–34 of the paper).
//!
//! The split is not an approximation at the nodes. A delta of weight `w` at
//! fractional index `k` contributes `w·(t_n − t_k)` to every node past it and
//! nothing before; splitting it as `(1−f)` at `⌊k⌋` and `f` at `⌈k⌉` keeps its
//! zeroth and first moments, and a double cumulative sum — inclusive for the
//! first, exclusive for the second — reproduces exactly that ramp at every
//! node. The one-way SIR is therefore the trapezoid sampled at every grid
//! node, to rounding; a corner before the first node still splits and cancels
//! correctly, its early ramp simply having no node to appear on. The
//! exactness stops at two integrations (the split changes the second and
//! third moments), which is why the round trip is the direct auto-convolution
//! of the one-way kernel rather than an integrated sixteen-delta train.
//!
//! # Degenerate patches
//!
//! `Δt₁ = 0` (the field point lies in one of the patch's symmetry planes)
//! collapses the trapezoid to a rectangle, and `Δt₂ = 0` (on the patch axis)
//! to `A·δ(t − l/c)`; neither has a finite `s`. These are deposited directly
//! as bin averages — the rectangle's overlap with each bin, the delta as `A/dt`
//! in its bin — so their area at `t ≥ 0` is preserved exactly. Between those
//! exact cases
//! a small `Δt₁` costs nothing at the nodes; its rounding is bounded by
//! `ε·dt/Δt₁` relative to `h_max`.
//!
//! # Sampling convention
//!
//! Node `n` sits at `(n + ½)·dt`, the bin-midpoint convention of the exact
//! kernels in the parent module, so the two families sample the same instants.
//!
//! # References
//! - Rivera, D. E., Demené, C., & Tanter, M. (2026). "Sparse Delta Integration
//!   method for the calculation of spatiotemporal pressure fields of arbitrary
//!   ultrasound transducer geometries." arXiv:2608.26891.
//! - Jensen, J. A., & Svendsen, N. B. (1992). "Calculation of pressure fields
//!   from arbitrarily shaped, apodized, and excited ultrasound transducers."
//!   *IEEE Trans. UFFC* 39(2), 262–267 — the far-field rectangle in Field II.

use super::{auto_convolve, SampledResponse};
use kwavers_core::error::{KwaversError, KwaversResult};
use std::f64::consts::PI;
use std::num::NonZeroUsize;

/// A flat rectangular piston in an infinite rigid baffle, centred on the
/// axis with half-widths `wx`, `wy`, tiled into `n_x × n_y` far-field patches.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FarFieldRectangleSir {
    half_width_x: f64,
    half_width_y: f64,
    patches_x: NonZeroUsize,
    patches_y: NonZeroUsize,
    sound_speed: f64,
}

/// One patch's far-field trapezoid at a field point.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Trapezoid {
    /// `t₁`, the first arrival \[s].
    onset: f64,
    /// `Δt₁`, the ramp duration \[s].
    ramp: f64,
    /// `Δt₂`, onset to the start of the fall \[s].
    span: f64,
    /// `A = w_x·w_y/(2π·l)`, the area `∫h dt` \[m].
    area: f64,
}

impl Trapezoid {
    /// `t₄`, the last arrival.
    fn end(&self) -> f64 {
        self.onset + self.ramp + self.span
    }
}

impl FarFieldRectangleSir {
    /// Create a far-field patch model from the element **half**-widths and the
    /// patch counts along each axis.
    ///
    /// # Errors
    /// - `KwaversError::InvalidInput` if any half-width or `sound_speed` is
    ///   non-finite or `≤ 0`.
    pub fn new(
        half_width_x: f64,
        half_width_y: f64,
        patches: [NonZeroUsize; 2],
        sound_speed: f64,
    ) -> KwaversResult<Self> {
        for (name, v) in [
            ("half_width_x", half_width_x),
            ("half_width_y", half_width_y),
            ("sound_speed", sound_speed),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "FarFieldRectangleSir requires {name} > 0, got {v}"
                )));
            }
        }
        Ok(Self {
            half_width_x,
            half_width_y,
            patches_x: patches[0],
            patches_y: patches[1],
            sound_speed,
        })
    }

    /// Full widths `[w_x, w_y]` of one patch \[m].
    #[must_use]
    pub fn patch_widths(&self) -> [f64; 2] {
        [
            2.0 * self.half_width_x / self.patches_x.get() as f64,
            2.0 * self.half_width_y / self.patches_y.get() as f64,
        ]
    }

    /// Far-field number `w²·f/(4·l·c)` of the larger patch width at distance
    /// `l` and frequency `f` (Rivera, Demené & Tanter 2026, Eq. 22).
    ///
    /// The trapezoid model holds when this is small: it is the ratio of the
    /// dropped sagitta term `w²/(8l)` to the half-wavelength `c/(2f)`, so at
    /// `0.1` the arrival-time error across a patch is a twentieth of a period.
    #[must_use]
    pub fn far_field_number(&self, distance: f64, frequency: f64) -> f64 {
        let [wx, wy] = self.patch_widths();
        let w = wx.max(wy);
        w * w * frequency / (4.0 * distance * self.sound_speed)
    }

    /// First-arrival time \[s] at `(x, y, z)` — the earliest patch onset `t₁`.
    #[must_use]
    pub fn first_arrival_time(&self, x: f64, y: f64, z: f64) -> f64 {
        self.trapezoids(x, y, z)
            .map(|t| t.onset)
            .fold(f64::INFINITY, f64::min)
    }

    /// Last-arrival time \[s] at `(x, y, z)` — the latest patch end `t₄`.
    #[must_use]
    pub fn last_arrival_time(&self, x: f64, y: f64, z: f64) -> f64 {
        self.trapezoids(x, y, z)
            .map(|t| t.end())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// One-way spatial impulse response `h(x, y, z, t)` \[m/s], sampled at the
    /// bin midpoints over its support by sparse delta integration.
    ///
    /// `(x, y)` is the field point's projection onto the aperture plane, `z`
    /// its distance from the plane (the sign is immaterial). The result is the
    /// sum of the patch trapezoids sampled at every node `n ≥ 0` — exact to
    /// rounding — with the degenerate patches deposited as bin averages
    /// (module docs). A support that begins before `t = 0` has no node to
    /// carry its early part: the values at the nodes that exist are still the
    /// trapezoid's, and a degenerate rectangle keeps the area it has at
    /// `t ≥ 0`. Such a field point lies within a patch width of the face and
    /// is far outside the model's validity in any case
    /// ([`Self::far_field_number`]). A field point at a patch centre in the
    /// plane (`l = 0`) is outside the model: the trapezoid has no finite
    /// corner times, and the response is a single `NaN` sample so that a
    /// consumer's non-finite check reports it rather than a silent fallback.
    ///
    /// # Panics
    /// Panics if `dt ≤ 0` (a non-positive sample step is a caller bug).
    #[must_use]
    pub fn response(&self, x: f64, y: f64, z: f64, dt: f64) -> SampledResponse {
        assert!(dt > 0.0, "response requires dt > 0, got {dt}");
        let node = |t: f64| t / dt - 0.5;

        let t_first = self.first_arrival_time(x, y, z);
        let t_last = self.last_arrival_time(x, y, z);
        // `l = 0` makes a patch's corner times NaN, which the min/max folds
        // discard; the support is then unbounded and the model void.
        if !t_first.is_finite() || !t_last.is_finite() {
            return SampledResponse {
                first_sample: 0,
                samples: vec![f64::NAN],
            };
        }
        // The buffer is anchored at the true floor node of the earliest onset,
        // signed, so a corner before the first grid node still splits and
        // cancels correctly; one node of margin below it, two above the
        // latest end (a delta's ceiling node and its successor).
        let first_bin = node(t_first).floor() as isize - 1;
        let end_bin = node(t_last).ceil() as isize + 2;
        let len = (end_bin - first_bin) as usize;
        let local = |bin: isize| (bin - first_bin).clamp(0, len as isize - 1) as usize;

        // Split deltas of the trapezoids' second derivatives, and the
        // bin-averaged deposits of the degenerate patches.
        let mut deltas = vec![0.0_f64; len];
        let mut direct = vec![0.0_f64; len];
        // First node past every patch's support: analytically zero from here.
        let mut support_end = first_bin;
        for trap in self.trapezoids(x, y, z) {
            if trap.span <= 0.0 {
                // On the patch axis: `A·δ(t − l/c)` as `A/dt` in its bin.
                let bin = (trap.onset / dt).floor() as isize;
                direct[local(bin)] += trap.area / dt;
                support_end = support_end.max(bin + 1);
            } else if trap.ramp <= 0.0 {
                // In a symmetry plane: a rectangle of height `A/Δt₂` over
                // `[t₁, t₃]`, each bin taking its overlap.
                let height = trap.area / trap.span;
                let (t_lo, t_hi) = (trap.onset, trap.onset + trap.span);
                let bin_lo = (t_lo / dt).floor() as isize;
                let bin_hi = (t_hi / dt).floor() as isize;
                for bin in bin_lo..=bin_hi {
                    let lo = (bin as f64 * dt).max(t_lo);
                    let hi = ((bin + 1) as f64 * dt).min(t_hi);
                    if hi > lo {
                        direct[local(bin)] += height * (hi - lo) / dt;
                    }
                }
                support_end = support_end.max(bin_hi + 1);
            } else {
                let slope = trap.area / (trap.ramp * trap.span);
                let corners = [
                    (trap.onset, slope),
                    (trap.onset + trap.ramp, -slope),
                    (trap.onset + trap.span, -slope),
                    (trap.end(), slope),
                ];
                for (t, weight) in corners {
                    let k = node(t);
                    let floor = k.floor();
                    let fraction = k - floor;
                    let at = local(floor as isize);
                    deltas[at] += weight * (1.0 - fraction);
                    deltas[at + 1] += weight * fraction;
                }
                support_end = support_end.max(node(trap.end()).ceil() as isize);
            }
        }

        // h[n] = dt·Σ_{j<n} g[j] with g the inclusive cumulative sum of the
        // deltas: the double integral that is exact at the nodes.
        let mut samples = vec![0.0_f64; len];
        let mut slope_sum = 0.0_f64;
        let mut ramp_sum = 0.0_f64;
        for (n, delta) in deltas.iter().enumerate() {
            samples[n] = ramp_sum * dt + direct[n];
            slope_sum += delta;
            ramp_sum += slope_sum;
        }
        // Past the support the four-delta sums cancel analytically; zero the
        // rounding residue so the kernel ends where the physics does.
        let tail = local(support_end);
        for sample in &mut samples[tail..] {
            *sample = 0.0;
        }
        // Nodes before index 0 do not exist on the grid.
        let representable = local(0);
        let Some(first) = samples[representable..]
            .iter()
            .position(|v| *v != 0.0)
            .map(|i| i + representable)
        else {
            return SampledResponse {
                first_sample: 0,
                samples: Vec::new(),
            };
        };
        let last = samples
            .iter()
            .rposition(|v| *v != 0.0)
            .expect("invariant: a non-zero sample exists, so a last one does");
        samples.truncate(last + 1);
        samples.drain(..first);
        SampledResponse {
            first_sample: (first_bin + first as isize) as usize,
            samples,
        }
    }

    /// Two-way (monostatic pulse-echo) diffraction kernel `(h ⊛ h)(t)` at
    /// `(x, y, z)`, the direct auto-convolution of [`Self::response`] over its
    /// support; the same contract as
    /// [`CircularPistonSir::round_trip_response`](super::CircularPistonSir::round_trip_response).
    ///
    /// # Panics
    /// Panics if `dt ≤ 0` (a non-positive sample step is a caller bug).
    #[must_use]
    pub fn round_trip_response(&self, x: f64, y: f64, z: f64, dt: f64) -> SampledResponse {
        let one_way = self.response(x, y, z, dt);
        SampledResponse {
            first_sample: 2 * one_way.first_sample,
            samples: auto_convolve(&one_way.samples, dt),
        }
    }

    /// The far-field trapezoid of every patch at `(x, y, z)`, row-major over
    /// the `n_x × n_y` tiling.
    fn trapezoids(&self, x: f64, y: f64, z: f64) -> impl Iterator<Item = Trapezoid> + '_ {
        let [wx, wy] = self.patch_widths();
        let (nx, ny) = (self.patches_x.get(), self.patches_y.get());
        let c = self.sound_speed;
        (0..nx * ny).map(move |index| {
            let (ix, iy) = (index / ny, index % ny);
            // Patch centre in the aperture plane.
            let cx = -self.half_width_x + (ix as f64 + 0.5) * wx;
            let cy = -self.half_width_y + (iy as f64 + 0.5) * wy;
            let (dx, dy) = (x - cx, y - cy);
            let l = (dx * dx + dy * dy + z * z).sqrt();
            let across_x = wx * dx.abs() / (l * c);
            let across_y = wy * dy.abs() / (l * c);
            let (ramp, span) = (across_x.min(across_y), across_x.max(across_y));
            Trapezoid {
                onset: l / c - 0.5 * (ramp + span),
                ramp,
                span,
                area: wx * wy / (2.0 * PI * l),
            }
        })
    }
}

#[cfg(test)]
mod tests;
