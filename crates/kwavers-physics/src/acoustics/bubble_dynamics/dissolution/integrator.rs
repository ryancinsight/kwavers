//! Numerical integration of a dissolution model's `R(t)` and dissolution time.

use super::traits::DissolutionModel;

/// Time history of a dissolving bubble and its dissolution time.
#[derive(Debug, Clone)]
pub struct DissolutionTrajectory {
    /// Time grid `s`.
    pub time: Vec<f64>,
    /// Bubble radius `R(t)` `m`.
    pub radius: Vec<f64>,
    /// Time at which `R` first falls to `r_dissolved` `s`, or `None` if the
    /// bubble had not dissolved (or was growing) by `t_max`.
    pub dissolution_time: Option<f64>,
}

impl DissolutionTrajectory {
    /// Residual gas volume fraction relative to the initial bubble, `(R/R₀)³`,
    /// at each recorded time — the quantity a residual-gas field accumulates.
    #[must_use]
    pub fn volume_ratio(&self) -> Vec<f64> {
        let r0 = self.radius.first().copied().unwrap_or(0.0).max(1e-300);
        self.radius.iter().map(|&r| (r / r0).powi(3)).collect()
    }
}

/// Integrate a [`DissolutionModel`] from `r0_m` with fixed-step RK4 until the
/// radius falls to `r_dissolved_m` or `t_max_s` is reached.
///
/// The transient diffusion term `1/√(πDt)` is integrable but singular at
/// `t = 0`; the first sub-step is evaluated at `t = dt/2` to keep the rate
/// finite while preserving the early-time contribution. Returns the trajectory
/// and the dissolution time (linearly interpolated at the `r_dissolved`
/// crossing).
#[must_use]
pub fn integrate_dissolution(
    model: &impl DissolutionModel,
    r0_m: f64,
    dt_s: f64,
    t_max_s: f64,
    r_dissolved_m: f64,
) -> DissolutionTrajectory {
    let mut time = Vec::new();
    let mut radius = Vec::new();
    let mut dissolution_time = None;
    if !(r0_m > 0.0 && dt_s > 0.0 && t_max_s > 0.0) {
        return DissolutionTrajectory {
            time,
            radius,
            dissolution_time,
        };
    }

    let rate = |r: f64, t: f64| model.radius_rate(r, t.max(0.5 * dt_s));

    let mut t = 0.0_f64;
    let mut r = r0_m;
    time.push(t);
    radius.push(r);
    let n_max = (t_max_s / dt_s).ceil() as usize;
    for _ in 0..n_max {
        let k1 = rate(r, t);
        let k2 = rate(r + 0.5 * dt_s * k1, t + 0.5 * dt_s);
        let k3 = rate(r + 0.5 * dt_s * k2, t + 0.5 * dt_s);
        let k4 = rate(r + dt_s * k3, t + dt_s);
        let r_next = r + dt_s / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        let t_next = t + dt_s;

        if r_next <= r_dissolved_m && dissolution_time.is_none() {
            // Linear interpolation of the crossing time within [t, t_next].
            let frac = if (r - r_next).abs() > f64::MIN_POSITIVE {
                ((r - r_dissolved_m) / (r - r_next)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            dissolution_time = Some(t + frac * dt_s);
            time.push(t_next);
            radius.push(r_dissolved_m.max(0.0));
            break;
        }

        r = r_next.max(0.0);
        t = t_next;
        time.push(t);
        radius.push(r);
        if r <= 0.0 {
            break;
        }
    }
    DissolutionTrajectory {
        time,
        radius,
        dissolution_time,
    }
}

/// Convenience: the dissolution time `R₀ → r_dissolved` for a model, via
/// integration (captures surface tension and the transient term, unlike the
/// closed-form [`DissolutionModel::dissolution_time`]).
#[must_use]
pub fn dissolution_time_numeric(
    model: &impl DissolutionModel,
    r0_m: f64,
    r_dissolved_m: f64,
) -> Option<f64> {
    // Step ≈ 1/2000 of the closed-form estimate (or a fallback) for accuracy;
    // integrate out to 100× that estimate.
    let est = model
        .dissolution_time(r0_m)
        .unwrap_or(r0_m * r0_m / model.params().rate_scale().max(1e-30));
    let dt = (est / 2000.0).max(1e-9);
    let t_max = (est * 100.0).max(dt * 10.0);
    integrate_dissolution(model, r0_m, dt, t_max, r_dissolved_m).dissolution_time
}
