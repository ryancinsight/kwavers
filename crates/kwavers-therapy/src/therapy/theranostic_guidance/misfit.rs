//! Robust trace misfits for the linear acoustic RTM channel.
//!
//! # Theorem: bounded Charbonnier adjoint source
//!
//! Let `r = d_obs - d_pred` and let `epsilon > 0`. The Charbonnier objective
//!
//! ```text
//! phi(r) = epsilon^2 (sqrt(1 + (r / epsilon)^2) - 1)
//! ```
//!
//! has derivative
//!
//! ```text
//! d phi / d r = r / sqrt(1 + (r / epsilon)^2)
//! ```
//!
//! and therefore `|d phi / d r| <= epsilon`. Proof: set
//! `x = |r| / epsilon`; then `|d phi / d r| = epsilon x / sqrt(1 + x^2)`,
//! which is bounded above by `epsilon` for all finite `x`.
//!
//! This module applies the derivative as the adjoint source for the
//! time-domain RTM residual. It does not change the finite-frequency Born
//! inverse and does not convert the workflow into full-waveform inversion.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveformMisfit {
    L2,
    Charbonnier,
}

impl WaveformMisfit {
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Charbonnier => "charbonnier",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "l2" | "least_squares" => Some(Self::L2),
            "charbonnier" | "robust_charbonnier" => Some(Self::Charbonnier),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TraceResidual {
    pub adjoint_source: Vec<f32>,
    pub objective_value: f64,
    pub scale: f32,
    pub misfit: WaveformMisfit,
}

#[must_use]
pub fn evaluate_trace_residual(
    observed: &[f32],
    predicted: &[f32],
    misfit: WaveformMisfit,
    scale_fraction: f64,
) -> TraceResidual {
    assert_eq!(observed.len(), predicted.len());
    let raw = observed
        .iter()
        .zip(predicted.iter())
        .map(|(obs, pred)| obs - pred)
        .collect::<Vec<_>>();
    match misfit {
        WaveformMisfit::L2 => l2_residual(raw),
        WaveformMisfit::Charbonnier => charbonnier_residual(raw, observed, scale_fraction),
    }
}

fn l2_residual(raw: Vec<f32>) -> TraceResidual {
    let objective_value = raw.iter().map(|value| 0.5 * (*value as f64).powi(2)).sum();
    TraceResidual {
        adjoint_source: raw,
        objective_value,
        scale: 0.0,
        misfit: WaveformMisfit::L2,
    }
}

fn charbonnier_residual(raw: Vec<f32>, observed: &[f32], scale_fraction: f64) -> TraceResidual {
    let scale = robust_scale(observed, scale_fraction);
    let inv_scale = 1.0 / scale as f64;
    let scale2 = (scale as f64).powi(2);
    let mut objective_value = 0.0_f64;
    let adjoint_source = raw
        .into_iter()
        .map(|value| {
            let x = value as f64 * inv_scale;
            objective_value += scale2 * ((1.0 + x * x).sqrt() - 1.0);
            (value as f64 / (1.0 + x * x).sqrt()) as f32
        })
        .collect();
    TraceResidual {
        adjoint_source,
        objective_value,
        scale,
        misfit: WaveformMisfit::Charbonnier,
    }
}

fn robust_scale(observed: &[f32], scale_fraction: f64) -> f32 {
    let rms = (observed
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        / observed.len().max(1) as f64)
        .sqrt();
    let fraction = scale_fraction.max(f64::EPSILON);
    (rms * fraction).max(f32::EPSILON as f64) as f32
}

#[cfg(test)]
mod tests {
    use super::{evaluate_trace_residual, WaveformMisfit};

    #[test]
    fn l2_residual_matches_least_squares_derivative() {
        let observed = [1.0_f32, -2.0, 0.5];
        let predicted = [0.25_f32, -1.0, 1.5];

        let residual = evaluate_trace_residual(&observed, &predicted, WaveformMisfit::L2, 0.01);

        assert_eq!(residual.adjoint_source, vec![0.75, -1.0, -1.0]);
        assert_eq!(residual.scale, 0.0);
        assert_eq!(residual.misfit.label(), "l2");
        assert!((residual.objective_value - 1.28125).abs() <= 1.0e-12);
    }

    #[test]
    fn charbonnier_residual_bounds_adjoint_source_by_noise_scale() {
        let observed = [10.0_f32, 0.0, -10.0, 0.0];
        let predicted = [0.0_f32, 0.0, 0.0, 0.0];

        let residual =
            evaluate_trace_residual(&observed, &predicted, WaveformMisfit::Charbonnier, 0.10);

        assert_eq!(residual.misfit.label(), "charbonnier");
        assert!(residual.scale > 0.0);
        for value in residual.adjoint_source {
            assert!(
                value.abs() <= residual.scale * (1.0 + 1.0e-6),
                "adjoint source {value} exceeds Charbonnier scale {}",
                residual.scale
            );
        }
        assert!(residual.objective_value > 0.0);
    }
}
