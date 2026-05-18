//! SourcePlan, DriveContext, and plan construction.

use crate::clinical::therapy::theranostic_guidance::nonlinear3d::{
    encoding::SourceEncoding,
    types::{Nonlinear3dAperture, Nonlinear3dConfig, SourceDomain, SourcePlanMetrics},
};

use super::super::TimeSchedule;
use super::stencil::finite_source_stencil;
use super::travel::source_focus_travel_time_s;

pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d::forward) struct SourcePlan {
    pub(super) source_stencils: Vec<Vec<(usize, f64)>>,
    pub(super) focused_delays_s: Vec<f64>,
    pub(super) encoding_weights: Vec<f64>,
}

pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d::forward) struct DriveContext<'a>
{
    pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d::forward) config:
        &'a Nonlinear3dConfig,
    pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d::forward) schedule:
        TimeSchedule,
    pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d::forward) source_scale: f64,
}

pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d::forward) fn build_source_plan(
    speed: &[f64],
    n: usize,
    spacing_m: f64,
    aperture: &Nonlinear3dAperture,
    encoding: SourceEncoding,
    source_body_mask: Option<&[bool]>,
) -> SourcePlan {
    if aperture.source_domain == SourceDomain::ExteriorCoupling {
        assert!(
            source_body_mask.is_some(),
            "exterior coupling sources require the CT-derived body mask"
        );
    }
    if let Some(mask) = source_body_mask {
        assert_eq!(
            mask.len(),
            n * n * n,
            "source body mask length must match the propagation grid"
        );
    }
    let source_stencils = aperture
        .sources
        .iter()
        .map(|idx| finite_source_stencil(*idx, n, spacing_m, aperture, source_body_mask))
        .collect::<Vec<_>>();
    let travel_times_s = aperture
        .sources
        .iter()
        .map(|source| source_focus_travel_time_s(speed, n, spacing_m, *source, aperture.focus))
        .collect::<Vec<_>>();
    let max_travel_time_s = travel_times_s.iter().copied().fold(0.0, f64::max);
    let raw_weights = aperture
        .sources
        .iter()
        .enumerate()
        .map(|(source, _)| encoding.source_weight(source, aperture.sources.len()))
        .collect::<Vec<_>>();
    let focused_delays_s = travel_times_s
        .iter()
        .map(|travel_time_s| (max_travel_time_s - *travel_time_s).max(0.0))
        .collect::<Vec<_>>();
    let encoding_weights = raw_weights
        .into_iter()
        .map(|weight| weight.signum())
        .collect::<Vec<_>>();
    SourcePlan {
        source_stencils,
        focused_delays_s,
        encoding_weights,
    }
}

pub(in crate::clinical::therapy::theranostic_guidance::nonlinear3d) fn source_plan_metrics(
    speed: &[f64],
    n: usize,
    spacing_m: f64,
    aperture: &Nonlinear3dAperture,
    encoding: SourceEncoding,
    source_body_mask: Option<&[bool]>,
) -> SourcePlanMetrics {
    let plan = build_source_plan(speed, n, spacing_m, aperture, encoding, source_body_mask);
    let support_min = plan.source_stencils.iter().map(Vec::len).min().unwrap_or(0);
    let support_max = plan.source_stencils.iter().map(Vec::len).max().unwrap_or(0);
    let support_mean = if plan.source_stencils.is_empty() {
        0.0
    } else {
        plan.source_stencils.iter().map(Vec::len).sum::<usize>() as f64
            / plan.source_stencils.len() as f64
    };
    let delay_min = plan
        .focused_delays_s
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let delay_max = plan.focused_delays_s.iter().copied().fold(0.0, f64::max);
    let focused_delay_min_s = if delay_min.is_finite() {
        delay_min
    } else {
        0.0
    };
    SourcePlanMetrics {
        source_support_min: support_min,
        source_support_mean: support_mean,
        source_support_max: support_max,
        focused_delay_min_s,
        focused_delay_max_s: delay_max,
        focused_delay_span_s: (delay_max - focused_delay_min_s).max(0.0),
    }
}
