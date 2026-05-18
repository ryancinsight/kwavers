use super::{
    flat_index, forward_with_schedule, ForwardInput, Nonlinear3dAperture, Nonlinear3dConfig,
    SourceEncoding, TimeSchedule,
};

pub(super) struct SourceCalibrationInput<'a> {
    pub background_speed: &'a [f64],
    pub density: &'a [f64],
    pub attenuation_alpha0: &'a [f64],
    pub attenuation_y: &'a [f64],
    pub body: &'a [bool],
    pub target: &'a [bool],
    pub n: usize,
    pub spacing_m: f64,
    pub aperture: &'a Nonlinear3dAperture,
    pub config: &'a Nonlinear3dConfig,
    pub schedule: TimeSchedule,
}

/// Compute a source-drive scale for the Westervelt solve.
///
/// # Contract
///
/// Let `P_ref` be the strongest finite pressure observed at the target mask
/// or focus voxel in the linear calibration solve, and let `P_cfg` be the
/// configured transducer drive. The returned scale is
/// `min(P_cfg / P_ref, 1)`.
///
/// # Proof
///
/// The linear calibration response is homogeneous in source amplitude:
/// `F(a s) = a F(s)`. Therefore `P_cfg / P_ref` is the exact target-pressure
/// scale in the linear limit. The Westervelt solve uses `P_cfg` as the
/// physical transducer pressure bound, so `min(., 1)` preserves exact
/// attenuation when the unscaled drive is too strong and prevents hidden
/// drive amplification that would violate the configured exposure.
pub(super) fn calibrated_source_scale(input: SourceCalibrationInput<'_>) -> f64 {
    let zero_beta = vec![0.0; input.background_speed.len()];
    let calibration = forward_with_schedule(ForwardInput {
        speed: input.background_speed,
        density: input.density,
        beta: &zero_beta,
        attenuation_np_per_m_mhz: Some(input.attenuation_alpha0),
        attenuation_power_law_y: Some(input.attenuation_y),
        source_body_mask: Some(input.body),
        n: input.n,
        spacing_m: input.spacing_m,
        aperture: input.aperture,
        config: input.config,
        schedule: input.schedule,
        encoding: SourceEncoding { index: 0, count: 1 },
        source_scale: 1.0,
        retain_history: false,
    });
    let target_peak = masked_peak(&calibration.peak_pressure, input.target);
    let focus_peak = calibration.peak_pressure[flat_index(input.aperture.focus, input.n)];
    let reference_peak = [target_peak, focus_peak]
        .into_iter()
        .filter(|value| finite_positive(*value))
        .fold(0.0, f64::max);
    assert!(
        finite_positive(reference_peak),
        "nonlinear 3-D source calibration requires a finite positive target or focus pressure"
    );
    (input.config.source_pressure_pa / reference_peak).min(1.0)
}

fn masked_peak(values: &[f64], mask: &[bool]) -> f64 {
    values
        .iter()
        .zip(mask.iter())
        .filter_map(|(value, active)| active.then_some(value.abs()))
        .fold(0.0, f64::max)
}

fn finite_positive(value: f64) -> bool {
    value.is_finite() && value > 0.0
}
