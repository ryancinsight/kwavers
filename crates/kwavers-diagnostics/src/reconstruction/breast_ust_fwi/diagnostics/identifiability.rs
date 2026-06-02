use kwavers_core::error::{KwaversError, KwaversResult};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreastUstSourceScalingPolicy {
    Fixed,
    Estimated,
}

impl BreastUstSourceScalingPolicy {
    pub fn parse(value: &str) -> KwaversResult<Self> {
        match value {
            "fixed" => Ok(Self::Fixed),
            "estimated" => Ok(Self::Estimated),
            other => Err(KwaversError::InvalidInput(format!(
                "unsupported source scaling policy: {other}"
            ))),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fixed => "fixed",
            Self::Estimated => "estimated",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BreastUstAcquisitionIdentifiability {
    pub unknown_voxels: usize,
    pub frequency_count: usize,
    pub complex_observations: usize,
    pub real_observation_dof: usize,
    pub source_scaling_policy: BreastUstSourceScalingPolicy,
    pub estimated_source_scale_real_dof: usize,
    pub informative_real_dof_upper_bound: usize,
    pub informative_dof_to_unknown_ratio: f64,
    pub underdetermined_by_rank_upper_bound: bool,
}

pub fn acquisition_identifiability(
    shape: (usize, usize, usize),
    frequencies_hz: &[f64],
    transmissions: usize,
    receivers: usize,
    source_scaling_policy: BreastUstSourceScalingPolicy,
) -> KwaversResult<BreastUstAcquisitionIdentifiability> {
    validate_shape(shape)?;
    validate_counts(frequencies_hz, transmissions, receivers)?;
    let unknown_voxels = shape.0 * shape.1 * shape.2;
    let frequency_count = frequencies_hz.len();
    let complex_observations = frequency_count * transmissions * receivers;
    let real_observation_dof = 2 * complex_observations;
    let nuisance_dof = source_scale_real_dof(frequency_count, transmissions, source_scaling_policy);
    let informative_dof = real_observation_dof.saturating_sub(nuisance_dof);
    Ok(BreastUstAcquisitionIdentifiability {
        unknown_voxels,
        frequency_count,
        complex_observations,
        real_observation_dof,
        source_scaling_policy,
        estimated_source_scale_real_dof: nuisance_dof,
        informative_real_dof_upper_bound: informative_dof,
        informative_dof_to_unknown_ratio: informative_dof as f64 / unknown_voxels as f64,
        underdetermined_by_rank_upper_bound: informative_dof < unknown_voxels,
    })
}

fn source_scale_real_dof(
    frequency_count: usize,
    transmissions: usize,
    source_scaling_policy: BreastUstSourceScalingPolicy,
) -> usize {
    match source_scaling_policy {
        BreastUstSourceScalingPolicy::Fixed => 0,
        BreastUstSourceScalingPolicy::Estimated => 2 * frequency_count * transmissions,
    }
}

fn validate_shape(shape: (usize, usize, usize)) -> KwaversResult<()> {
    if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "shape must contain three positive axes, got {shape:?}"
        )));
    }
    Ok(())
}

fn validate_counts(
    frequencies_hz: &[f64],
    transmissions: usize,
    receivers: usize,
) -> KwaversResult<()> {
    if frequencies_hz.is_empty() {
        return Err(KwaversError::InvalidInput(
            "frequencies_hz must not be empty".into(),
        ));
    }
    for &frequency_hz in frequencies_hz {
        if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "frequency must be positive and finite, got {frequency_hz}"
            )));
        }
    }
    if transmissions == 0 {
        return Err(KwaversError::InvalidInput(
            "transmissions must be positive".into(),
        ));
    }
    if receivers == 0 {
        return Err(KwaversError::InvalidInput(
            "receivers must be positive".into(),
        ));
    }
    Ok(())
}
