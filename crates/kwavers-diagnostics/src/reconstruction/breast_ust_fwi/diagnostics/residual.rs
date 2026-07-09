use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{
    /* s -- no leto equivalent */,
    Array3,
    ArrayView1,
};
use kwavers_math::fft::Complex64;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BreastUstReceiverChannelPolicy {
    All,
    ActiveOnly,
    PassiveOnly,
}

impl BreastUstReceiverChannelPolicy {
    pub fn parse(value: &str) -> KwaversResult<Self> {
        match value {
            "all" => Ok(Self::All),
            "active_only" => Ok(Self::ActiveOnly),
            "passive_only" => Ok(Self::PassiveOnly),
            other => Err(KwaversError::InvalidInput(format!(
                "unsupported receiver channel policy: {other}"
            ))),
        }
    }

    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::ActiveOnly => "active_only",
            Self::PassiveOnly => "passive_only",
        }
    }

    pub(crate) fn selects(
        self,
        transmit: usize,
        receiver: usize,
        circumferential_elements: usize,
    ) -> bool {
        let active = receiver % circumferential_elements == transmit;
        match self {
            Self::All => true,
            Self::ActiveOnly => active,
            Self::PassiveOnly => !active,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreastUstScaledObservationResidualMetrics {
    pub frequency_count: usize,
    pub transmission_count: usize,
    pub receiver_count: usize,
    pub row_count: usize,
    pub selected_receiver_count: usize,
    pub observed_l2_norm: f64,
    pub scaled_residual_l2_norm: f64,
    pub normalized_l2_residual: f64,
    pub max_abs_scaled_residual: f64,
    pub row_normalized_l2_residual_mean: f64,
    pub row_normalized_l2_residual_max: f64,
    pub source_scale_magnitude_min: f64,
    pub source_scale_magnitude_max: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreastUstSourceChannelResidualDiagnostics {
    pub active_receiver_count_per_row: usize,
    pub passive_receiver_count_per_row: usize,
    pub all_channel_normalized_l2_residual: f64,
    pub passive_only_normalized_l2_residual: f64,
    pub passive_only_scaled_residual_l2_norm: f64,
    pub active_full_scale_residual_l2_norm: f64,
    pub passive_full_scale_residual_l2_norm: f64,
    pub active_full_scale_observed_l2_norm: f64,
    pub passive_full_scale_observed_l2_norm: f64,
    pub active_full_scale_normalized_l2_residual: f64,
    pub passive_full_scale_normalized_l2_residual: f64,
    pub active_full_scale_residual_energy_fraction: f64,
    pub passive_full_scale_residual_energy_fraction: f64,
}

pub fn scaled_observation_residual_metrics(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    receiver_mask: Option<&Array3<bool>>,
) -> KwaversResult<BreastUstScaledObservationResidualMetrics> {
    let mask = receiver_mask.map_or_else(
        || Ok(Array3::from_elem(predicted.dim(), true)),
        |mask| validated_receiver_mask(predicted.dim(), mask),
    )?;
    scaled_observation_residual_metrics_by_receiver(predicted, observed, |f, t, r| mask[[f, t, r]])
}

pub(crate) fn scaled_observation_residual_metrics_by_receiver(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    selected: impl Fn(usize, usize, usize) -> bool,
) -> KwaversResult<BreastUstScaledObservationResidualMetrics> {
    validate_observation_pair(predicted, observed)?;
    let (frequency_count, transmission_count, receiver_count) = predicted.dim();
    let mut residual_norm_sq = 0.0;
    let mut observed_norm_sq = 0.0;
    let mut max_abs_residual = 0.0;
    let mut row_residual_sum = 0.0;
    let mut row_residual_max = 0.0;
    let mut row_count = 0usize;
    let mut selected_receiver_count = 0usize;
    let mut scale_magnitude_min = f64::INFINITY;
    let mut scale_magnitude_max = f64::NEG_INFINITY;

    for frequency_index in 0..frequency_count {
        for transmit_index in 0..transmission_count {
            let predicted_row = predicted.slice(s![frequency_index, transmit_index, ..]);
            let observed_row = observed.slice(s![frequency_index, transmit_index, ..]);
            let scale = row_scale_selected(predicted_row, observed_row, |receiver| {
                selected(frequency_index, transmit_index, receiver)
            })?;
            scale_magnitude_min = f64::min(scale_magnitude_min, scale.norm());
            scale_magnitude_max = f64::max(scale_magnitude_max, scale.norm());
            let mut row_residual_sq = 0.0;
            let mut row_observed_sq = 0.0;
            for receiver_index in 0..receiver_count {
                if !selected(frequency_index, transmit_index, receiver_index) {
                    continue;
                }
                let residual = scale * predicted[[frequency_index, transmit_index, receiver_index]]
                    - observed[[frequency_index, transmit_index, receiver_index]];
                row_residual_sq += residual.norm_sqr();
                row_observed_sq +=
                    observed[[frequency_index, transmit_index, receiver_index]].norm_sqr();
                max_abs_residual = f64::max(max_abs_residual, residual.norm());
                selected_receiver_count += 1;
            }
            if row_observed_sq <= f64::EPSILON {
                return Err(KwaversError::InvalidInput(
                    "selected observed row has zero energy".into(),
                ));
            }
            residual_norm_sq += row_residual_sq;
            observed_norm_sq += row_observed_sq;
            let row_residual = (row_residual_sq / row_observed_sq).sqrt();
            row_residual_sum += row_residual;
            row_residual_max = f64::max(row_residual_max, row_residual);
            row_count += 1;
        }
    }

    if row_count == 0 || observed_norm_sq <= f64::EPSILON {
        return Err(KwaversError::InvalidInput(
            "residual metrics require nonzero observed energy".into(),
        ));
    }
    Ok(BreastUstScaledObservationResidualMetrics {
        frequency_count,
        transmission_count,
        receiver_count,
        row_count,
        selected_receiver_count,
        observed_l2_norm: observed_norm_sq.sqrt(),
        scaled_residual_l2_norm: residual_norm_sq.sqrt(),
        normalized_l2_residual: residual_norm_sq.sqrt() / observed_norm_sq.sqrt(),
        max_abs_scaled_residual: max_abs_residual,
        row_normalized_l2_residual_mean: row_residual_sum / row_count as f64,
        row_normalized_l2_residual_max: row_residual_max,
        source_scale_magnitude_min: scale_magnitude_min,
        source_scale_magnitude_max: scale_magnitude_max,
    })
}

pub(crate) fn scaled_observation_residual_metrics_by_policy(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    receiver_channel_policy: BreastUstReceiverChannelPolicy,
) -> KwaversResult<BreastUstScaledObservationResidualMetrics> {
    validate_observation_pair(predicted, observed)?;
    let (_, transmission_count, receiver_count) = predicted.dim();
    validate_ring_channel_policy_shape(
        transmission_count,
        receiver_count,
        receiver_channel_policy,
    )?;
    scaled_observation_residual_metrics_by_receiver(predicted, observed, |_, transmit, receiver| {
        receiver_channel_policy.selects(transmit, receiver, transmission_count)
    })
}

pub fn source_channel_residual_diagnostics(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
    circumferential_elements: usize,
    rows: usize,
) -> KwaversResult<BreastUstSourceChannelResidualDiagnostics> {
    validate_observation_pair(predicted, observed)?;
    let shape = predicted.dim();
    let active_mask = source_receiver_mask(shape, circumferential_elements, rows)?;
    let passive_mask = passive_receiver_mask(shape, circumferential_elements, rows)?;
    let full_metrics = scaled_observation_residual_metrics(predicted, observed, None)?;
    let passive_metrics =
        scaled_observation_residual_metrics(predicted, observed, Some(&passive_mask))?;
    let residual = scaled_residual_cube(predicted, observed)?;
    let active_residual_sq = masked_l2_norm_sq(&residual, &active_mask);
    let passive_residual_sq = masked_l2_norm_sq(&residual, &passive_mask);
    let active_observed_sq = masked_l2_norm_sq(observed, &active_mask);
    let passive_observed_sq = masked_l2_norm_sq(observed, &passive_mask);
    let total_residual_sq = active_residual_sq + passive_residual_sq;

    Ok(BreastUstSourceChannelResidualDiagnostics {
        active_receiver_count_per_row: rows,
        passive_receiver_count_per_row: shape.2 - rows,
        all_channel_normalized_l2_residual: full_metrics.normalized_l2_residual,
        passive_only_normalized_l2_residual: passive_metrics.normalized_l2_residual,
        passive_only_scaled_residual_l2_norm: passive_metrics.scaled_residual_l2_norm,
        active_full_scale_residual_l2_norm: active_residual_sq.sqrt(),
        passive_full_scale_residual_l2_norm: passive_residual_sq.sqrt(),
        active_full_scale_observed_l2_norm: active_observed_sq.sqrt(),
        passive_full_scale_observed_l2_norm: passive_observed_sq.sqrt(),
        active_full_scale_normalized_l2_residual: active_residual_sq.sqrt()
            / active_observed_sq.sqrt().max(f64::EPSILON),
        passive_full_scale_normalized_l2_residual: passive_residual_sq.sqrt()
            / passive_observed_sq.sqrt().max(f64::EPSILON),
        active_full_scale_residual_energy_fraction: active_residual_sq
            / total_residual_sq.max(f64::EPSILON),
        passive_full_scale_residual_energy_fraction: passive_residual_sq
            / total_residual_sq.max(f64::EPSILON),
    })
}

pub fn source_receiver_mask(
    observation_shape: (usize, usize, usize),
    circumferential_elements: usize,
    rows: usize,
) -> KwaversResult<Array3<bool>> {
    validate_observation_shape(observation_shape)?;
    let (frequency_count, transmissions, receivers) = observation_shape;
    if circumferential_elements < 2 || rows == 0 {
        return Err(KwaversError::InvalidInput(
            "array topology requires at least two angular elements and one row".into(),
        ));
    }
    if transmissions != circumferential_elements {
        return Err(KwaversError::DimensionMismatch(
            "transmission count must equal circumferential_elements".into(),
        ));
    }
    let expected_receivers = circumferential_elements * rows;
    if receivers != expected_receivers {
        return Err(KwaversError::DimensionMismatch(format!(
            "receiver count mismatch: expected {expected_receivers}, got {receivers}"
        )));
    }
    let mut mask = Array3::<bool>::from_elem(observation_shape, false);
    for frequency_index in 0..frequency_count {
        for transmit_index in 0..transmissions {
            for row_index in 0..rows {
                let receiver_index = row_index * circumferential_elements + transmit_index;
                mask[[frequency_index, transmit_index, receiver_index]] = true;
            }
        }
    }
    Ok(mask)
}

pub fn passive_receiver_mask(
    observation_shape: (usize, usize, usize),
    circumferential_elements: usize,
    rows: usize,
) -> KwaversResult<Array3<bool>> {
    Ok(source_receiver_mask(observation_shape, circumferential_elements, rows)?.mapv(|v| !v))
}

pub(crate) fn row_scale(
    predicted: ArrayView1<'_, Complex64>,
    observed: ArrayView1<'_, Complex64>,
) -> KwaversResult<Complex64> {
    row_scale_selected(predicted, observed, |_| true)
}

fn scaled_residual_cube(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
) -> KwaversResult<Array3<Complex64>> {
    let (frequency_count, transmission_count, receiver_count) = predicted.dim();
    let mut residual = Array3::<Complex64>::zeros(predicted.dim());
    for frequency_index in 0..frequency_count {
        for transmit_index in 0..transmission_count {
            let predicted_row = predicted.slice(s![frequency_index, transmit_index, ..]);
            let observed_row = observed.slice(s![frequency_index, transmit_index, ..]);
            let scale = row_scale(predicted_row, observed_row)?;
            for receiver_index in 0..receiver_count {
                residual[[frequency_index, transmit_index, receiver_index]] = scale
                    * predicted[[frequency_index, transmit_index, receiver_index]]
                    - observed[[frequency_index, transmit_index, receiver_index]];
            }
        }
    }
    Ok(residual)
}

pub(crate) fn row_scale_selected(
    predicted: ArrayView1<'_, Complex64>,
    observed: ArrayView1<'_, Complex64>,
    selected: impl Fn(usize) -> bool,
) -> KwaversResult<Complex64> {
    if predicted.len() != observed.len() || predicted.is_empty() {
        return Err(KwaversError::DimensionMismatch(
            "observation rows must have equal nonzero length".into(),
        ));
    }
    let mut numerator = Complex64::new(0.0, 0.0);
    let mut denominator = 0.0;
    let mut selected_count = 0usize;
    for (receiver, (&p, &d)) in predicted.iter().zip(observed.iter()).enumerate() {
        if !selected(receiver) {
            continue;
        }
        if !p.re.is_finite() || !p.im.is_finite() || !d.re.is_finite() || !d.im.is_finite() {
            return Err(KwaversError::InvalidInput(
                "observation row contains nonfinite complex values".into(),
            ));
        }
        numerator += p.conj() * d;
        denominator += p.norm_sqr();
        selected_count += 1;
    }
    if selected_count == 0 {
        return Err(KwaversError::InvalidInput(
            "receiver_mask must select at least one receiver for every frequency/transmit row"
                .into(),
        ));
    }
    if denominator <= f64::EPSILON {
        return Err(KwaversError::InvalidInput(
            "predicted observation row has zero energy".into(),
        ));
    }
    Ok(numerator / denominator)
}

fn masked_l2_norm_sq(values: &Array3<Complex64>, mask: &Array3<bool>) -> f64 {
    values
        .iter()
        .zip(mask.iter())
        .filter_map(|(&value, &selected)| selected.then_some(value.norm_sqr()))
        .sum()
}

fn validated_receiver_mask(
    observation_shape: (usize, usize, usize),
    receiver_mask: &Array3<bool>,
) -> KwaversResult<Array3<bool>> {
    if receiver_mask.dim() != observation_shape {
        return Err(KwaversError::DimensionMismatch(format!(
            "receiver_mask shape mismatch: expected {:?}, got {:?}",
            observation_shape,
            receiver_mask.dim()
        )));
    }
    let (frequency_count, transmission_count, receiver_count) = observation_shape;
    for frequency_index in 0..frequency_count {
        for transmit_index in 0..transmission_count {
            if !(0..receiver_count)
                .any(|receiver| receiver_mask[[frequency_index, transmit_index, receiver]])
            {
                return Err(KwaversError::InvalidInput(
                    "receiver_mask must select at least one receiver for every frequency/transmit row"
                        .into(),
                ));
            }
        }
    }
    Ok(receiver_mask.clone())
}

pub(crate) fn validate_ring_channel_policy_shape(
    transmission_count: usize,
    receiver_count: usize,
    receiver_channel_policy: BreastUstReceiverChannelPolicy,
) -> KwaversResult<()> {
    if receiver_channel_policy == BreastUstReceiverChannelPolicy::All {
        return Ok(());
    }
    if transmission_count == 0 || receiver_count == 0 {
        return Err(KwaversError::InvalidInput(
            "receiver channel policy requires positive observation axes".into(),
        ));
    }
    if !receiver_count.is_multiple_of(transmission_count) {
        return Err(KwaversError::DimensionMismatch(format!(
            "receiver count {receiver_count} must be an integer multiple of transmission count {transmission_count}"
        )));
    }
    Ok(())
}

pub(crate) fn validate_observation_pair(
    predicted: &Array3<Complex64>,
    observed: &Array3<Complex64>,
) -> KwaversResult<()> {
    if predicted.dim() != observed.dim() {
        return Err(KwaversError::DimensionMismatch(format!(
            "observation shape mismatch: predicted {:?}, observed {:?}",
            predicted.dim(),
            observed.dim()
        )));
    }
    validate_observation_shape(predicted.dim())
}

fn validate_observation_shape(shape: (usize, usize, usize)) -> KwaversResult<()> {
    if shape.0 == 0 || shape.1 == 0 || shape.2 == 0 {
        return Err(KwaversError::InvalidInput(format!(
            "observation dimensions must be positive, got {shape:?}"
        )));
    }
    Ok(())
}
