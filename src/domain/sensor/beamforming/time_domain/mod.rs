use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

pub const DEFAULT_DELAY_REFERENCE: DelayReference = DelayReference::SensorIndex(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayReference {
    SensorIndex(usize),
    EarliestArrival,
    LatestArrival,
}

impl DelayReference {
    #[must_use]
    pub const fn recommended_default() -> Self {
        Self::SensorIndex(0)
    }

    pub fn resolve_reference_delay_s(self, delays_s: &[f64]) -> KwaversResult<f64> {
        if delays_s.is_empty() {
            return Err(KwaversError::InvalidInput(
                "DelayReference: delays_s must be non-empty".to_string(),
            ));
        }

        for (i, &tau) in delays_s.iter().enumerate() {
            if !tau.is_finite() {
                return Err(KwaversError::InvalidInput(format!(
                    "DelayReference: delay[{i}] = {tau} is non-finite"
                )));
            }
            if tau < 0.0 {
                return Err(KwaversError::InvalidInput(format!(
                    "DelayReference: delay[{i}] = {tau} is negative"
                )));
            }
        }

        let tau_ref = match self {
            DelayReference::SensorIndex(idx) => delays_s.get(idx).copied().ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "DelayReference::SensorIndex({idx}) out of bounds for delays_s (len={})",
                    delays_s.len()
                ))
            })?,
            DelayReference::EarliestArrival => *delays_s
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
            DelayReference::LatestArrival => *delays_s
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        };

        if !tau_ref.is_finite() {
            return Err(KwaversError::InvalidInput(
                "DelayReference: resolved reference delay is non-finite".to_string(),
            ));
        }

        Ok(tau_ref)
    }

    pub fn compute_relative_delays(self, delays_s: &[f64]) -> KwaversResult<Vec<f64>> {
        let tau_ref = self.resolve_reference_delay_s(delays_s)?;
        Ok(delays_s.iter().map(|&tau| tau - tau_ref).collect())
    }

    pub fn compute_alignment_shifts(self, delays_s: &[f64]) -> KwaversResult<Vec<f64>> {
        let tau_ref = self.resolve_reference_delay_s(delays_s)?;
        Ok(delays_s.iter().map(|&tau| tau_ref - tau).collect())
    }
}

pub fn relative_delays_s(delays_s: &[f64], reference: DelayReference) -> KwaversResult<Vec<f64>> {
    reference.compute_relative_delays(delays_s)
}

pub fn alignment_shifts_s(
    delays_s: &[f64],
    reference: DelayReference,
) -> KwaversResult<Vec<f64>> {
    reference.compute_alignment_shifts(delays_s)
}

pub fn delay_and_sum(
    sensor_data: &Array3<f64>,
    sampling_frequency_hz: f64,
    delays_s: &[f64],
    weights: &[f64],
    reference: DelayReference,
) -> KwaversResult<Array3<f64>> {
    let (n_elements, channels, n_samples) = sensor_data.dim();

    if channels != 1 {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::delay_and_sum expects sensor_data shape (n_elements, 1, n_samples); got channels={channels}"
        )));
    }
    if n_elements == 0 || n_samples == 0 {
        return Err(KwaversError::InvalidInput(
            "time_domain::delay_and_sum requires n_elements > 0 and n_samples > 0".to_string(),
        ));
    }

    if !sampling_frequency_hz.is_finite() || sampling_frequency_hz <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::delay_and_sum requires sampling_frequency_hz to be finite and > 0; got {sampling_frequency_hz}"
        )));
    }

    if delays_s.len() != n_elements {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::delay_and_sum: delays_s length ({}) must equal n_elements ({n_elements})",
            delays_s.len()
        )));
    }
    if weights.len() != n_elements {
        return Err(KwaversError::InvalidInput(format!(
            "time_domain::delay_and_sum: weights length ({}) must equal n_elements ({n_elements})",
            weights.len()
        )));
    }

    let rel_delays_s = relative_delays_s(delays_s, reference)?;

    let mut shifts: Vec<isize> = Vec::with_capacity(n_elements);
    for (i, &dt) in rel_delays_s.iter().enumerate() {
        let k = (dt * sampling_frequency_hz).round();
        if !k.is_finite() {
            return Err(KwaversError::InvalidInput(format!(
                "time_domain::delay_and_sum: relative_delay[{i}] = {dt} produced non-finite sample shift (fs={sampling_frequency_hz})"
            )));
        }
        shifts.push(k as isize);
    }

    let min_shift = shifts.iter().copied().min().unwrap_or(0);
    let offset = -min_shift;

    let mut output = Array3::<f64>::zeros((1, 1, n_samples));

    for elem_idx in 0..n_elements {
        let effective_shift = shifts[elem_idx] + offset;
        let eff = if effective_shift < 0 {
            0usize
        } else {
            effective_shift as usize
        };

        if eff >= n_samples {
            continue;
        }

        let w = weights[elem_idx];
        for t in eff..n_samples {
            output[[0, 0, t - eff]] += sensor_data[[elem_idx, 0, t]] * w;
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delay_reference_defaults() {
        assert_eq!(
            DelayReference::recommended_default(),
            DelayReference::SensorIndex(0)
        );
        assert_eq!(DEFAULT_DELAY_REFERENCE, DelayReference::SensorIndex(0));
    }

    #[test]
    fn delay_and_sum_runs() {
        let mut sensor_data = Array3::<f64>::zeros((2, 1, 8));
        sensor_data[[0, 0, 3]] = 1.0;
        sensor_data[[1, 0, 5]] = 1.0;

        let fs = 10.0;
        let delays = vec![1.0, 1.2];
        let weights = vec![1.0, 1.0];

        let output = delay_and_sum(
            &sensor_data,
            fs,
            &delays,
            &weights,
            DelayReference::SensorIndex(0),
        )
        .expect("das output");

        assert!((output[[0, 0, 3]] - 2.0).abs() < 1e-12);
    }
}
