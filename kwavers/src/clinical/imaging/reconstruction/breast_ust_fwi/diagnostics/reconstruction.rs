use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreastUstReconstructionMetrics {
    pub rmse_m_s: f64,
    pub normalized_rmse: f64,
    pub pearson_correlation: f64,
    pub reference_min_m_s: f64,
    pub reference_max_m_s: f64,
    pub estimate_min_m_s: f64,
    pub estimate_max_m_s: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BreastUstTable1Parity {
    pub phantom_index: f64,
    pub table1_3d_rmse_m_s: f64,
    pub table1_3d_pearson_correlation: f64,
    pub rmse_threshold_m_s: f64,
    pub pcc_threshold: f64,
    pub rmse_pass: bool,
    pub pcc_pass: bool,
    pub passes: bool,
}

pub fn reconstruction_metrics(
    reference_m_s: &Array3<f64>,
    estimate_m_s: &Array3<f64>,
) -> KwaversResult<BreastUstReconstructionMetrics> {
    validate_pair(reference_m_s, estimate_m_s)?;
    let n = reference_m_s.len() as f64;
    let rmse = (reference_m_s
        .iter()
        .zip(estimate_m_s.iter())
        .map(|(&reference, &estimate)| {
            let diff = estimate - reference;
            diff * diff
        })
        .sum::<f64>()
        / n)
        .sqrt();
    let reference_rms = (reference_m_s.iter().map(|value| value * value).sum::<f64>() / n).sqrt();
    if reference_rms <= f64::EPSILON {
        return Err(KwaversError::InvalidInput("reference RMS is zero".into()));
    }

    let mean_reference = reference_m_s.iter().sum::<f64>() / n;
    let mean_estimate = estimate_m_s.iter().sum::<f64>() / n;
    let mut covariance = 0.0;
    let mut reference_var = 0.0;
    let mut estimate_var = 0.0;
    for (&reference, &estimate) in reference_m_s.iter().zip(estimate_m_s.iter()) {
        let dr = reference - mean_reference;
        let de = estimate - mean_estimate;
        covariance += dr * de;
        reference_var += dr * dr;
        estimate_var += de * de;
    }
    let correlation_denominator = (reference_var * estimate_var).sqrt();
    if correlation_denominator <= f64::EPSILON {
        return Err(KwaversError::InvalidInput(
            "Pearson correlation requires nonconstant inputs".into(),
        ));
    }

    Ok(BreastUstReconstructionMetrics {
        rmse_m_s: rmse,
        normalized_rmse: rmse / reference_rms,
        pearson_correlation: covariance / correlation_denominator,
        reference_min_m_s: reference_m_s.iter().copied().fold(f64::INFINITY, f64::min),
        reference_max_m_s: reference_m_s
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
        estimate_min_m_s: estimate_m_s.iter().copied().fold(f64::INFINITY, f64::min),
        estimate_max_m_s: estimate_m_s
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    })
}

pub fn table1_parity(
    rmse_m_s: f64,
    pearson_correlation: f64,
    phantom_index: usize,
    rmse_multiplier: f64,
    pcc_fraction: f64,
) -> KwaversResult<BreastUstTable1Parity> {
    let Some((table1_rmse, table1_pcc)) = table1_reference(phantom_index) else {
        return Err(KwaversError::InvalidInput(format!(
            "phantom_index must be one of 1, 2, 3, got {phantom_index}"
        )));
    };
    if !rmse_multiplier.is_finite() || rmse_multiplier <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "rmse_multiplier must be positive and finite, got {rmse_multiplier}"
        )));
    }
    if !pcc_fraction.is_finite() || pcc_fraction <= 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "pcc_fraction must be positive and finite, got {pcc_fraction}"
        )));
    }
    if !rmse_m_s.is_finite() || rmse_m_s < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "rmse_m_s must be nonnegative and finite, got {rmse_m_s}"
        )));
    }
    if !pearson_correlation.is_finite() {
        return Err(KwaversError::InvalidInput(format!(
            "pearson_correlation must be finite, got {pearson_correlation}"
        )));
    }
    let rmse_threshold = table1_rmse * rmse_multiplier;
    let pcc_threshold = table1_pcc * pcc_fraction;
    let rmse_pass = rmse_m_s <= rmse_threshold;
    let pcc_pass = pearson_correlation >= pcc_threshold;
    Ok(BreastUstTable1Parity {
        phantom_index: phantom_index as f64,
        table1_3d_rmse_m_s: table1_rmse,
        table1_3d_pearson_correlation: table1_pcc,
        rmse_threshold_m_s: rmse_threshold,
        pcc_threshold,
        rmse_pass,
        pcc_pass,
        passes: rmse_pass && pcc_pass,
    })
}

fn validate_pair(reference_m_s: &Array3<f64>, estimate_m_s: &Array3<f64>) -> KwaversResult<()> {
    validate_volume(reference_m_s)?;
    validate_volume(estimate_m_s)?;
    if reference_m_s.dim() != estimate_m_s.dim() {
        return Err(KwaversError::DimensionMismatch(format!(
            "shape mismatch: reference {:?}, estimate {:?}",
            reference_m_s.dim(),
            estimate_m_s.dim()
        )));
    }
    Ok(())
}

fn validate_volume(volume: &Array3<f64>) -> KwaversResult<()> {
    if volume.is_empty() {
        return Err(KwaversError::InvalidInput(
            "volume must not be empty".into(),
        ));
    }
    for &value in volume {
        if !value.is_finite() {
            return Err(KwaversError::InvalidInput(
                "volume contains nonfinite sound-speed values".into(),
            ));
        }
        if value <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "sound-speed volume must be strictly positive".into(),
            ));
        }
    }
    Ok(())
}

fn table1_reference(phantom_index: usize) -> Option<(f64, f64)> {
    match phantom_index {
        1 => Some((15.5, 0.8848)),
        2 => Some((10.1, 0.8981)),
        3 => Some((8.4, 0.8967)),
        _ => None,
    }
}
