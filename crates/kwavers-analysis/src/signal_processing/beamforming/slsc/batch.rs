use super::{SlscBeamformer, SlscConfig};
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use num_complex::Complex64;

/// Batch processing for multiple frames
/// # Errors
/// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
/// - Propagates any [`KwaversError`] returned by called functions.
///
pub fn process_slsc_batch(
    data: &ndarray::Array3<Complex64>,
    config: &SlscConfig,
) -> KwaversResult<Array2<f64>> {
    let (n_elements, n_frames, n_samples) = (data.dim().0, data.dim().1, data.dim().2);

    if n_elements < 2 {
        return Err(KwaversError::Validation(
            kwavers_core::error::ValidationError::InvalidParameter {
                parameter: "n_elements".to_owned(),
                reason: "SLSC requires at least 2 array elements".to_owned(),
            },
        ));
    }

    let slsc = SlscBeamformer::with_config(config.clone());
    let mut results = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let frame_data: Array2<Complex64> = data
            .slice(ndarray::s![.., frame_idx, ..])
            .to_owned()
            .into_dimensionality()
            .map_err(|_| {
                KwaversError::Validation(kwavers_core::error::ValidationError::InvalidFormat {
                    field: "frame_data".to_owned(),
                    expected: "Array2".to_owned(),
                    actual: "Array3 slice".to_owned(),
                })
            })?;

        let coherence = slsc.process(&frame_data)?;
        results.push(coherence);
    }

    let mut output = Array2::zeros((n_frames, n_samples));
    for (frame_idx, frame_result) in results.iter().enumerate() {
        output.row_mut(frame_idx).assign(frame_result);
    }

    Ok(output)
}
