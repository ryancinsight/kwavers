use super::{SlscBeamformer, SlscConfig};
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use num_complex::Complex64;

/// Batch processing for multiple frames
pub fn process_slsc_batch(
    data: &ndarray::Array3<Complex64>,
    config: &SlscConfig,
) -> KwaversResult<Array2<f64>> {
    let (n_elements, n_frames, n_samples) = (data.dim().0, data.dim().1, data.dim().2);

    if n_elements < 2 {
        return Err(KwaversError::Validation(
            crate::core::error::ValidationError::InvalidParameter {
                parameter: "n_elements".to_string(),
                reason: "SLSC requires at least 2 array elements".to_string(),
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
                KwaversError::Validation(crate::core::error::ValidationError::InvalidFormat {
                    field: "frame_data".to_string(),
                    expected: "Array2".to_string(),
                    actual: "Array3 slice".to_string(),
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
