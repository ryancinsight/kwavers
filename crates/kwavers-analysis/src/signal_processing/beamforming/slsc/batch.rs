use super::{SlscBeamformer, SlscConfig};
use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array2;

/// Batch processing for multiple frames
/// # Errors
/// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
/// - Propagates any `KwaversError` returned by called functions.
///
pub fn process_slsc_batch(
    data: &leto::Array3<Complex64>,
    config: &SlscConfig,
) -> KwaversResult<Array2<f64>> {
    let [n_elements, n_frames, n_samples] = data.shape();

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
            .index_axis::<2>(1, frame_idx)
            .map_err(|_| {
                KwaversError::Validation(kwavers_core::error::ValidationError::InvalidFormat {
                    field: "frame_data".to_owned(),
                    expected: "Array2".to_owned(),
                    actual: "Array3 slice".to_owned(),
                })
            })?
            .to_contiguous();

        let coherence = slsc.process(&frame_data)?;
        results.push(coherence);
    }

    let mut output = Array2::zeros((n_frames, n_samples));
    for (frame_idx, frame_result) in results.iter().enumerate() {
        output
            .index_axis_mut::<1>(0, frame_idx)
            .expect("invariant: frame_idx < n_frames")
            .assign(frame_result);
    }

    Ok(output)
}