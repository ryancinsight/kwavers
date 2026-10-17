//! One authoritative source-fit, residual and objective reduction per row.

use kwavers_core::error::KwaversResult;
use kwavers_math::fft::Complex64;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    complex_l2_objective, complex_source_scale,
};

/// Static callbacks select coefficient storage once per row. Both
/// representations retain the complete source-fitting/residual algorithm and
/// the original receiver and objective reduction order.
pub(super) fn accumulate(
    observed: &[Complex64],
    fit_source: bool,
    predict: impl Fn(usize) -> Complex64,
    adjoint: impl FnOnce(Complex64, &[Complex64], &mut [f64]),
    objective: &mut f64,
    gradient: &mut [f64],
) -> KwaversResult<()> {
    let (source_scale, mut scaled_prediction) = if fit_source {
        let predicted = observed
            .iter()
            .enumerate()
            .map(|(index, _)| predict(index))
            .collect::<Vec<_>>();
        let scale = complex_source_scale(&predicted, observed)?;
        (
            scale,
            predicted
                .iter()
                .map(|&value| scale * value)
                .collect::<Vec<_>>(),
        )
    } else {
        (Complex64::new(1.0, 0.0), Vec::with_capacity(observed.len()))
    };
    let mut residuals = Vec::with_capacity(observed.len());
    for (receiver, &observed) in observed.iter().enumerate() {
        if !fit_source {
            scaled_prediction.push(source_scale * predict(receiver));
        }
        residuals.push(scaled_prediction[receiver] - observed);
    }
    adjoint(source_scale, &residuals, gradient);
    *objective +=
        complex_l2_objective(&scaled_prediction, observed).expect("validated equal row lengths");
    Ok(())
}
