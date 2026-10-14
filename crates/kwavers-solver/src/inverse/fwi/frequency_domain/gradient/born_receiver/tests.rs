//! Exact dyadic oracle for the row-major Born adjoint projection.

use super::super::super::forward::voxel_centers;
use super::*;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::helmholtz_slowness_derivative;

/// For `a = source_scale * incident * green`, the real adjoint is
/// `dV * (2 omega² s) * (a.re residual.re + a.im residual.im)`.
/// All fixture values are dyadic and bounded so these products and sums are
/// exactly representable. The coordinate oracle is independent of flat storage.
fn check_adjoint<const K: usize>() {
    let _subscriber = tracing::subscriber::set_default(super::super::test_support::subscriber());
    let span = tracing::info_span!("dense_adjoint_matches_coordinate_oracle");
    let _entered = span.enter();
    for shape in [(2, 3, 4), (3, 2, 1), (1, 1, 1), (3, 3, 2)] {
        let arrange = tracing::info_span!("arrange", ?shape).entered();
        let centers = voxel_centers(shape, 0.5);
        // The largest index is 23; conversion to the domain's native precision
        // is exact and supplies distinct field values at every voxel.
        let ordinal: Vec<_> = (0..centers.len()).map(|i| (i + 1) as f64).collect();
        let incident: Vec<_> = ordinal
            .iter()
            .map(|&q| Complex64::new(q / 2.0, -q / 4.0))
            .collect();
        let green: Vec<_> = ordinal
            .iter()
            .map(|&q| Complex64::new((q + 1.0) / 4.0, q / 8.0))
            .collect();
        let slowness: Vec<_> = ordinal.iter().map(|&q| q / 8.0).collect();
        let omega = 2.0;
        let cell_volume = 0.125;
        let potential_volume: Vec<_> = ordinal.iter().map(|&q| q / 16.0 * cell_volume).collect();
        let derivative_volume: Vec<_> = slowness
            .iter()
            .map(|&value| helmholtz_slowness_derivative(omega, value) * cell_volume)
            .collect();
        let terms = BornReceiverTerms {
            incident: &incident,
            potential_volume: &potential_volume,
            derivative_volume: &derivative_volume,
        };
        drop(arrange);
        for source_scale in [Complex64::new(1.0, 0.0), Complex64::new(1.5, -0.5)] {
            let scaled_incident: Vec<_> =
                incident.iter().map(|&value| source_scale * value).collect();
            let mut gradient =
                leto::Array3::from_shape_vec([shape.0, shape.1, shape.2], ordinal.clone()).unwrap();
            let mut expected = gradient.clone();
            for residual in [Complex64::new(0.5, -0.25), Complex64::new(-0.25, 0.75)] {
                {
                    let _act = tracing::info_span!("act").entered();
                    terms.accumulate_adjoint::<K>(
                        gradient.as_slice_mut().unwrap(),
                        &scaled_incident,
                        residual,
                        &green,
                    );
                }
                let _assert = tracing::info_span!("assert").entered();
                for (k, _) in &centers {
                    let incident = incident[*k];
                    let value = green[*k];
                    let re = source_scale.re * incident.re - source_scale.im * incident.im;
                    let im = source_scale.re * incident.im + source_scale.im * incident.re;
                    let product_re = re * value.re - im * value.im;
                    let product_im = re * value.im + im * value.re;
                    let factor = 2.0 * omega * omega * slowness[*k] * cell_volume;
                    let index = [
                        *k / (shape.1 * shape.2),
                        (*k / shape.2) % shape.1,
                        *k % shape.2,
                    ];
                    expected[index] +=
                        factor * (product_re * residual.re + product_im * residual.im);
                }
                for (&actual, &reference) in gradient.iter().zip(expected.iter()) {
                    assert_eq!(actual.to_bits(), reference.to_bits());
                }
            }
        }
    }
}

#[test]
fn dense_adjoint_matches_coordinate_oracle() {
    check_adjoint::<1>();
    check_adjoint::<4>();
    check_adjoint::<8>();
    check_adjoint::<16>();
}
