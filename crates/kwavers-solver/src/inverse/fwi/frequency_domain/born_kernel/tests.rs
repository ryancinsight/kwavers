//! Exact scalar and differential arithmetic contracts.

use super::fields::{FactoredProducts, StoredProducts};
use super::*;

/// A cubic voxel has volume `(h*h)*h`; two IEEE products fix its rounding.
/// The spacings come from the existing FD reconstruction/gradient fixtures.
#[test]
fn cell_volume_has_repeatable_ieee_products() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("cell_volume_has_repeatable_ieee_products").entered();
    for spacing in [0.001, 0.005] {
        let arrange = tracing::info_span!("arrange", spacing).entered();
        let grid = super::super::cbs::GridSpec::new((12, 12, 1), spacing).unwrap();
        let expected = (spacing * spacing) * spacing;
        drop(arrange);
        for _ in 0..16 {
            let actual = {
                let _act = tracing::info_span!("act").entered();
                grid.cell_volume_m3()
            };
            let _assert = tracing::info_span!("assert").entered();
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

/// `(1+2i)(3-4i)/2 = 11/2+i`; every operand/result is dyadic and exactly
/// representable in each tested native precision. No widened oracle is used.
fn check_scalar_contract<T: Arithmetic>() {
    let arrange = tracing::info_span!("arrange", scalar = std::any::type_name::<T>()).entered();
    let two = T::from_count(2);
    let incident = Complex {
        re: T::ONE,
        im: two,
    };
    let green = Complex {
        re: T::from_count(3),
        im: -T::from_count(4),
    };
    let weight = T::ONE / two;
    let expected = Complex {
        re: T::from_count(11) / two,
        im: T::ONE,
    };
    drop(arrange);
    let (actual, integrated, projection, accumulated) = {
        let _act = tracing::info_span!("act").entered();
        let mut actual = [Complex {
            re: T::ZERO,
            im: T::ZERO,
        }];
        weighted_products(
            &FactoredProducts::new(&[incident], &[green]),
            0..1,
            &[weight],
            &mut actual,
            &mut Complex {
                re: T::ZERO,
                im: T::ZERO,
            },
        );
        let mut projected = [T::ZERO];
        project_adjoint::<_, _, 1, false>(
            &StoredProducts::new(&actual),
            0..actual.len(),
            &[T::ONE; 1][..actual.len()],
            Complex {
                re: two,
                im: -T::ONE,
            },
            &mut [Complex {
                re: T::ZERO,
                im: T::ZERO,
            }; 1],
            &mut projected,
        );
        let mut accumulated = [T::from_count(3)];
        project_adjoint::<_, _, 1, true>(
            &StoredProducts::new(&actual),
            0..actual.len(),
            &[T::ONE; 1][..actual.len()],
            Complex {
                re: two,
                im: -T::ONE,
            },
            &mut [Complex {
                re: T::ZERO,
                im: T::ZERO,
            }; 1],
            &mut accumulated,
        );
        let [actual] = actual;
        (
            actual,
            integrate::<_, _, 1>(
                &FactoredProducts::new(&[incident; 17], &[green; 17]),
                &[weight; 17],
            ),
            projected[0],
            accumulated[0],
        )
    };
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(actual, expected);
    for integrated in [
        integrate::<_, _, 4>(
            &FactoredProducts::new(&[incident; 17], &[green; 17]),
            &[weight; 17],
        ),
        integrate::<_, _, 8>(
            &FactoredProducts::new(&[incident; 17], &[green; 17]),
            &[weight; 17],
        ),
        integrate::<_, _, 16>(
            &FactoredProducts::new(&[incident; 17], &[green; 17]),
            &[weight; 17],
        ),
    ] {
        assert_eq!(
            integrated,
            Complex {
                re: T::from_count(187) / two,
                im: T::from_count(17)
            }
        );
    }

    assert_eq!(
        integrated,
        Complex {
            re: T::from_count(187) / two,
            im: T::from_count(17)
        }
    );
    assert_eq!(projection, T::from_count(10));
    assert_eq!(accumulated, T::from_count(13));
}

#[test]
fn weighted_integration_preserves_native_precision() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("weighted_integration_preserves_native_precision").entered();
    check_scalar_contract::<f64>();
    check_scalar_contract::<f32>();
    check_scalar_contract::<eunomia::F16>();
    check_scalar_contract::<eunomia::Bf16>();
}

/// Block inputs vary by exact quarter-turns and signs, so a reused or reordered
/// lane cannot satisfy the complex-operation oracle.
fn check_block_expression<const K: usize>(
    incident: Complex<f64>,
    green: Complex<f64>,
    weight: f64,
    residual: Complex<f64>,
) {
    let _arrange = tracing::info_span!("arrange_block", K).entered();
    let incidents: [_; K] = std::array::from_fn(|lane| {
        if lane % 2 == 0 {
            incident
        } else {
            Complex::new(incident.im, -incident.re)
        }
    });
    let greens: [_; K] = std::array::from_fn(|lane| {
        if lane % 3 == 0 {
            green
        } else {
            Complex::new(-green.im, green.re)
        }
    });
    let weights: [_; K] = std::array::from_fn(|lane| if lane % 2 == 0 { weight } else { -weight });
    drop(_arrange);
    let (actual, projected) = {
        let _act = tracing::info_span!("act_block", K).entered();
        let mut actual = [Complex::new(0.0, 0.0); K];
        let mut projected = [0.0; K];
        weighted_products(
            &FactoredProducts::new(&incidents, &greens),
            0..K,
            &weights,
            &mut actual,
            &mut Complex::new(0.0, 0.0),
        );
        project_adjoint::<_, _, K, false>(
            &StoredProducts::new(&actual),
            0..actual.len(),
            &[1.0; K][..actual.len()],
            residual,
            &mut [Complex::new(0.0, 0.0); K],
            &mut projected,
        );
        (actual, projected)
    };
    let _assert = tracing::info_span!("assert_block", K).entered();
    for lane in 0..K {
        let expected = incidents[lane] * greens[lane] * weights[lane];
        assert_eq!(actual[lane].re.to_bits(), expected.re.to_bits());
        assert_eq!(actual[lane].im.to_bits(), expected.im.to_bits());
        assert_eq!(
            projected[lane].to_bits(),
            (expected.conj() * residual).re.to_bits()
        );
    }
}

/// Bitwise differential checks include non-dyadic operands, signed zero and
/// subnormals. The existing complex API defines the exact expression tree.
#[test]
fn weighted_product_and_projection_match_complex_operations() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("weighted_product_matches_complex_operations").entered();
    for q in [
        0.0,
        -0.0,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        1.0 / 3.0,
        -1.0 / 7.0,
        11.0 / 13.0,
    ] {
        for incident in [Complex::new(q, 1.0 / 3.0), Complex::new(-q, -0.0)] {
            for green in [Complex::new(-1.0 / 7.0, q), Complex::new(q, -q)] {
                let arrange = tracing::info_span!("arrange", q).entered();
                let weight = 11.0 / 13.0;
                let reference = incident * green * weight;
                let residual = Complex::new(-q, 1.0 / 3.0);
                drop(arrange);
                let (actual, projection) = {
                    let _act = tracing::info_span!("act").entered();
                    let mut actual = [Complex::new(0.0, 0.0)];
                    let mut projected = [0.0];
                    weighted_products(
                        &FactoredProducts::new(&[incident], &[green]),
                        0..1,
                        &[weight],
                        &mut actual,
                        &mut Complex::new(0.0, 0.0),
                    );
                    project_adjoint::<_, _, 1, false>(
                        &StoredProducts::new(&actual),
                        0..actual.len(),
                        &[1.0; 1][..actual.len()],
                        residual,
                        &mut [Complex::new(0.0, 0.0); 1],
                        &mut projected,
                    );
                    (actual[0], projected[0])
                };
                let _assert = tracing::info_span!("assert").entered();
                assert_eq!(actual.re.to_bits(), reference.re.to_bits());
                assert_eq!(actual.im.to_bits(), reference.im.to_bits());
                assert_eq!(
                    projection.to_bits(),
                    (reference.conj() * residual).re.to_bits()
                );
                check_block_expression::<4>(incident, green, weight, residual);
                check_block_expression::<8>(incident, green, weight, residual);
                check_block_expression::<16>(incident, green, weight, residual);
            }
        }
    }
}

/// Partial blocks consume precisely the valid prefix and leave all spare
/// output lanes unchanged. The quarter-turn fields give each lane a distinct
/// exact dyadic value; the existing complex operations define the oracle.
fn check_partial_blocks<const K: usize>() {
    let incident: [_; K] = std::array::from_fn(|lane| {
        let q = (lane + 1) as f64;
        Complex::new(q / 2.0, -q / 4.0)
    });
    let green: [_; K] = std::array::from_fn(|lane| {
        let q = (lane + 1) as f64;
        Complex::new(-q / 8.0, q / 16.0)
    });
    let weights = [0.5; K];
    let residual = Complex::new(0.5, -0.25);
    let spare = Complex::new(11.0 / 2.0, -7.0 / 4.0);
    for len in [0, 1, K - 1, K] {
        let mut values = [spare; K];
        let mut projected = [13.0 / 8.0; K];
        let valid = weighted_products(
            &FactoredProducts::new(&incident[..len], &green[..len]),
            0..len,
            &weights[..len],
            &mut values,
            &mut Complex::new(0.0, 0.0),
        );
        assert_eq!(valid.len(), len);
        let projected_prefix = project_adjoint::<_, _, K, false>(
            &StoredProducts::new(valid),
            0..(valid).len(),
            &[1.0; K][..(valid).len()],
            residual,
            &mut [Complex::new(0.0, 0.0); K],
            &mut projected,
        );
        assert_eq!(projected_prefix.len(), len);
        let mut sum = Complex::new(0.0, 0.0);
        for lane in 0..len {
            let expected = incident[lane] * green[lane] * weights[lane];
            assert_eq!(valid[lane].re.to_bits(), expected.re.to_bits());
            assert_eq!(valid[lane].im.to_bits(), expected.im.to_bits());
            assert_eq!(
                projected_prefix[lane].to_bits(),
                (expected.conj() * residual).re.to_bits()
            );
            sum += expected;
        }
        let integrated = integrate::<_, _, K>(
            &FactoredProducts::new(&incident[..len], &green[..len]),
            &weights[..len],
        );
        assert_eq!(integrated.re.to_bits(), sum.re.to_bits());
        assert_eq!(integrated.im.to_bits(), sum.im.to_bits());
        assert!(values[len..].iter().all(|&value| value == spare));
        assert!(projected[len..].iter().all(|&value| value == 13.0 / 8.0));
    }
}

#[test]
fn checked_partial_blocks_preserve_values_and_spare_storage() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("checked_partial_blocks").entered();
    check_partial_blocks::<4>();
    check_partial_blocks::<8>();
    check_partial_blocks::<16>();
    check_partial_blocks::<256>();
}

#[test]
#[should_panic(expected = "Born field/Green lengths differ")]
fn weighted_products_reject_mismatched_fields() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("act_reject_mismatched_fields").entered();
    let mut output = [Complex::new(0.0, 0.0); 4];
    weighted_products(
        &FactoredProducts::new(&[Complex::new(1.0, 0.5)], &[]),
        0..1,
        &[0.5],
        &mut output,
        &mut Complex::new(0.0, 0.0),
    );
}

#[test]
#[should_panic(expected = "Born field/weight lengths differ")]
fn weighted_products_reject_mismatched_weights() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("act_reject_mismatched_weights").entered();
    let field = [Complex::new(1.0, 0.5)];
    let mut output = [Complex::new(0.0, 0.0); 4];
    weighted_products(
        &FactoredProducts::new(&field, &field),
        0..1,
        &[],
        &mut output,
        &mut Complex::new(0.0, 0.0),
    );
}

#[test]
#[should_panic(expected = "Born block exceeds output capacity")]
fn weighted_products_reject_excess_capacity() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("act_reject_excess_capacity").entered();
    let field = [Complex::new(1.0, 0.5); 2];
    weighted_products(
        &FactoredProducts::new(&field, &field),
        0..2,
        &[0.5; 2],
        &mut [Complex::new(0.0, 0.0)],
        &mut Complex::new(0.0, 0.0),
    );
}

#[test]
#[should_panic(expected = "Born block exceeds output capacity")]
fn adjoint_projection_rejects_excess_capacity() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("act_reject_projection_capacity").entered();
    project_adjoint::<_, _, 1, false>(
        &StoredProducts::new(&[Complex::new(1.0, 0.5); 2]),
        0..[Complex::new(1.0, 0.5); 2].len(),
        &[1.0; 2],
        Complex::new(0.5, -0.25),
        &mut [Complex::new(0.0, 0.0); 1],
        &mut [0.0],
    );
}

/// Assignment exposes the projection bits; accumulation adds that exact value
/// once to the existing gradient. Spare lanes never enter either operation.
#[test]
fn projection_modes_preserve_exact_accumulation_and_spare_storage() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("projection_modes").entered();
    for q in [0.0, -0.0, f64::from_bits(1), f64::MIN_POSITIVE, 1.0 / 3.0] {
        let arrange = tracing::info_span!("arrange", q).entered();
        let derivatives = std::array::from_fn::<_, 8, _>(|lane| {
            if lane % 2 == 0 {
                Complex::new(q, -q)
            } else {
                Complex::new(-q, 1.0 / 7.0)
            }
        });
        let initial =
            std::array::from_fn::<_, 8, _>(|lane| if lane % 2 == 0 { -0.0 } else { 11.0 / 13.0 });
        let residual = Complex::new(-1.0 / 7.0, 1.0 / 3.0);
        drop(arrange);
        for len in [0, 1, 7, 8] {
            let mut assigned = initial;
            let mut accumulated = initial;
            let (assigned_len, accumulated_len) = {
                let _act = tracing::info_span!("act", len).entered();
                (
                    project_adjoint::<_, _, 8, false>(
                        &StoredProducts::new(&derivatives[..len]),
                        0..derivatives[..len].len(),
                        &[1.0; 8][..derivatives[..len].len()],
                        residual,
                        &mut [Complex::new(0.0, 0.0); 8],
                        &mut assigned,
                    )
                    .len(),
                    project_adjoint::<_, _, 8, true>(
                        &StoredProducts::new(&derivatives[..len]),
                        0..derivatives[..len].len(),
                        &[1.0; 8][..derivatives[..len].len()],
                        residual,
                        &mut [Complex::new(0.0, 0.0); 8],
                        &mut accumulated,
                    )
                    .len(),
                )
            };
            let _assert = tracing::info_span!("assert", len).entered();
            assert_eq!(assigned_len, len);
            assert_eq!(accumulated_len, len);
            for lane in 0..len {
                let expected = (derivatives[lane].conj() * residual).re;
                assert_eq!(assigned[lane].to_bits(), expected.to_bits());
                assert_eq!(
                    accumulated[lane].to_bits(),
                    (initial[lane] + expected).to_bits()
                );
            }
            for lane in len..8 {
                assert_eq!(assigned[lane].to_bits(), initial[lane].to_bits());
                assert_eq!(accumulated[lane].to_bits(), initial[lane].to_bits());
            }
        }
    }
}
