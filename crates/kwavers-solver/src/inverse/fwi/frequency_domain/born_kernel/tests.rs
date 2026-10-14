//! Exact scalar and differential arithmetic contracts.

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
fn check_scalar_contract<T: FloatElement + Neg<Output = T>>() {
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
    let (actual, integrated, projection) = {
        let _act = tracing::info_span!("act").entered();
        let actual = weighted_product(incident, green, weight);
        (
            actual,
            integrate::<_, 1>(&[incident; 17], &[green; 17], &[weight; 17]),
            project_adjoint(
                actual,
                Complex {
                    re: two,
                    im: -T::ONE,
                },
            ),
        )
    };
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(actual, expected);
    for integrated in [
        integrate::<_, 4>(&[incident; 17], &[green; 17], &[weight; 17]),
        integrate::<_, 8>(&[incident; 17], &[green; 17], &[weight; 17]),
        integrate::<_, 16>(&[incident; 17], &[green; 17], &[weight; 17]),
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
                    let actual = weighted_product(incident, green, weight);
                    (actual, project_adjoint(actual, residual))
                };
                let _assert = tracing::info_span!("assert").entered();
                assert_eq!(actual.re.to_bits(), reference.re.to_bits());
                assert_eq!(actual.im.to_bits(), reference.im.to_bits());
                assert_eq!(
                    projection.to_bits(),
                    (reference.conj() * residual).re.to_bits()
                );
            }
        }
    }
}
