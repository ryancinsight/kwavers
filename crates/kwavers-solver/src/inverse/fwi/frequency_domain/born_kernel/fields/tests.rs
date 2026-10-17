//! Native-precision equivalence of spatial-product representations.

use super::super::arithmetic::Arithmetic;
use super::super::{integrate, integrate_products};
use super::*;

/// The spatial coefficient is independent of model weights. Factored and
/// prepared representations must agree with the original complex expression
/// for full blocks, nonzero offsets and a partial tail in native precision.
fn check_product_representations<T: Arithmetic>() {
    let arrange = tracing::info_span!("arrange_product_representations").entered();
    let two = T::from_count(2);
    let incident: [_; 9] = std::array::from_fn(|lane| {
        let q = T::from_count(lane + 1);
        Complex {
            re: q / two,
            im: -q / T::from_count(4),
        }
    });
    let green: [_; 9] = std::array::from_fn(|lane| {
        let q = T::from_count(lane + 1);
        Complex {
            re: -q / T::from_count(8),
            im: q / T::from_count(16),
        }
    });
    let weights: [_; 9] =
        std::array::from_fn(|lane| if lane % 3 == 0 { T::ZERO } else { T::ONE / two });
    let factored = FactoredProducts::new(&incident, &green);
    let mut prepared = [Complex {
        re: T::ZERO,
        im: T::ZERO,
    }; 9];
    factored.write_products(0..9, &mut prepared);
    let stored = StoredProducts::new(&prepared);
    drop(arrange);
    let (factored_sum, stored_sum) = {
        let _act = tracing::info_span!("act_product_representations").entered();
        (
            integrate::<_, _, 4>(&factored, &weights),
            integrate::<_, _, 4>(&stored, &weights),
        )
    };
    let _assert = tracing::info_span!("assert_product_representations").entered();
    let mut expected = Complex {
        re: T::ZERO,
        im: T::ZERO,
    };
    for lane in 0..9 {
        let product = incident[lane] * green[lane];
        assert_eq!(prepared[lane], product);
        expected += product * weights[lane];
    }
    assert_eq!(factored_sum, expected);
    assert_eq!(stored_sum, expected);
    let mut output = [Complex {
        re: T::ONE,
        im: -T::ONE,
    }; 8];
    let initial = Complex {
        re: T::ONE,
        im: -T::ONE,
    };
    let mut sum = initial;
    let prefix = integrate_products(&stored, 3..8, &weights[3..8], &mut output, &mut sum);
    assert_eq!(prefix.len(), 5);
    let mut expected_sum = initial;
    for lane in 0..5 {
        expected_sum += incident[lane + 3] * green[lane + 3] * weights[lane + 3];
        assert_eq!(prefix[lane], incident[lane + 3] * green[lane + 3]);
    }
    assert_eq!(sum, expected_sum);
    assert_eq!(
        output[5..],
        [Complex {
            re: T::ONE,
            im: -T::ONE
        }; 3]
    );
}

#[test]
fn product_representations_match_native_complex_arithmetic() {
    let _subscriber =
        tracing::subscriber::set_default(super::super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("product_representations").entered();
    check_product_representations::<f64>();
    check_product_representations::<f32>();
    check_product_representations::<eunomia::F16>();
    check_product_representations::<eunomia::Bf16>();
}
