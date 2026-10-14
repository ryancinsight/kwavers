//! Bitwise contracts for native operation lowering.

use eunomia::Complex;

#[test]
fn native_arithmetic_preserves_the_original_expression_tree() {
    use super::Arithmetic;
    let _subscriber =
        tracing::subscriber::set_default(super::super::super::gradient::test_support::subscriber());
    let _span = tracing::info_span!("native_arithmetic_expression_tree").entered();
    for q in [
        0.0,
        -0.0,
        f64::from_bits(1),
        f64::MIN_POSITIVE,
        1.0 / 3.0,
        -1.0 / 7.0,
    ] {
        let arrange = tracing::info_span!("arrange", q).entered();
        let incident = Complex::new(q, -1.0 / 7.0);
        let green = Complex::new(11.0 / 13.0, q);
        let weight = 11.0 / 13.0;
        let residual = Complex::new(-q, 1.0 / 3.0);
        let expected_product = incident * green;
        let expected_weighted = expected_product * weight;
        let expected_projection = (expected_weighted.conj() * residual).re;
        drop(arrange);
        for seed in [0.0, -0.0, 1.0 / 3.0] {
            let (product, weighted, projected, accumulated, sum) = {
                let _act = tracing::info_span!("act", seed).entered();
                let product = <f64 as Arithmetic>::product(incident, green);
                let mut weighted = [product];
                <f64 as Arithmetic>::weight_block(&mut weighted, &[weight]);
                let mut projected = [seed];
                let mut accumulated = [seed];
                <f64 as Arithmetic>::project_block::<1, false>(
                    &[product],
                    &[weight],
                    residual,
                    &mut projected,
                );
                <f64 as Arithmetic>::project_block::<1, true>(
                    &[product],
                    &[weight],
                    residual,
                    &mut accumulated,
                );
                (
                    product,
                    weighted[0],
                    projected[0],
                    accumulated[0],
                    <f64 as Arithmetic>::accumulate_block(Complex::new(seed, -seed), &weighted),
                )
            };
            let _assert = tracing::info_span!("assert", seed).entered();
            assert_eq!(product.re.to_bits(), expected_product.re.to_bits());
            assert_eq!(product.im.to_bits(), expected_product.im.to_bits());
            assert_eq!(weighted.re.to_bits(), expected_weighted.re.to_bits());
            assert_eq!(weighted.im.to_bits(), expected_weighted.im.to_bits());
            assert_eq!(projected.to_bits(), expected_projection.to_bits());
            assert_eq!(
                accumulated.to_bits(),
                (seed + expected_projection).to_bits()
            );
            let expected_sum = Complex::new(seed, -seed) + expected_weighted;
            assert_eq!(sum.re.to_bits(), expected_sum.re.to_bits());
            assert_eq!(sum.im.to_bits(), expected_sum.im.to_bits());
        }
    }
}
