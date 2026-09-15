use super::{
    accumulate_stratum, apply_pressure_absorption, build_weighted_divergence,
    multiply_spectral_operator,
};
use kwavers_math::fft::Complex64;
use leto::Array3;

/// One volume below the lane walker's parallel floor and one above it.
const SHAPES: [[usize; 3]; 2] = [[5, 3, 7], [37, 29, 31]];

fn real_field(shape: [usize; 3], seed: f64) -> Array3<f64> {
    let values = (0..shape.iter().product::<usize>())
        .map(|index| (index as f64).mul_add(0.754_8, seed).sin())
        .collect();
    Array3::from_shape_vec(shape, values).expect("values match the shape")
}

fn same_bits(actual: &Array3<f64>, expected: &[f64]) -> bool {
    actual
        .iter()
        .zip(expected)
        .all(|(a, b)| a.to_bits() == b.to_bits())
}

// Each lane kernel computes the expression of the per-element loop it
// replaced, in the same order, so the results agree to the bit.

#[test]
fn weighted_divergence_is_the_per_element_formula_to_the_bit() {
    for shape in SHAPES {
        let (x, y, z, rho) = (
            real_field(shape, 1.0),
            real_field(shape, 2.0),
            real_field(shape, 3.0),
            real_field(shape, 4.0),
        );
        let expected: Vec<f64> = (0..x.len())
            .map(|i| {
                let [x, y, z, rho] = [&x, &y, &z, &rho].map(|a| a.as_slice().expect("owned")[i]);
                rho * (x + y + z)
            })
            .collect();
        let mut output = real_field(shape, 5.0);
        build_weighted_divergence(&mut output, &x, &y, &z, &rho);
        assert!(
            same_bits(&output, &expected),
            "weighted divergence diverges at {shape:?}"
        );
    }
}

#[test]
fn spectral_operator_is_the_per_element_product_to_the_bit() {
    for shape in SHAPES {
        let operator = real_field(shape, 1.0);
        let mut spectrum = real_field(shape, 2.0).mapv(|v| Complex64::new(v, 0.5 - v));
        let expected: Vec<Complex64> = spectrum
            .iter()
            .zip(operator.iter())
            .map(|(&value, &factor)| value * factor)
            .collect();
        multiply_spectral_operator(&mut spectrum, &operator);
        let same = spectrum
            .iter()
            .zip(&expected)
            .all(|(a, b)| a.re.to_bits() == b.re.to_bits() && a.im.to_bits() == b.im.to_bits());
        assert!(same, "spectral operator diverges at {shape:?}");
    }
}

#[test]
fn stratum_accumulation_is_the_per_element_blend_to_the_bit() {
    for shape in SHAPES {
        let values = real_field(shape, 1.0);
        let weight_hi = real_field(shape, 2.0).mapv(|v| v.abs());
        let lower: Vec<u32> = (0..values.len()).map(|i| (i % 3) as u32).collect();
        let bracket_lo = Array3::from_shape_vec(shape, lower).expect("values match the shape");
        for stratum in 0..3_u32 {
            let initial = real_field(shape, 3.0);
            let expected: Vec<f64> = (0..values.len())
                .map(|i| {
                    let lower = bracket_lo.as_slice().expect("owned")[i];
                    let hi = weight_hi.as_slice().expect("owned")[i];
                    let weight = if lower == stratum {
                        1.0 - hi
                    } else if lower + 1 == stratum {
                        hi
                    } else {
                        0.0
                    };
                    initial.as_slice().expect("owned")[i]
                        + weight * values.as_slice().expect("owned")[i]
                })
                .collect();
            let mut accumulator = initial.clone();
            accumulate_stratum(&mut accumulator, &values, &bracket_lo, &weight_hi, stratum);
            assert!(
                same_bits(&accumulator, &expected),
                "stratum {stratum} diverges at {shape:?}"
            );
        }
    }
}

#[test]
fn pressure_absorption_is_the_per_element_correction_to_the_bit() {
    for shape in SHAPES {
        let [c0, tau, eta, l1, l2] = [1.0, 2.0, 3.0, 4.0, 5.0].map(|seed| real_field(shape, seed));
        let initial = real_field(shape, 6.0);
        let expected: Vec<f64> = (0..initial.len())
            .map(|i| {
                let [c, tau, eta, l1, l2, p] =
                    [&c0, &tau, &eta, &l1, &l2, &initial].map(|a| a.as_slice().expect("owned")[i]);
                p + c * c * tau.mul_add(l1, -(eta * l2))
            })
            .collect();
        let mut pressure = initial.clone();
        apply_pressure_absorption(&mut pressure, &c0, &tau, &eta, &l1, &l2);
        assert!(
            same_bits(&pressure, &expected),
            "pressure absorption diverges at {shape:?}"
        );
    }
}
