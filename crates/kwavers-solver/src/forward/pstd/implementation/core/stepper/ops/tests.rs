use super::{
    add_density_source_components, add_gradient_source_term, add_masked_source_term,
    multiply_complex_by_real_field, scale_real_field,
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

fn at(field: &Array3<f64>, index: usize) -> f64 {
    field.as_slice().expect("owned arrays are contiguous")[index]
}

// Each lane kernel computes the expression of the per-element loop it
// replaced, in the same order, so the results agree to the bit.

#[test]
fn scaling_is_the_per_element_product_to_the_bit() {
    for shape in SHAPES {
        let initial = real_field(shape, 1.0);
        let mut field = initial.clone();
        scale_real_field(&mut field, 0.731);
        let same = (0..field.len())
            .all(|i| at(&field, i).to_bits() == (at(&initial, i) * 0.731).to_bits());
        assert!(same, "scaling diverges at {shape:?}");
    }
}

#[test]
fn masked_source_is_the_per_element_sum_to_the_bit() {
    for shape in SHAPES {
        // Every third element is masked out, so the skip branch is exercised.
        let mask = Array3::from_shape_vec(
            shape,
            (0..shape.iter().product::<usize>())
                .map(|index| {
                    if index % 3 == 0 {
                        0.0
                    } else {
                        (index as f64).mul_add(0.754_8, 2.0).sin()
                    }
                })
                .collect(),
        )
        .expect("values match the shape");
        let initial = real_field(shape, 3.0);
        let mut dst = initial.clone();
        add_masked_source_term(&mut dst, &mask, 1.25);
        let same = (0..dst.len()).all(|i| {
            let m = at(&mask, i);
            let expected = if m.abs() > 1e-12 {
                at(&initial, i) + m * 1.25
            } else {
                at(&initial, i)
            };
            at(&dst, i).to_bits() == expected.to_bits()
        });
        assert!(same, "masked source diverges at {shape:?}");
    }
}

#[test]
fn gradient_source_is_the_per_element_sum_to_the_bit() {
    for shape in SHAPES {
        let mask = real_field(shape, 4.0);
        let initial = real_field(shape, 5.0);
        let mut dst = initial.clone();
        add_gradient_source_term(&mut dst, &mask, -2.5);
        let same = (0..dst.len())
            .all(|i| at(&dst, i).to_bits() == (at(&initial, i) + at(&mask, i) * -2.5).to_bits());
        assert!(same, "gradient source diverges at {shape:?}");
    }
}

#[test]
fn complex_by_real_is_the_per_element_product_to_the_bit() {
    for shape in SHAPES {
        let factors = real_field(shape, 6.0);
        let initial = real_field(shape, 7.0).mapv(|v| Complex64::new(v, 0.5 - v));
        let mut field = initial.clone();
        multiply_complex_by_real_field(&mut field, &factors);
        let same = field
            .iter()
            .zip(initial.iter())
            .enumerate()
            .all(|(i, (a, b))| {
                let expected = *b * at(&factors, i);
                a.re.to_bits() == expected.re.to_bits() && a.im.to_bits() == expected.im.to_bits()
            });
        assert!(same, "complex by real diverges at {shape:?}");
    }
}

#[test]
fn density_source_is_the_per_element_sum_to_the_bit() {
    for shape in SHAPES {
        let source = real_field(shape, 5.0);
        let (rhox, rhoy, rhoz) = (
            real_field(shape, 6.0),
            real_field(shape, 7.0),
            real_field(shape, 8.0),
        );
        for (with_y, with_z) in [(true, true), (true, false), (false, true), (false, false)] {
            let (mut rx, mut ry, mut rz) = (rhox.clone(), rhoy.clone(), rhoz.clone());
            add_density_source_components(
                &mut rx,
                with_y.then_some(&mut ry),
                with_z.then_some(&mut rz),
                &source,
            );
            let same = (0..rx.len()).all(|i| {
                let s = at(&source, i);
                let y = if with_y {
                    at(&rhoy, i) + s
                } else {
                    at(&rhoy, i)
                };
                let z = if with_z {
                    at(&rhoz, i) + s
                } else {
                    at(&rhoz, i)
                };
                at(&rx, i).to_bits() == (at(&rhox, i) + s).to_bits()
                    && at(&ry, i).to_bits() == y.to_bits()
                    && at(&rz, i).to_bits() == z.to_bits()
            });
            assert!(
                same,
                "density source diverges at {shape:?} with y {with_y} and z {with_z}"
            );
        }
    }
}
