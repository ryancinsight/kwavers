use super::{accumulate_split_density, apply_linear_eos, apply_nonlinear_eos};
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
fn split_density_sum_is_the_per_element_formula_to_the_bit() {
    for shape in SHAPES {
        let [rhox, rhoy, rhoz] = [1.0, 2.0, 3.0].map(|seed| real_field(shape, seed));
        let mut div_u = real_field(shape, 4.0);
        accumulate_split_density(&mut div_u, &rhox, &rhoy, &rhoz);
        let same = (0..div_u.len()).all(|i| {
            at(&div_u, i).to_bits() == (at(&rhox, i) + at(&rhoy, i) + at(&rhoz, i)).to_bits()
        });
        assert!(same, "split density sum diverges at {shape:?}");
    }
}

#[test]
fn nonlinear_eos_is_the_per_element_formula_to_the_bit() {
    for shape in SHAPES {
        let div_u = real_field(shape, 1.0);
        let c0 = real_field(shape, 2.0).mapv(|v| v.mul_add(10.0, 1500.0));
        let bon = real_field(shape, 3.0).mapv(|v| v.mul_add(0.5, 3.5));
        let rho0 = real_field(shape, 4.0).mapv(|v| v.mul_add(20.0, 1000.0));
        let mut pressure = real_field(shape, 5.0);
        apply_nonlinear_eos(&mut pressure, &div_u, &c0, &bon, &rho0);
        let same = (0..pressure.len()).all(|i| {
            let rho_sum = at(&div_u, i);
            let nonlinear = (at(&bon, i) / (2.0 * at(&rho0, i))) * rho_sum * rho_sum;
            let c = at(&c0, i);
            at(&pressure, i).to_bits() == (c * c * (rho_sum + nonlinear)).to_bits()
        });
        assert!(same, "nonlinear EOS diverges at {shape:?}");
    }
}

#[test]
fn linear_eos_is_the_per_element_formula_to_the_bit() {
    for shape in SHAPES {
        let [rhox, rhoy, rhoz] = [1.0, 2.0, 3.0].map(|seed| real_field(shape, seed));
        let c0 = real_field(shape, 4.0).mapv(|v| v.mul_add(10.0, 1500.0));
        let mut div_u = real_field(shape, 5.0);
        let mut pressure = real_field(shape, 6.0);
        apply_linear_eos(&mut div_u, &mut pressure, &rhox, &rhoy, &rhoz, &c0);
        let same = (0..div_u.len()).all(|i| {
            let rho_sum = at(&rhox, i) + at(&rhoy, i) + at(&rhoz, i);
            let c = at(&c0, i);
            at(&div_u, i).to_bits() == rho_sum.to_bits()
                && at(&pressure, i).to_bits() == (c * c * rho_sum).to_bits()
        });
        assert!(same, "linear EOS diverges at {shape:?}");
    }
}
