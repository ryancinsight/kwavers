//! Tests for optical polarization physics

use super::*;
use crate::domain::grid::Grid;
use approx::assert_relative_eq;
use num_complex::Complex64;

#[test]
fn test_jones_vector_construction() {
    let horizontal = JonesVector::horizontal(1.0);
    assert_eq!(horizontal.ex, Complex64::new(1.0, 0.0));
    assert_eq!(horizontal.ey, Complex64::new(0.0, 0.0));

    let vertical = JonesVector::vertical(2.0);
    assert_eq!(vertical.ex, Complex64::new(0.0, 0.0));
    assert_eq!(vertical.ey, Complex64::new(2.0, 0.0));

    let right_circ = JonesVector::right_circular(2.0);
    let expected = 2.0 / std::f64::consts::SQRT_2;
    assert_relative_eq!(right_circ.ex.re, expected, epsilon = 1e-10);
    assert_relative_eq!(right_circ.ey.im, -expected, epsilon = 1e-10);

    let left_circ = JonesVector::left_circular(2.0);
    assert_relative_eq!(left_circ.ex.re, expected, epsilon = 1e-10);
    assert_relative_eq!(left_circ.ey.im, expected, epsilon = 1e-10);
}

#[test]
fn test_jones_vector_intensity() {
    let horizontal = JonesVector::horizontal(2.0);
    assert_relative_eq!(horizontal.intensity(), 2.0, epsilon = 1e-10);

    let vertical = JonesVector::vertical(2.0);
    assert_relative_eq!(vertical.intensity(), 2.0, epsilon = 1e-10);

    let circular = JonesVector::right_circular(2.0);
    assert_relative_eq!(circular.intensity(), 2.0, epsilon = 1e-10);
}

#[test]
fn test_jones_matrix_operations() {
    let identity = JonesMatrix::identity();
    let input = JonesVector::horizontal(1.0);
    let output = identity.apply(&input);
    assert_eq!(output.ex, input.ex);
    assert_eq!(output.ey, input.ey);

    let polarizer = JonesMatrix::horizontal_polarizer();
    let input = JonesVector::new(Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0));
    let output = polarizer.apply(&input);
    assert_eq!(output.ex, Complex64::new(1.0, 0.0));
    assert_eq!(output.ey, Complex64::new(0.0, 0.0));

    let polarizer = JonesMatrix::vertical_polarizer();
    let output = polarizer.apply(&input);
    assert_eq!(output.ex, Complex64::new(0.0, 0.0));
    assert_eq!(output.ey, Complex64::new(1.0, 0.0));
}

#[test]
fn test_waveplate_operations() {
    let qwp = JonesMatrix::quarter_wave_plate();
    let input = JonesVector::horizontal(1.0);
    let output = qwp.apply(&input);

    let sqrt_half = std::f64::consts::FRAC_1_SQRT_2;
    assert_relative_eq!(output.ex.re, sqrt_half, epsilon = 1e-10);
    assert_relative_eq!(output.ex.im, sqrt_half, epsilon = 1e-10);
    assert_relative_eq!(output.ey.re, sqrt_half, epsilon = 1e-10);
    assert_relative_eq!(output.ey.im, -sqrt_half, epsilon = 1e-10);

    let hwp = JonesMatrix::half_wave_plate();
    let input = JonesVector::horizontal(1.0);
    let output = hwp.apply(&input);
    assert!(output.intensity() > 0.0);
}

#[test]
fn test_jones_polarization_model() {
    let mut model = JonesPolarizationModel::new(500e-9);
    model.add_element(JonesMatrix::horizontal_polarizer());

    let mut fluence = ndarray::Array3::<f64>::zeros((2, 2, 2));
    fluence.fill(1.0);

    let mut polarization_state = ndarray::Array4::<Complex64>::zeros((2, 2, 2, 2));
    polarization_state.fill(Complex64::new(1.0, 0.0));

    let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();

    let medium = crate::domain::medium::HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
    model.apply_polarization(&mut fluence, &mut polarization_state, &grid, &medium);

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                assert_eq!(polarization_state[[0, i, j, k]], Complex64::new(1.0, 0.0));
                assert_eq!(polarization_state[[1, i, j, k]], Complex64::new(0.0, 0.0));
                assert_relative_eq!(fluence[[i, j, k]], 0.5, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_polarization_degree_calculation() {
    let horizontal = JonesVector::horizontal(1.0);
    assert_relative_eq!(horizontal.degree_of_polarization(), 1.0, epsilon = 1e-10);

    let vertical = JonesVector::vertical(1.0);
    assert_relative_eq!(vertical.degree_of_polarization(), 1.0, epsilon = 1e-10);

    let circular = JonesVector::right_circular(1.0);
    assert_relative_eq!(circular.degree_of_polarization(), 1.0, epsilon = 1e-10);

    let zero = JonesVector::new(Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0));
    assert_relative_eq!(zero.degree_of_polarization(), 0.0, epsilon = 1e-10);
}

#[test]
fn test_matrix_multiplication() {
    let rot1 = JonesMatrix::rotation(std::f64::consts::PI / 4.0);
    let rot2 = JonesMatrix::rotation(std::f64::consts::PI / 4.0);
    let combined = rot1.multiply(&rot2);

    let rot90 = JonesMatrix::rotation(std::f64::consts::PI / 2.0);

    assert_relative_eq!(combined.m11.re, rot90.m11.re, epsilon = 1e-10);
    assert_relative_eq!(combined.m12.re, rot90.m12.re, epsilon = 1e-10);
}
