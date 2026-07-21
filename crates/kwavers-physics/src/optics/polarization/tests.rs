//! Tests for optical polarization physics

use super::*;
use eunomia::assert_relative_eq;
use eunomia::Complex64;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_grid::Grid;

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

    // QWP(45°) on H-input → circular: Ex=(1+i)/2, Ey=(1-i)/2
    // Intensity conserved: (|Ex|²+|Ey|²)/2 = (0.5+0.5)/2 = 0.5 = input.intensity() ✓
    assert_relative_eq!(output.ex.re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(output.ex.im, 0.5, epsilon = 1e-10);
    assert_relative_eq!(output.ey.re, 0.5, epsilon = 1e-10);
    assert_relative_eq!(output.ey.im, -0.5, epsilon = 1e-10);
    assert_relative_eq!(output.intensity(), input.intensity(), epsilon = 1e-10);

    // HWP (fast axis 0°) on H-polarised input: [[1,0],[0,-1]]×[1,0] = [1,0]  (unchanged)
    let hwp = JonesMatrix::half_wave_plate();
    let h_in = JonesVector::horizontal(1.0);
    let h_out = hwp.apply(&h_in);
    assert_relative_eq!(h_out.ex.re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(h_out.ex.im, 0.0, epsilon = 1e-10);
    assert_relative_eq!(h_out.ey.re, 0.0, epsilon = 1e-10);
    assert_relative_eq!(h_out.ey.im, 0.0, epsilon = 1e-10);
    assert_relative_eq!(h_out.intensity(), h_in.intensity(), epsilon = 1e-10);
    // HWP (fast axis 0°) on V-polarised input: [[1,0],[0,-1]]×[0,1] = [0,-1]  (sign flip, same intensity)
    let v_in = JonesVector::vertical(1.0);
    let v_out = hwp.apply(&v_in);
    assert_relative_eq!(v_out.ex.re, 0.0, epsilon = 1e-10);
    assert_relative_eq!(v_out.ey.re, -1.0, epsilon = 1e-10);
    assert_relative_eq!(v_out.intensity(), v_in.intensity(), epsilon = 1e-10);
}

#[test]
fn test_jones_polarization_model() {
    let mut model = JonesPolarizationModel::new(500e-9);
    model.add_element(JonesMatrix::horizontal_polarizer());

    let mut fluence = leto::Array3::<f64>::zeros((2, 2, 2));
    fluence.fill(1.0);

    let mut polarization_state = leto::Array4::<Complex64>::zeros((2, 2, 2, 2));
    polarization_state.fill(Complex64::new(1.0, 0.0));

    let grid = Grid::new(2, 2, 2, 1.0, 1.0, 1.0).unwrap();

    let medium =
        kwavers_medium::HomogeneousMedium::new(1000.0, SOUND_SPEED_WATER_SIM, 0.5, 1.0, &grid);
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
