use leto::Array2;

use super::{
    active_grid, build_fundamental_matrix, encode_measurements, fundamental_operator,
    solve_tikhonov_h1, EncodedOperator, LinearOperator, PcgSettings, PlanarPoint, RowMatrix,
    SameApertureMedium, SameApertureSettings,
};

#[test]
fn graph_laplacian_matches_edge_energy_identity() {
    let mask = Array2::from_elem((2, 2), true);
    let active = active_grid(&mask, 1.0);
    let values = [1.0_f32, 2.0, 4.0, 8.0];
    let mut laplacian = [0.0_f32; 4];
    active.graph_laplacian_into(&values, &mut laplacian);

    let energy = values
        .iter()
        .zip(laplacian.iter())
        .map(|(x, lx)| x * lx)
        .sum::<f32>();
    let edge_energy = (values[0] - values[1]).powi(2)
        + (values[0] - values[2]).powi(2)
        + (values[1] - values[3]).powi(2)
        + (values[2] - values[3]).powi(2);

    assert_eq!(energy, edge_energy);
    assert!(energy >= 0.0);
}

#[test]
fn row_matrix_products_match_manual_linear_algebra() {
    let mut matrix = RowMatrix::zeros(2, 3);
    matrix.row_mut(0).copy_from_slice(&[1.0, 2.0, 0.0]);
    matrix.row_mut(1).copy_from_slice(&[0.0, -1.0, 3.0]);
    let mut y = [0.0_f32; 2];
    matrix.matvec(&[2.0, 3.0, 5.0], &mut y);
    assert_eq!(y, [8.0, 12.0]);

    let mut x = [0.0_f32; 3];
    matrix.t_matvec(&[4.0, -2.0], &mut x);
    assert_eq!(x, [4.0, 10.0, -6.0]);
    assert_eq!(matrix.normal_diag(), vec![1.0, 5.0, 9.0]);
}

#[test]
fn finite_frequency_rows_are_normalized_and_input_sensitive() {
    let mask = Array2::from_elem((3, 3), true);
    let active = active_grid(&mask, 0.001);
    let attenuation = Array2::from_shape_fn((3, 3), |[ix, iy]| 0.1 + 0.01 * (ix + iy) as f64);
    let medium = SameApertureMedium {
        attenuation_np_per_m_mhz: &attenuation,
        spacing_m: 0.001,
    };
    let therapy = [
        PlanarPoint {
            x_m: -0.004,
            y_m: 0.0,
        },
        PlanarPoint {
            x_m: 0.003,
            y_m: 0.001,
        },
        PlanarPoint {
            x_m: 0.005,
            y_m: -0.003,
        },
    ];
    let settings = SameApertureSettings {
        frequencies_hz: &[500_000.0],
        receiver_offsets: &[1],
        phase_speed_m_s: super::C_REF_M_S,
    };
    let matrix = build_fundamental_matrix(medium, &therapy, &active, settings);

    assert_eq!(matrix.rows, 3);
    assert_eq!(matrix.cols, 9);
    for row in 0..matrix.rows {
        let norm = matrix
            .row(row)
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() <= 1.0e-5, "row {row} norm={norm}");
    }
    assert_ne!(matrix.row(0), matrix.row(1));
}

#[test]
fn phase_speed_changes_pitch_catch_phase_without_changing_row_normalization() {
    let mask = Array2::from_elem((4, 4), true);
    let active = active_grid(&mask, 0.001);
    let attenuation = Array2::from_elem((4, 4), 0.06);
    let medium = SameApertureMedium {
        attenuation_np_per_m_mhz: &attenuation,
        spacing_m: 0.001,
    };
    let therapy = [
        PlanarPoint {
            x_m: -0.004,
            y_m: -0.002,
        },
        PlanarPoint {
            x_m: 0.004,
            y_m: -0.002,
        },
    ];
    let acoustic_settings = SameApertureSettings {
        frequencies_hz: &[1_000.0],
        receiver_offsets: &[1],
        phase_speed_m_s: super::C_REF_M_S,
    };
    let shear_settings = SameApertureSettings {
        frequencies_hz: &[1_000.0],
        receiver_offsets: &[1],
        phase_speed_m_s: 2.5,
    };
    let acoustic = build_fundamental_matrix(medium, &therapy, &active, acoustic_settings);
    let shear = build_fundamental_matrix(medium, &therapy, &active, shear_settings);
    let max_difference = acoustic
        .row(0)
        .iter()
        .zip(shear.row(0).iter())
        .map(|(a, s)| (a - s).abs())
        .fold(0.0_f32, f32::max);

    assert_eq!(acoustic.rows(), shear.rows());
    assert_eq!(acoustic.cols(), shear.cols());
    assert!(
        max_difference > 1.0e-4,
        "phase-speed change must alter pitch-catch row values, max_difference={max_difference}"
    );
    for matrix in [&acoustic, &shear] {
        for row in 0..matrix.rows() {
            let norm = matrix
                .row(row)
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            assert!((norm - 1.0).abs() <= 1.0e-5, "row {row} norm={norm}");
        }
    }
}

#[test]
fn matrix_free_operator_matches_materialized_rows() {
    let mask = Array2::from_elem((4, 4), true);
    let active = active_grid(&mask, 0.001);
    let attenuation = Array2::from_shape_fn((4, 4), |[ix, iy]| 0.08 + 0.02 * (ix + iy) as f64);
    let medium = SameApertureMedium {
        attenuation_np_per_m_mhz: &attenuation,
        spacing_m: 0.001,
    };
    let therapy = [
        PlanarPoint {
            x_m: -0.005,
            y_m: -0.001,
        },
        PlanarPoint {
            x_m: 0.002,
            y_m: 0.004,
        },
        PlanarPoint {
            x_m: 0.006,
            y_m: -0.002,
        },
    ];
    let settings = SameApertureSettings {
        frequencies_hz: &[250_000.0, 500_000.0],
        receiver_offsets: &[1, 2],
        phase_speed_m_s: super::C_REF_M_S,
    };
    let matrix = build_fundamental_matrix(medium, &therapy, &active, settings);
    let operator = fundamental_operator(medium, &therapy, &active, settings);
    let x = (0..(active.len()))
        .map(|idx| 0.25 + 0.03125 * idx as f32)
        .collect::<Vec<_>>();
    let y = (0..matrix.rows())
        .map(|idx| -0.5 + 0.0625 * idx as f32)
        .collect::<Vec<_>>();
    let mut dense_ax = vec![0.0; matrix.rows()];
    let mut free_ax = vec![0.0; operator.rows()];
    let mut dense_aty = vec![0.0; matrix.cols()];
    let mut free_aty = vec![0.0; operator.cols()];

    matrix.matvec(&x, &mut dense_ax);
    operator.matvec(&x, &mut free_ax);
    matrix.t_matvec(&y, &mut dense_aty);
    operator.t_matvec(&y, &mut free_aty);

    assert_vec_close(&dense_ax, &free_ax, 1.0e-6);
    assert_vec_close(&dense_aty, &free_aty, 1.0e-5);
    assert_vec_close(&matrix.normal_diag(), &operator.normal_diag(), 1.0e-5);
    assert!(operator.storage_values() < matrix.storage_values());
    assert_eq!(operator.dense_values(), matrix.storage_values());
}

#[test]
fn encoded_operator_matches_materialized_source_encoding() {
    let mask = Array2::from_elem((4, 4), true);
    let active = active_grid(&mask, 0.001);
    let attenuation = Array2::from_shape_fn((4, 4), |[ix, iy]| 0.06 + 0.015 * (ix + iy) as f64);
    let medium = SameApertureMedium {
        attenuation_np_per_m_mhz: &attenuation,
        spacing_m: 0.001,
    };
    let therapy = [
        PlanarPoint {
            x_m: -0.005,
            y_m: -0.001,
        },
        PlanarPoint {
            x_m: -0.001,
            y_m: 0.005,
        },
        PlanarPoint {
            x_m: 0.004,
            y_m: 0.003,
        },
        PlanarPoint {
            x_m: 0.006,
            y_m: -0.003,
        },
    ];
    let settings = SameApertureSettings {
        frequencies_hz: &[250_000.0, 500_000.0],
        receiver_offsets: &[1, 2],
        phase_speed_m_s: super::C_REF_M_S,
    };
    let matrix = build_fundamental_matrix(medium, &therapy, &active, settings);
    let operator = fundamental_operator(medium, &therapy, &active, settings);
    let encoded = EncodedOperator::deterministic_signs(operator, 3);
    let encoded_matrix = encoded.materialize();
    let x = (0..(active.len()))
        .map(|idx| -0.25 + 0.041_666_668 * idx as f32)
        .collect::<Vec<_>>();
    let y = (0..encoded.rows())
        .map(|idx| 0.125 - 0.03125 * idx as f32)
        .collect::<Vec<_>>();
    let mut dense_ax = vec![0.0; matrix.rows()];
    let mut encoded_ax = vec![0.0; encoded.rows()];
    let mut materialized_encoded_ax = vec![0.0; encoded.rows()];
    let mut encoded_aty = vec![0.0; encoded.cols()];
    let mut materialized_encoded_aty = vec![0.0; encoded.cols()];

    matrix.matvec(&x, &mut dense_ax);
    encoded.matvec(&x, &mut encoded_ax);
    encoded_matrix.matvec(&x, &mut materialized_encoded_ax);
    encoded.t_matvec(&y, &mut encoded_aty);
    encoded_matrix.t_matvec(&y, &mut materialized_encoded_aty);
    let encoded_data = encode_measurements(&encoded, &dense_ax);

    assert_eq!(encoded.encoding_spec().original_rows(), matrix.rows());
    assert_eq!(encoded.encoding_spec().encoded_rows(), encoded.rows());
    assert_eq!(encoded.encoding_spec().rows_per_code(), 3);
    assert!(encoded.rows() < matrix.rows());
    assert_eq!(encoded.dense_values(), encoded.rows() * encoded.cols());
    assert!(encoded.storage_values() < matrix.storage_values());
    assert_vec_close(&encoded_ax, &materialized_encoded_ax, 1.0e-6);
    assert_vec_close(&encoded_data, &encoded_ax, 1.0e-6);
    assert_vec_close(&encoded_aty, &materialized_encoded_aty, 1.0e-5);
    assert_vec_close(
        &encoded.normal_diag(),
        &encoded_matrix.normal_diag(),
        1.0e-5,
    );
}

#[test]
fn pcg_solves_regularized_identity_normal_equations() {
    let mask = Array2::from_elem((1, 2), true);
    let active = active_grid(&mask, 1.0);
    let mut matrix = RowMatrix::zeros(2, 2);
    matrix.row_mut(0).copy_from_slice(&[1.0, 0.0]);
    matrix.row_mut(1).copy_from_slice(&[0.0, 1.0]);
    let result = solve_tikhonov_h1(
        &matrix,
        &[1.0, -2.0],
        &active,
        PcgSettings {
            iterations: 4,
            regularization: 1.0,
            smoothness_weight: 0.0,
            noise_fraction: 0.0,
        },
    );

    assert!((result.model[0] - 0.5).abs() <= 1.0e-6);
    assert!((result.model[1] + 1.0).abs() <= 1.0e-6);
    assert!((result.objective_history.len()) >= 2);
    assert!(result.objective_history[1] <= result.objective_history[0]);
}

fn assert_vec_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!((actual.len()), (expected.len()));
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*a - *e).abs() <= tolerance,
            "idx={idx}, actual={a}, expected={e}, tolerance={tolerance}"
        );
    }
}
