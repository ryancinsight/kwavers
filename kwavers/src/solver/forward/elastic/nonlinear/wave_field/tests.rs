use super::*;

#[test]
fn test_wave_field_creation() {
    let field = NonlinearElasticWaveField::new(10, 10, 10, 3);

    assert_eq!(field.u_fundamental.dim(), (10, 10, 10));
    assert_eq!(field.u_second.dim(), (10, 10, 10));
    assert_eq!(field.u_harmonics.len(), 1); // 3 total - 2 = 1 additional
    assert_eq!(field.num_harmonics(), 3);
    assert_eq!(field.time, 0.0);
    assert_eq!(field.frequency, 50.0);
}

#[test]
fn test_total_displacement_magnitude() {
    let field = NonlinearElasticWaveField::new(10, 10, 10, 3);

    let magnitude = field.total_displacement_magnitude();
    assert_eq!(magnitude.dim(), (10, 10, 10));

    for &val in magnitude.iter() {
        assert!((val - 0.0).abs() < 1e-10);
    }
}

#[test]
fn test_get_harmonic() {
    let mut field = NonlinearElasticWaveField::new(10, 10, 10, 4);

    field.u_fundamental[[0, 0, 0]] = 1.0;
    field.u_second[[0, 0, 0]] = 0.5;
    field.u_harmonics[0][[0, 0, 0]] = 0.25;

    assert_eq!(field.get_harmonic(1)[[0, 0, 0]], 1.0);
    assert_eq!(field.get_harmonic(2)[[0, 0, 0]], 0.5);
    assert_eq!(field.get_harmonic(3)[[0, 0, 0]], 0.25);
}

#[test]
fn test_harmonic_spectrum() {
    let mut field = NonlinearElasticWaveField::new(10, 10, 10, 3);

    field.u_fundamental[[5, 5, 5]] = 1.0;
    field.u_second[[5, 5, 5]] = 0.1;
    field.u_harmonics[0][[5, 5, 5]] = 0.01;

    let spectrum = field.harmonic_spectrum(5, 5, 5);
    assert_eq!(spectrum.len(), 3);
    assert_eq!(spectrum[0], 1.0);
    assert_eq!(spectrum[1], 0.1);
    assert_eq!(spectrum[2], 0.01);
}

#[test]
fn test_estimate_nonlinearity() {
    let mut field = NonlinearElasticWaveField::new(10, 10, 10, 3);

    field.u_fundamental[[5, 5, 5]] = 1e-3;
    field.u_second[[5, 5, 5]] = 1e-7;

    let u_ref = 1e-3;
    let beta = field.estimate_nonlinearity(5, 5, 5, u_ref);

    // β ≈ u₂ / (u₁²/u_ref) = 1e-7 / ((1e-3)²/1e-3) = 1e-7 / 1e-3 = 1e-4
    assert!((beta - 1e-4).abs() < 1e-10);
}

#[test]
fn test_reset() {
    let mut field = NonlinearElasticWaveField::new(10, 10, 10, 3);

    field.u_fundamental.fill(1.0);
    field.u_second.fill(0.5);
    field.time = 1.0;

    field.reset();

    for &val in field.u_fundamental.iter() {
        assert_eq!(val, 0.0);
    }
    for &val in field.u_second.iter() {
        assert_eq!(val, 0.0);
    }
    assert_eq!(field.time, 0.0);
}

#[test]
fn test_get_harmonic_mut() {
    let mut field = NonlinearElasticWaveField::new(10, 10, 10, 3);

    {
        let h1 = field.get_harmonic_mut(1);
        h1[[0, 0, 0]] = 2.0;
    }

    assert_eq!(field.u_fundamental[[0, 0, 0]], 2.0);
}

#[test]
#[should_panic(expected = "out of range")]
fn test_get_harmonic_invalid_index() {
    let field = NonlinearElasticWaveField::new(10, 10, 10, 3);
    let _ = field.get_harmonic(10); // Should panic
}

#[test]
fn test_num_harmonics() {
    let field2 = NonlinearElasticWaveField::new(10, 10, 10, 2);
    assert_eq!(field2.num_harmonics(), 2);

    let field5 = NonlinearElasticWaveField::new(10, 10, 10, 5);
    assert_eq!(field5.num_harmonics(), 5);
}
