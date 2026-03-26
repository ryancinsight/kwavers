use super::*;

#[test]
fn test_shock_formation_distance() {
    let params = NonlinearParameters::water();
    let p0 = 1e6; // 1 MPa
    let f = 1e6; // 1 MHz

    let l_s = shock::shock_formation_distance(p0, f, &params);

    // Theory check: l_s = rho0 * c0^3 / (beta * omega * P0)
    // Water: beta ~ 3.5, rho0 ~ 1000, c0 ~ 1500
    // l_s ~ (1000 * 3.375e9) / (3.5 * 2pi*1e6 * 1e6)
    // ~ 3.375e12 / (2.2e13) ~ 0.15 m
    assert!(l_s > 0.05 && l_s < 0.3, "Shock distance out of expected theoretical bounds: {}", l_s);
}

#[test]
fn test_second_harmonic_generation() {
    let params = NonlinearParameters::soft_tissue();
    let p0 = 2e6;
    let f = 2e6;
    let z = 0.05; // 5 cm

    let p2 = harmonics::second_harmonic_amplitude(p0, f, z, &params);
    
    // P2 should be significantly less than P0 initially
    assert!(p2 < p0);
    assert!(p2 > p0 * 0.01, "Second harmonic should be measurable at 5cm");
}

#[test]
fn test_acoustic_saturation() {
    let params = NonlinearParameters::water();
    let f = 1e6;
    let z = 0.1;

    let p_sat = saturation::acoustic_saturation_pressure(f, z, &params);
    
    // Should be a finite positive value
    assert!(p_sat > 0.0);
    assert!(p_sat.is_finite());
}

#[test]
fn test_burgers_equation() {
    let params = NonlinearParameters::soft_tissue();
    let p0 = 1e6;
    let f = 1e6;
    let z = 0.01; // Pre-shock distance

    let p_z = burgers::burgers_equation(p0, f, z, &params);
    assert!(p_z > 0.0 && p_z <= p0);
}

#[test]
fn test_parametric_array() {
    let params = NonlinearParameters::water();
    
    // Two high-frequency primaries
    let p1 = 1e6;
    let p2 = 1e6;
    let f1 = 2.0e6;
    let f2 = 2.1e6;
    let z = 1.0; // 1 meter far-field

    let p_diff = parametric::difference_frequency_amplitude(p1, p2, f1, f2, z, &params);
    
    // Difference frequency is 100 kHz.
    // Amplitude should be physical and non-zero.
    assert!(p_diff > 0.0);
    assert!(p_diff < p1 * 0.1, "Difference frequency amplitude is bounded by Demodulation/Westervelt limits");
}
