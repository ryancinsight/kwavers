use super::config::DivergingWaveConfig;
use super::processor::DivergingWave;
use ndarray::Array1;

/// Build a small N-element array with uniform pitch `pitch` centred at x=0.
fn uniform_array(n: usize, pitch: f64) -> DivergingWave {
    let x0 = -(n as f64 - 1.0) / 2.0 * pitch;
    let positions: Vec<f64> = (0..n).map(|i| x0 + i as f64 * pitch).collect();
    DivergingWave::new(DivergingWaveConfig {
        element_positions: positions,
        sound_speed: 1540.0,
        virtual_source_depth: 0.010,
        f_number: 1.5,
        sampling_frequency: 40.0e6,
    })
}

/// PRF_max = c / (2·z_max).
///
/// For c=1540 m/s, z_max=40 mm: PRF_max = 19 250 Hz.
#[test]
fn test_max_prf_formula() {
    let dw = uniform_array(8, 3.0e-4);
    let z_max = 0.040; // 40 mm
    let prf = dw.max_prf(z_max);
    let expected = 1540.0 / (2.0 * 0.040);
    assert!(
        (prf - expected).abs() / expected < 1e-10,
        "PRF_max = {prf:.2} Hz, expected {expected:.2} Hz"
    );
}

/// On-axis (x=0) with center element (x_i=0) the transmit delay reduces to:
///   τ_tx(0, z, center) = [sqrt(0 + (z+F)²) − F] / c = z / c
///
/// This matches a plane-wave normal-incidence transmit delay at z, confirming
/// the virtual-source formula collapses correctly on axis.
#[test]
fn test_on_axis_tx_delay_equals_z_over_c() {
    let dw = uniform_array(9, 3.0e-4); // 9 elements → center at index 4 (x=0)
    let z = 0.020; // 20 mm
    let center = 4;
    // Center element is at x=0 by construction
    assert!(
        dw.config.element_positions[center].abs() < 1e-12,
        "Center element must be at x=0"
    );
    let tau = dw.transmit_delay(0.0, z, center).unwrap();
    let expected = z / dw.config.sound_speed;
    assert!(
        (tau - expected).abs() < 1e-12,
        "On-axis τ_tx = {tau:.6e} s, expected z/c = {expected:.6e} s"
    );
}

/// Receive delay at z=0 from a scatterer directly on the element face equals zero.
#[test]
fn test_receive_delay_at_face_is_zero() {
    let dw = uniform_array(8, 3.0e-4);
    // Element 3 at position x₃
    let x3 = dw.config.element_positions[3];
    let tau = dw.receive_delay(x3, 0.0, 3).unwrap();
    assert!(tau.abs() < 1e-12, "τ_rx at face must be 0, got {tau:.4e}");
}

/// Transmit delays are non-negative for all imaging depths z ≥ 0.
///
/// This follows from ||(x−xᵢ, z+F)|| ≥ F for any (x, z) with z ≥ 0 and F > 0.
#[test]
fn test_transmit_delays_non_negative() {
    let dw = uniform_array(8, 3.0e-4);
    let x_test = [-0.01, 0.0, 0.01];
    let z_test = [0.005, 0.010, 0.030, 0.050];
    for &x in &x_test {
        for &z in &z_test {
            for elem in 0..dw.n_elements() {
                let tau = dw.transmit_delay(x, z, elem).unwrap();
                assert!(
                    tau >= -1e-15,
                    "τ_tx negative: x={x:.3e} z={z:.3e} elem={elem} τ={tau:.4e}"
                );
            }
        }
    }
}

/// Lateral symmetry: for symmetric element array and on-axis x=0,
/// the transmit delay from element i must equal that from the mirror element.
#[test]
fn test_lateral_symmetry() {
    let n = 8;
    let dw = uniform_array(n, 3.0e-4);
    let z = 0.020;
    let x = 0.005; // off-axis
    for i in 0..n / 2 {
        let j = n - 1 - i; // mirror element
                           // Symmetric test: transmit from i with query x, vs from j with query -x
        let tau_i = dw.transmit_delay(x, z, i).unwrap();
        let tau_j = dw.transmit_delay(-x, z, j).unwrap();
        assert!(
            (tau_i - tau_j).abs() < 1e-12,
            "Symmetry broken: i={i} j={j} τ_i={tau_i:.6e} τ_j={tau_j:.6e}"
        );
    }
}

/// STA delay = transmit + receive.
#[test]
fn test_sta_delay_is_sum() {
    let dw = uniform_array(8, 3.0e-4);
    let (x, z, tx, rx) = (0.003, 0.015, 2, 5);
    let tau_tx = dw.transmit_delay(x, z, tx).unwrap();
    let tau_rx = dw.receive_delay(x, z, rx).unwrap();
    let tau_sta = dw.sta_delay(x, z, tx, rx).unwrap();
    assert!(
        (tau_sta - (tau_tx + tau_rx)).abs() < 1e-15,
        "STA delay {tau_sta:.6e} ≠ τ_tx + τ_rx = {:.6e}",
        tau_tx + tau_rx
    );
}

/// Hann apodization weight at distance 0 from centre of aperture must equal 1.0.
#[test]
fn test_hann_apodization_center_is_one() {
    let dw = uniform_array(8, 3.0e-4);
    let z = 0.020;
    // Find element closest to x=0
    let center_idx = dw
        .config
        .element_positions
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let xc = dw.config.element_positions[center_idx];
    let w = dw.hann_apodization(xc, z, center_idx);
    assert!(
        (w - 1.0).abs() < 1e-12,
        "Hann weight at dist=0 must be 1.0, got {w:.6}"
    );
}

/// Elements outside the F-number cone receive zero weight.
#[test]
fn test_hann_apodization_out_of_aperture_is_zero() {
    let dw = uniform_array(8, 3.0e-4);
    let z = 0.001; // Very shallow → narrow aperture cone
                   // All elements are outside the D_half = (z+F)/(2*f_num) = (0.001+0.010)/3.0 = 3.67 mm
                   // Element 0 is at x ≈ −1.05 mm, which is within the cone
                   // Use an extreme lateral point that must be outside
    let x_far = 0.020; // 20 mm lateral, far from any element
    for elem in 0..dw.n_elements() {
        let xj = dw.config.element_positions[elem];
        let f = dw.config.virtual_source_depth;
        let d_half = (z + f) / (2.0 * dw.config.f_number);
        if (xj - x_far).abs() > d_half {
            let w = dw.hann_apodization(x_far, z, elem);
            assert!(
                w == 0.0,
                "Element {elem} outside aperture must have w=0, got {w}"
            );
        }
    }
}

/// transmit_delay_surface shape must be [n_elements × (nx·nz)].
#[test]
fn test_transmit_delay_surface_shape() {
    let dw = uniform_array(8, 3.0e-4);
    let x_px = Array1::linspace(-0.005, 0.005, 16);
    let z_px = Array1::linspace(0.005, 0.030, 32);
    let surf = dw.transmit_delay_surface(&x_px, &z_px).unwrap();
    assert_eq!(
        surf.dim(),
        (8, 16 * 32),
        "transmit_delay_surface shape mismatch"
    );
}

/// sta_delay_table shape must be [n_tx × n_rx × (nx·nz)].
#[test]
fn test_sta_delay_table_shape() {
    let dw = uniform_array(4, 3.0e-4);
    let x_px = Array1::linspace(-0.003, 0.003, 8);
    let z_px = Array1::linspace(0.005, 0.020, 16);
    let table = dw.sta_delay_table(&x_px, &z_px).unwrap();
    assert_eq!(
        table.dim(),
        (4, 4, 8 * 16),
        "sta_delay_table shape mismatch"
    );
}

/// Transmit delay is consistent between scalar method and delay table.
#[test]
fn test_transmit_delay_surface_matches_scalar() {
    let dw = uniform_array(4, 3.0e-4);
    let x_px = Array1::from_vec(vec![0.0, 0.002]);
    let z_px = Array1::from_vec(vec![0.010, 0.020]);
    let surf = dw.transmit_delay_surface(&x_px, &z_px).unwrap();

    // pixel at (ix=0, iz=1) → index = 1*2+0 = 2
    let tau_table = surf[[2, 2]]; // elem=2, pixel iz=1,ix=0
    let tau_scalar = dw.transmit_delay(x_px[0], z_px[1], 2).unwrap();
    assert!(
        (tau_table - tau_scalar).abs() < 1e-15,
        "Surface/scalar mismatch: table={tau_table:.6e} scalar={tau_scalar:.6e}"
    );
}

/// At normal incidence (all center, z=20mm), STA delay must be 2z/c.
#[test]
fn test_monostatic_sta_delay_equals_round_trip() {
    let dw = uniform_array(9, 3.0e-4);
    let center = 4; // x=0
    let z = 0.020;
    // Monostatic: same element transmits and receives
    // τ_tx(0, z, 0) = z/c  (on-axis), τ_rx(0, z, 0) = z/c → total = 2z/c
    let tau = dw.sta_delay(0.0, z, center, center).unwrap();
    let expected = 2.0 * z / dw.config.sound_speed;
    assert!(
        (tau - expected).abs() < 1e-12,
        "Monostatic STA delay {tau:.6e} ≠ 2z/c = {expected:.6e}"
    );
}

/// Out-of-range element index returns an error.
#[test]
fn test_out_of_range_index_errors() {
    let dw = uniform_array(4, 3.0e-4);
    assert!(dw.transmit_delay(0.0, 0.01, 10).is_err());
    assert!(dw.receive_delay(0.0, 0.01, 10).is_err());
    assert!(dw.sta_delay(0.0, 0.01, 0, 10).is_err());
}
