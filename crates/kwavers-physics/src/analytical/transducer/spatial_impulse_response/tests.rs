//! Value-semantic tests for the circular-piston spatial impulse response.
//! Expected bounds and values are derived from the Stepanishen (1971) closed form.

use super::CircularPistonSir;

const A: f64 = 5e-3; // 5 mm radius
const C: f64 = 1500.0; // m/s
const Z: f64 = 20e-3; // 20 mm axial

fn sir() -> CircularPistonSir {
    CircularPistonSir::new(A, C).unwrap()
}

#[test]
fn on_axis_is_a_rectangular_pulse_of_height_c() {
    let s = sir();
    let t1 = Z / C; // first arrival
    let t2 = (Z * Z + A * A).sqrt() / C; // last arrival
    assert!((s.first_arrival_time(0.0, Z) - t1).abs() < 1e-15);
    assert!((s.last_arrival_time(0.0, Z) - t2).abs() < 1e-15);
    // Plateau height c strictly inside the support.
    let tm = 0.5 * (t1 + t2);
    assert!((s.evaluate(0.0, Z, tm) - C).abs() < 1e-9, "on-axis plateau must be c");
    // Zero outside the support.
    assert_eq!(s.evaluate(0.0, Z, t1 * 0.5), 0.0);
    assert_eq!(s.evaluate(0.0, Z, t2 * 1.5), 0.0);
}

#[test]
fn on_axis_time_integral_equals_path_difference() {
    // ∫ h(0,z,t) dt = c·(t2 − t1) = √(z²+a²) − z (Stepanishen on-axis result).
    let s = sir();
    let t1 = Z / C;
    let t2 = (Z * Z + A * A).sqrt() / C;
    let n = 200_000;
    let dt = (t2 - t1) / n as f64;
    let mut integral = 0.0;
    for k in 0..n {
        let t = t1 + (k as f64 + 0.5) * dt;
        integral += s.evaluate(0.0, Z, t) * dt;
    }
    let expected = (Z * Z + A * A).sqrt() - Z;
    assert!(
        (integral - expected).abs() / expected < 1e-4,
        "∫h dt = {integral}, expected path difference {expected}"
    );
}

#[test]
fn off_axis_inside_has_plateau_then_decaying_arc() {
    // r < a: plateau h=c over [z, √(z²+(a−r)²)], then a decaying arc to 0.
    // (Evaluated strictly inside each region — the arccos boundary at ct=d is the
    // singular point of the closed form, FP-unstable to sample exactly.)
    let s = sir();
    let r = 0.5 * A;
    let d_plateau = (Z * Z + (A - r).powi(2)).sqrt();
    let d_max = (Z * Z + (A + r).powi(2)).sqrt();
    // Mid-plateau: h = c.
    let t_plateau = 0.5 * (Z + d_plateau) / C;
    assert!((s.evaluate(r, Z, t_plateau) - C).abs() < 1e-9, "inside plateau must be c");
    // Just inside the arc (5% in): 0 < h < c, and below the plateau value.
    let t_arc = (d_plateau + 0.5 * (d_max - d_plateau)) / C;
    let h_arc = s.evaluate(r, Z, t_arc);
    assert!(h_arc > 0.0 && h_arc < C, "arc value must be in (0, c): {h_arc}");
    // Well beyond the last arrival: zero.
    assert_eq!(s.evaluate(r, Z, d_max / C * 1.01), 0.0);
}

#[test]
fn off_axis_outside_is_arc_only() {
    // r > a: no plateau; support [√(z²+(r−a)²), √(z²+(r+a)²)], arc in (0, c).
    let s = sir();
    let r = 2.0 * A;
    let d_min = (Z * Z + (r - A).powi(2)).sqrt();
    let d_max = (Z * Z + (r + A).powi(2)).sqrt();
    assert!((s.first_arrival_time(r, Z) - d_min / C).abs() < 1e-15);
    assert!((s.last_arrival_time(r, Z) - d_max / C).abs() < 1e-15);
    // No full-circle plateau anywhere: even mid-support h < c.
    let t_mid = 0.5 * (d_min + d_max) / C;
    let h = s.evaluate(r, Z, t_mid);
    assert!(h > 0.0 && h < C, "arc value must be in (0,c): {h}");
    // Zero strictly outside the support.
    assert_eq!(s.evaluate(r, Z, d_min / C * 0.5), 0.0);
    assert_eq!(s.evaluate(r, Z, d_max / C * 1.01), 0.0);
}

#[test]
fn response_is_bounded_in_zero_to_c_everywhere() {
    let s = sir();
    for &r in &[0.0, 0.3 * A, A, 1.5 * A, 3.0 * A] {
        let t_end = s.last_arrival_time(r, Z) * 1.2;
        let n = 1000;
        for k in 0..n {
            let t = t_end * k as f64 / n as f64;
            let h = s.evaluate(r, Z, t);
            assert!((0.0..=C + 1e-9).contains(&h), "h={h} out of [0,c] at r={r}, t={t}");
        }
    }
}

#[test]
fn rejects_invalid_parameters() {
    assert!(CircularPistonSir::new(0.0, C).is_err());
    assert!(CircularPistonSir::new(-1.0, C).is_err());
    assert!(CircularPistonSir::new(A, 0.0).is_err());
    assert!(CircularPistonSir::new(A, f64::NAN).is_err());
}
