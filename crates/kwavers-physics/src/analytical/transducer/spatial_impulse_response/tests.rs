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
    assert!(
        (s.evaluate(0.0, Z, tm) - C).abs() < 1e-9,
        "on-axis plateau must be c"
    );
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
    assert!(
        (s.evaluate(r, Z, t_plateau) - C).abs() < 1e-9,
        "inside plateau must be c"
    );
    // Just inside the arc (5% in): 0 < h < c, and below the plateau value.
    let t_arc = (d_plateau + 0.5 * (d_max - d_plateau)) / C;
    let h_arc = s.evaluate(r, Z, t_arc);
    assert!(
        h_arc > 0.0 && h_arc < C,
        "arc value must be in (0, c): {h_arc}"
    );
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
            assert!(
                (0.0..=C + 1e-9).contains(&h),
                "h={h} out of [0,c] at r={r}, t={t}"
            );
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

// --- Rectangular piston SIR (Lockwood & Willette 1973) ---

use super::RectangularPistonSir;
use std::f64::consts::PI;

const WX: f64 = 4e-3; // 4 mm half-width (x)
const WY: f64 = 3e-3; // 3 mm half-width (y)

fn rect() -> RectangularPistonSir {
    RectangularPistonSir::new(WX, WY, C).unwrap()
}

/// Independent oracle for the in-rectangle arc measure Φ: brute-force θ sampling
/// (vs the implementation's exact breakpoint-interval enumeration).
fn phi_sampled(x: f64, y: f64, rho: f64) -> f64 {
    let n = 400_000usize;
    let mut inside = 0usize;
    for k in 0..n {
        let theta = 2.0 * PI * (k as f64 + 0.5) / n as f64;
        let px = x + rho * theta.cos();
        let py = y + rho * theta.sin();
        if (-WX..=WX).contains(&px) && (-WY..=WY).contains(&py) {
            inside += 1;
        }
    }
    2.0 * PI * inside as f64 / n as f64
}

#[test]
fn rect_on_axis_is_plateau_of_height_c_until_nearest_edge() {
    let s = rect();
    let t1 = Z / C; // first arrival (foot of perpendicular, projection inside)
    let t_edge = (Z * Z + WY * WY).sqrt() / C; // circle first hits the nearest (y) edge
    let t_corner = (Z * Z + WX * WX + WY * WY).sqrt() / C; // far corner = last arrival
    assert!((s.first_arrival_time(0.0, 0.0, Z) - t1).abs() < 1e-15);
    assert!((s.last_arrival_time(0.0, 0.0, Z) - t_corner).abs() < 1e-15);
    // Mid-plateau (circle fully inside) ⇒ Φ = 2π ⇒ h = c.
    let tm = 0.5 * (t1 + t_edge);
    assert!(
        (s.evaluate(0.0, 0.0, Z, tm) - C).abs() < 1e-9,
        "on-axis plateau must be c"
    );
    // Zero outside the support.
    assert_eq!(s.evaluate(0.0, 0.0, Z, t1 * 0.5), 0.0);
    assert_eq!(s.evaluate(0.0, 0.0, Z, t_corner * 1.01), 0.0);
}

#[test]
fn rect_phi_matches_sampling_oracle_across_geometry() {
    // KEYSTONE: the exact breakpoint-interval Φ equals the independent θ-sampled Φ
    // for projection inside, on an edge, at a corner, and fully outside.
    let s = rect();
    let cases = [
        (0.0, 0.0),    // center (inside)
        (2e-3, 1e-3),  // inside, off-center
        (WX, 0.0),     // on the x-edge
        (WX, WY),      // at a corner
        (6e-3, 0.0),   // outside in x
        (6e-3, 5e-3),  // outside diagonally
        (-3e-3, 2e-3), // inside, negative quadrant
    ];
    for &(x, y) in &cases {
        for &rho in &[0.5e-3, 2e-3, 3.5e-3, 5e-3, 8e-3] {
            // h = (c/2π)·Φ ⇒ Φ_impl = 2π·h/c, with ct = √(ρ²+z²).
            let ct = (rho * rho + Z * Z).sqrt();
            let t = ct / C;
            let phi_impl = 2.0 * PI * s.evaluate(x, y, Z, t) / C;
            let phi_ref = phi_sampled(x, y, rho);
            assert!(
                (phi_impl - phi_ref).abs() < 2e-3,
                "x={x}, y={y}, rho={rho}: Φ_impl={phi_impl} vs sampled {phi_ref}"
            );
        }
    }
}

#[test]
fn rect_is_symmetric_under_reflection() {
    // The centered rectangle is symmetric in x and y ⇒ h(±x, ±y) all equal.
    let s = rect();
    let (x, y) = (2e-3, 1.5e-3);
    let rho = 3e-3;
    let t = (rho * rho + Z * Z).sqrt() / C;
    let h = s.evaluate(x, y, Z, t);
    for &(sx, sy) in &[(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)] {
        let hr = s.evaluate(sx * x, sy * y, Z, t);
        assert!(
            (h - hr).abs() < 1e-9,
            "reflection symmetry broken: {h} vs {hr}"
        );
    }
}

#[test]
fn rect_response_is_bounded_in_zero_to_c() {
    let s = rect();
    for &(x, y) in &[(0.0, 0.0), (2e-3, 1e-3), (WX, WY), (7e-3, 0.0)] {
        let t_end = s.last_arrival_time(x, y, Z) * 1.2;
        let n = 1000;
        for k in 0..n {
            let t = t_end * k as f64 / n as f64;
            let h = s.evaluate(x, y, Z, t);
            assert!(
                (0.0..=C + 1e-9).contains(&h),
                "h={h} out of [0,c] at ({x},{y}), t={t}"
            );
        }
    }
}

#[test]
fn rect_rejects_invalid_parameters() {
    assert!(RectangularPistonSir::new(0.0, WY, C).is_err());
    assert!(RectangularPistonSir::new(WX, -1.0, C).is_err());
    assert!(RectangularPistonSir::new(WX, WY, 0.0).is_err());
    assert!(RectangularPistonSir::new(WX, WY, f64::NAN).is_err());
}

// --- Two-way (pulse-echo) diffraction kernel h⊛h (COV-4 finite-aperture) ---

#[test]
fn round_trip_kernel_integral_equals_oneway_squared() {
    // KEYSTONE: the convolution integral factorizes, ∫(h⊛h)dt = (∫h dt)², and
    // on-axis ∫h dt = √(z²+a²)−z, so Σ_k out[k]·dt = (√(z²+a²)−z)² exactly.
    let s = sir();
    let dt = 2e-9;
    let last = s.last_arrival_time(0.0, Z);
    let n = (2.2 * last / dt).ceil() as usize; // cover the two-way support 2·d_max
    let kernel = s.round_trip_response(0.0, Z, dt, n);
    let integral: f64 = kernel.iter().sum::<f64>() * dt;

    // Exact (to machine precision) factorization against the SAME discretization
    // of the one-way SIR — this is the convolution identity ∫(h⊛h)dt = (∫h dt)².
    let oneway_discrete: f64 = (0..n)
        .map(|k| s.evaluate(0.0, Z, (k as f64 + 0.5) * dt))
        .sum::<f64>()
        * dt;
    assert!(
        (integral - oneway_discrete * oneway_discrete).abs() / (oneway_discrete * oneway_discrete)
            < 1e-9,
        "∫(h⊛h)dt = {integral} must equal (∫h dt)² = {}",
        oneway_discrete * oneway_discrete
    );
    // And the one-way integral matches the Stepanishen closed form √(z²+a²)−z to
    // within the O(dt) rect-edge discretization (~1%).
    let oneway_closed = (Z * Z + A * A).sqrt() - Z;
    assert!(
        (oneway_discrete - oneway_closed).abs() / oneway_closed < 1e-2,
        "discrete ∫h dt = {oneway_discrete} vs closed form {oneway_closed}"
    );
}

#[test]
fn round_trip_kernel_is_a_triangle_over_the_two_way_support() {
    // On-axis the one-way SIR is a rect over [z, √(z²+a²)]/c, so its
    // auto-convolution is a triangle over [2z, 2√(z²+a²)]/c peaking at
    // (z+√(z²+a²))/c. Verify onset and peak location.
    let s = sir();
    let dt = 2e-9;
    let t1 = Z / C; // one-way first arrival
    let t2 = (Z * Z + A * A).sqrt() / C; // one-way last arrival
    let n = (2.2 * t2 / dt).ceil() as usize;
    let kernel = s.round_trip_response(0.0, Z, dt, n);

    // No echo before the round-trip onset 2·t1 (allow a couple bins of slack).
    let n_onset = (2.0 * t1 / dt) as usize;
    assert!(
        kernel[..n_onset.saturating_sub(3)]
            .iter()
            .all(|&v| v == 0.0),
        "two-way kernel must be zero before 2·t1"
    );
    // Peak near (t1 + t2).
    let k_peak = kernel
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let t_peak = (k_peak as f64 + 0.5) * dt;
    assert!(
        (t_peak - (t1 + t2)).abs() < 5.0 * dt,
        "triangle peak at {t_peak}, expected {}",
        t1 + t2
    );
}
