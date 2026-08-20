//! Value-semantic tests for the finite-aperture seam (ADR 113).

use super::super::{RfSynthesisConfig, ScattererCloud};
use super::*;

/// A circular-piston round-trip kernel, reimplemented here as a test double so
/// this crate's tests do not reach into `kwavers-physics` — the dependency the
/// seam exists to avoid. Mirrors `CircularPistonSir::round_trip_response`:
/// the one-way Tupholme–Stepanishen SIR sampled at bin midpoints and discretely
/// auto-convolved.
struct CircularPiston {
    radius: f64,
    sound_speed: f64,
}

impl CircularPiston {
    /// One-way SIR `h(r, z, t)` of a flat circular piston in a rigid baffle.
    fn evaluate(&self, r: f64, z: f64, t: f64) -> f64 {
        let (a, c) = (self.radius, self.sound_speed);
        let ct = c * t;
        if ct <= z {
            return 0.0;
        }
        let rho_sq = ct.mul_add(ct, -(z * z));
        if rho_sq <= 0.0 {
            return 0.0;
        }
        let rho = rho_sq.sqrt();
        if r < a && rho <= a - r {
            return c;
        }
        if rho >= r + a || rho <= (r - a).abs() {
            return 0.0;
        }
        let cos = (rho.mul_add(rho, r.mul_add(r, -(a * a)))) / (2.0 * r * rho);
        c * cos.clamp(-1.0, 1.0).acos() / core::f64::consts::PI
    }
}

impl RoundTripKernel for CircularPiston {
    fn round_trip(&self, r_m: f64, z_m: f64, dt_s: f64, n_samples: usize) -> Vec<f64> {
        let h: Vec<f64> = (0..n_samples)
            .map(|k| self.evaluate(r_m, z_m, (k as f64 + 0.5) * dt_s))
            .collect();
        let mut out = vec![0.0_f64; n_samples];
        for (i, &hi) in h.iter().enumerate() {
            if hi == 0.0 {
                continue;
            }
            for (j, &hj) in h.iter().enumerate().take(n_samples - i) {
                out[i + j] += hi * hj * dt_s;
            }
        }
        out
    }
}

fn config(fs: f64, num_samples: usize) -> RfSynthesisConfig {
    RfSynthesisConfig {
        sound_speed: 1540.0,
        sampling_frequency: fs,
        num_samples,
        min_distance: 1.0e-4,
        attenuation_db_cm_mhz: 0.0,
        center_frequency_hz: 3.0e6,
    }
}

/// ## Theorem
/// The on-axis two-way kernel integrates to `(√(z²+a²) − z)²`.
///
/// ## Why this is the oracle
/// The convolution integral factorizes, `∫(h⊛h)dt = (∫h dt)²`, and on axis
/// `∫h dt = √(z²+a²) − z`. So the kernel a provider supplies can be checked
/// against a closed form rather than a recorded trace — and if the test double
/// here drifts from that identity, every result built on it is suspect.
#[test]
fn round_trip_kernel_area_matches_the_closed_form() {
    let (a, c, z) = (5.0e-3, 1540.0, 30.0e-3);
    let piston = CircularPiston {
        radius: a,
        sound_speed: c,
    };
    let dt = 1.0 / 200.0e6;
    // Support ends at 2*d_max/c with d_max = sqrt(z^2 + a^2).
    let n = (2.0 * (z * z + a * a).sqrt() / c / dt).ceil() as usize + 8;

    let kernel = piston.round_trip(0.0, z, dt, n);
    let area: f64 = kernel.iter().sum::<f64>() * dt;
    let expected = ((z * z + a * a).sqrt() - z).powi(2);

    assert!(
        (area - expected).abs() <= 2.0e-2 * expected,
        "kernel area {area:.6e} against closed form {expected:.6e}"
    );
}

/// ## Theorem
/// As the aperture radius tends to zero, aperture-coupled RF converges on the
/// point-element `synthesize_rf` output.
///
/// ## Why this is the oracle (ADR 113)
/// A refinement must reduce to what it refines. The kernel enters unit-area, so
/// a shrinking aperture drives it to a delta and the convolution to the
/// identity. This is also what rules out convolving the *raw* kernel: its area
/// `(√(z²+a²) − z)²` tends to zero with the radius, so raw coupling converges on
/// silence — the assertion below would fail against an all-zero trace.
#[test]
fn vanishing_aperture_converges_on_the_point_element_model() {
    let cloud =
        ScattererCloud::from_points(&[[0.0, 0.0, 5.0e-3], [1.0e-3, 0.0, 6.0e-3]], &[1.0, -0.5])
            .expect("cloud");

    let fs = 100.0e6;
    // Deepest echo: 2*6 mm / 1540 m/s at 100 MHz lands near sample 780, so the
    // window must outrun it or the trace is empty and proves nothing.
    let cfg = config(fs, 1200);
    let pulse = [0.0, 1.0, -0.7, 0.2];

    let positions = [[0.0, 0.0, 0.0], [1.0e-3, 0.0, 0.0]];
    let elements: Vec<ApertureElement> = positions
        .iter()
        .map(|&p| ApertureElement::new(p, [0.0, 0.0, 1.0]).expect("element"))
        .collect();

    let reference = cloud
        .synthesize_rf(&positions, &pulse, &cfg)
        .expect("point-element reference");

    let tiny = CircularPiston {
        radius: 1.0e-6,
        sound_speed: cfg.sound_speed,
    };
    let refined = cloud
        // Kernel sampled from t = 0, so it must reach past 2*d_max/c.
        .synthesize_rf_with_aperture(&elements, &pulse, &cfg, &tiny, 900)
        .expect("aperture-coupled");

    let peak = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(peak > 0.0, "reference must be non-trivial");
    let worst = reference
        .iter()
        .zip(refined.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));

    assert!(
        worst <= 1.0e-6 * peak,
        "a 1 um aperture must reproduce the point-element model: worst {worst:.3e} against peak {peak:.3e}"
    );
}

/// A finite aperture must actually change the trace — otherwise the seam is
/// wired up but inert, and the convergence test above would pass on a
/// no-op. Pairs with it: one bounds the limit, this one proves the mechanism
/// is live away from that limit.
#[test]
fn a_finite_aperture_smears_the_echo_in_time() {
    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, 8.0e-3]], &[1.0]).expect("cloud");

    let fs = 100.0e6;
    // Echo at 2*8 mm / 1540 m/s lands near sample 1039; leave room for the
    // kernel tail so the smeared echo is not clipped by the window edge.
    let cfg = config(fs, 1600);
    let pulse = [1.0];
    let elements = [ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("element")];

    let wide = CircularPiston {
        radius: 6.0e-3,
        sound_speed: cfg.sound_speed,
    };
    let refined = cloud
        .synthesize_rf_with_aperture(&elements, &pulse, &cfg, &wide, 1300)
        .expect("aperture-coupled");

    let occupied = refined.iter().filter(|v| v.abs() > 1.0e-12).count();
    assert!(
        occupied > 1,
        "a 6 mm aperture must spread a single-sample pulse over several samples, got {occupied}"
    );

    // Unit-area normalization is what keeps this a refinement: the smeared echo
    // carries the same integrated amplitude as the point-element impulse.
    let reference = cloud
        .synthesize_rf(&[[0.0, 0.0, 0.0]], &pulse, &cfg)
        .expect("reference");
    let refined_sum: f64 = refined.iter().sum();
    let reference_sum: f64 = reference.iter().sum();
    assert!(
        (refined_sum - reference_sum).abs() <= 1.0e-3 * reference_sum.abs(),
        "smearing must conserve the echo's integrated amplitude: {refined_sum:.6e} against {reference_sum:.6e}"
    );
}

#[test]
fn field_point_rejects_targets_at_or_behind_the_face() {
    let element = ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("element");
    assert!(element.field_point([0.0, 0.0, -1.0e-3]).is_none());
    assert!(element.field_point([0.0, 0.0, 0.0]).is_none());

    let (r, z) = element.field_point([3.0, 0.0, 4.0]).expect("in front");
    assert!((z - 4.0).abs() <= 1e-12, "axial distance along the normal");
    assert!((r - 3.0).abs() <= 1e-12, "lateral offset from the axis");
}

#[test]
fn aperture_element_rejects_a_degenerate_normal() {
    assert!(ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]).is_err());
    assert!(ApertureElement::new([f64::NAN, 0.0, 0.0], [0.0, 0.0, 1.0]).is_err());
}

/// A kernel window too short to reach the round trip is a caller sizing error,
/// not a silently empty trace.
///
/// The sub-resolution fallback and an undersized window both produce an
/// all-zero kernel, so without this discriminator the two are indistinguishable
/// and an undersized call would quietly return the point-element answer while
/// claiming to model the aperture.
#[test]
fn an_undersized_kernel_window_is_an_error() {
    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, 8.0e-3]], &[1.0]).expect("cloud");
    let cfg = config(100.0e6, 1600);
    let elements = [ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("element")];
    let piston = CircularPiston {
        radius: 3.0e-3,
        sound_speed: cfg.sound_speed,
    };

    // Round trip to 8 mm is ~1039 samples at 100 MHz; 100 cannot reach it.
    let err = cloud
        .synthesize_rf_with_aperture(&elements, &[1.0], &cfg, &piston, 100)
        .expect_err("an unreachable kernel window must be reported");
    let message = format!("{err}");
    assert!(
        message.contains("kernel_samples"),
        "the error must name the parameter to fix, got: {message}"
    );
}
