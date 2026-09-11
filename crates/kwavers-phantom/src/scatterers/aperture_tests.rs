//! Value-semantic tests for the finite-aperture seam (ADR 113).

use super::super::{ArrayTransmit, RfSynthesisConfig, ScattererCloud};
use super::*;
use kwavers_core::constants::acoustic_parameters::NP_TO_DB;
use kwavers_math::numerics::convolution::convolve_into;

/// A circular-piston kernel, reimplemented here as a test double so this
/// crate's tests do not reach into `kwavers-physics` — the dependency the seam
/// exists to avoid. Mirrors `CircularPistonSir::response`: the one-way
/// Tupholme–Stepanishen SIR sampled at the bin midpoints inside its support,
/// returned from its onset; the round trip is the seam's default.
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

impl CircularPiston {
    /// One-way support `[d_min, d_max)/c` as the bins whose midpoints fall in it.
    fn support_bins(&self, r: f64, z: f64, dt: f64) -> core::ops::Range<usize> {
        let (a, c) = (self.radius, self.sound_speed);
        let d_min = if r <= a {
            z
        } else {
            (z * z + (r - a).powi(2)).sqrt()
        };
        let d_max = (z * z + (r + a).powi(2)).sqrt();
        let first = (d_min / c / dt - 0.5).ceil().max(0.0) as usize;
        let end = (d_max / c / dt - 0.5).ceil().max(0.0) as usize;
        first..end
    }
}

impl ApertureKernel for CircularPiston {
    fn response(&self, x_m: f64, y_m: f64, z_m: f64, dt_s: f64) -> SupportSamples {
        let r_m = x_m.hypot(y_m);
        let bins = self.support_bins(r_m, z_m, dt_s);
        if bins.is_empty() {
            // No node inside the support: the impulse of the area a²/(2·d̄), as
            // the physics provider returns it.
            let (a, c) = (self.radius, self.sound_speed);
            let d_min = if r_m <= a {
                z_m
            } else {
                (z_m * z_m + (r_m - a).powi(2)).sqrt()
            };
            let d_max = (z_m * z_m + (r_m + a).powi(2)).sqrt();
            let mean = 0.5 * (d_min + d_max);
            return SupportSamples {
                first_sample: (mean / c / dt_s).floor() as usize,
                samples: vec![a * a / (2.0 * mean) / dt_s],
            };
        }
        SupportSamples {
            first_sample: bins.start,
            samples: bins
                .map(|k| self.evaluate(r_m, z_m, (k as f64 + 0.5) * dt_s))
                .collect(),
        }
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

    let kernel = piston.round_trip(0.0, 0.0, z, dt);
    let area = kernel.area(dt);
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
        .map(|&p| ApertureElement::new(p, [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).expect("element"))
        .collect();

    let reference = cloud
        .synthesize_rf(&positions, &pulse, &cfg)
        .expect("point-element reference");

    let tiny = CircularPiston {
        radius: 1.0e-6,
        sound_speed: cfg.sound_speed,
    };
    let refined = cloud
        .synthesize_rf_with_aperture(&elements, &pulse, &cfg, &tiny)
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
    let elements = [
        ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).expect("element"),
    ];

    let wide = CircularPiston {
        radius: 6.0e-3,
        sound_speed: cfg.sound_speed,
    };
    let refined = cloud
        .synthesize_rf_with_aperture(&elements, &pulse, &cfg, &wide)
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
    let element =
        ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).expect("element");
    assert!(element.field_point([0.0, 0.0, -1.0e-3]).is_none());
    assert!(element.field_point([0.0, 0.0, 0.0]).is_none());

    let [x, y, z] = element.field_point([3.0, -2.0, 4.0]).expect("in front");
    assert!((z - 4.0).abs() <= 1e-12, "axial distance along the normal");
    assert!((x - 3.0).abs() <= 1e-12, "offset along the width axis");
    assert!(
        (y + 2.0).abs() <= 1e-12,
        "offset along the height axis (normal × width)"
    );
}

/// The frame is orthonormalized from the inputs: a skewed width direction is
/// projected into the face plane, and the height axis closes a right-handed
/// triad, so a rectangular kernel sees a consistent orientation whatever the
/// caller's rounding.
#[test]
fn aperture_element_orthonormalizes_its_frame() {
    let element =
        ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [1.0, 0.0, 0.5]).expect("element");
    assert!((element.width_axis[0] - 1.0).abs() <= 1e-15);
    assert!(
        element.width_axis[2].abs() <= 1e-15,
        "normal component removed"
    );
    let h = element.height_axis();
    assert!((h[1] - 1.0).abs() <= 1e-15, "height = normal × width = +y");
}

#[test]
fn aperture_element_rejects_a_degenerate_frame() {
    assert!(ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).is_err());
    assert!(ApertureElement::new([f64::NAN, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).is_err());
    assert!(
        ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 3.0]).is_err(),
        "a width axis parallel to the normal spans no face plane"
    );
}

/// A kernel that depends on the in-plane direction — wider across `y` than
/// `x` — so that the element frame's orientation is observable in the RF.
struct AnisotropicBox {
    width_x: f64,
    width_y: f64,
    sound_speed: f64,
}

impl ApertureKernel for AnisotropicBox {
    fn response(&self, x_m: f64, y_m: f64, z_m: f64, dt_s: f64) -> SupportSamples {
        let l = x_m.hypot(y_m).hypot(z_m);
        let span = (self.width_x * x_m.abs() + self.width_y * y_m.abs()) / (l * self.sound_speed);
        let bins = (span / dt_s).ceil().max(1.0) as usize;
        SupportSamples {
            first_sample: (l / self.sound_speed / dt_s) as usize,
            samples: vec![1.0; bins],
        }
    }
}

/// ## Theorem
/// Rotating the whole scene — elements and scatterers together — leaves the
/// RF unchanged, because the kernel sees the field point in the element's
/// own frame.
///
/// This is what the width axis exists for: an anisotropic kernel evaluated
/// through a mis-oriented frame would change under a rigid rotation. The two
/// scenes differ only by rounding in the frame dot products, so the bound is
/// a few `ε` of the peak.
#[test]
fn rf_is_invariant_under_a_rigid_rotation_of_the_scene() {
    let cfg = config(100.0e6, 1500);
    let pulse = [0.3, 1.0, -0.5];
    let kernel = AnisotropicBox {
        width_x: 0.4e-3,
        width_y: 4.0e-3,
        sound_speed: cfg.sound_speed,
    };
    let scatterers = [[1.5e-3, 2.0e-3, 7.0e-3], [-0.7e-3, -1.0e-3, 9.0e-3]];
    let amplitudes = [1.0, -0.4];

    // Rotation by 40° about the x axis then 25° about z (exact trig products).
    let (ca, sa) = (40.0_f64.to_radians().cos(), 40.0_f64.to_radians().sin());
    let (cb, sb) = (25.0_f64.to_radians().cos(), 25.0_f64.to_radians().sin());
    let rotate = |v: [f64; 3]| -> [f64; 3] {
        let about_x = [v[0], ca * v[1] - sa * v[2], sa * v[1] + ca * v[2]];
        [
            cb * about_x[0] - sb * about_x[1],
            sb * about_x[0] + cb * about_x[1],
            about_x[2],
        ]
    };

    let upright = ScattererCloud::from_points(&scatterers, &amplitudes).expect("cloud");
    let rotated_points: Vec<[f64; 3]> = scatterers.iter().map(|&p| rotate(p)).collect();
    let rotated = ScattererCloud::from_points(&rotated_points, &amplitudes).expect("cloud");

    let elements = [
        ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).expect("element"),
        ApertureElement::new([1.0e-3, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
            .expect("element"),
    ];
    let rotated_elements: Vec<ApertureElement> = elements
        .iter()
        .map(|e| {
            ApertureElement::new(rotate(e.position), rotate(e.normal), rotate(e.width_axis))
                .expect("element")
        })
        .collect();

    let reference = upright
        .synthesize_rf_with_aperture(&elements, &pulse, &cfg, &kernel)
        .expect("upright");
    let turned = rotated
        .synthesize_rf_with_aperture(&rotated_elements, &pulse, &cfg, &kernel)
        .expect("rotated");

    let peak = reference.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(peak > 0.0, "reference must be non-trivial");
    let worst = reference
        .iter()
        .zip(turned.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        worst <= 1.0e-12 * peak,
        "a rigid rotation must leave the RF unchanged: worst {worst:.3e} against peak {peak:.3e}"
    );

    // And the frame is load-bearing: swapping the width axis onto the height
    // direction changes the trace for this anisotropic kernel.
    let swapped = [
        ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]).expect("element"),
        ApertureElement::new([1.0e-3, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])
            .expect("element"),
    ];
    let misoriented = upright
        .synthesize_rf_with_aperture(&swapped, &pulse, &cfg, &kernel)
        .expect("swapped");
    let difference = reference
        .iter()
        .zip(misoriented.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        difference > 1.0e-3 * peak,
        "an anisotropic kernel must see the frame's orientation"
    );
}

/// ## Theorem
/// Convolution distributes over addition: `pulse ⊛ Σ_s a_s·k_s(t − τ_s)` equals
/// `Σ_s a_s·(pulse ⊛ k_s)(t − τ_s)`.
///
/// ## Why this is the oracle
/// Synthesis accumulates every scatterer's scaled, delayed unit-area kernel
/// into one trace per element and convolves the pulse once (the per-pair cost
/// model of Rivera, Demené & Tanter 2026). The reference below is the other
/// association — convolve per pair, then place — written out naively. The two
/// differ only by floating-point reassociation: each output sample is a sum of
/// at most `S·L` products (`S` scatterers, `L` pulse taps), so the bound is
/// `S·L·ε ≈ 2.7e-15` of the largest term at `S·L = 12`, and `1e-12` relative
/// to the trace peak holds that with over two orders of margin.
#[test]
fn trace_accumulation_matches_the_per_pair_association() {
    let cloud = ScattererCloud::from_points(
        &[
            [0.0, 0.0, 6.0e-3],
            [0.8e-3, 0.3e-3, 9.0e-3],
            [-0.5e-3, 0.0, 7.5e-3],
        ],
        &[1.0, -0.6, 0.3],
    )
    .expect("cloud");
    let fs = 100.0e6;
    let cfg = config(fs, 1500);
    let dt = 1.0 / fs;
    let pulse = [0.2, 1.0, -0.7, 0.1];
    let elements = [
        ApertureElement::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).expect("element"),
        ApertureElement::new([1.0e-3, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0])
            .expect("element"),
    ];
    let piston = CircularPiston {
        radius: 2.0e-3,
        sound_speed: cfg.sound_speed,
    };

    let refined = cloud
        .synthesize_rf_with_aperture(&elements, &pulse, &cfg, &piston)
        .expect("aperture-coupled");

    // Naive per-pair association: unit-area kernel ⊛ pulse, placed at the delay.
    let mut reference = vec![vec![0.0_f64; cfg.num_samples]; elements.len()];
    for (element, row) in elements.iter().zip(reference.iter_mut()) {
        for scatterer in cloud.scatterers() {
            let d = element.position;
            let p = scatterer.position;
            let distance = (d[0] - p[0]).hypot(d[1] - p[1]).hypot(d[2] - p[2]);
            let [x, y, z] = element.field_point(p).expect("in front");
            let kernel = piston.round_trip(x, y, z, dt).samples;
            let area: f64 = kernel.iter().sum::<f64>() * dt;
            let shape: Vec<f64> = kernel.iter().map(|k| k / area).collect();
            let mut echo = vec![0.0_f64; pulse.len() + shape.len() - 1];
            for (i, &pi) in pulse.iter().enumerate() {
                for (j, &sj) in shape.iter().enumerate() {
                    echo[i + j] += pi * sj * dt;
                }
            }
            let amplitude = scatterer.amplitude / (distance * distance);
            let delay = (2.0 * distance / cfg.sound_speed * fs).round() as usize;
            for (offset, &e) in echo.iter().enumerate() {
                if delay + offset < cfg.num_samples {
                    row[delay + offset] += amplitude * e;
                }
            }
        }
    }

    let peak = reference
        .iter()
        .flatten()
        .fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(peak > 0.0, "reference must be non-trivial");
    let mut worst = 0.0_f64;
    for (e, row) in reference.iter().enumerate() {
        for (k, &expected) in row.iter().enumerate() {
            worst = worst.max((refined[[e, k]] - expected).abs());
        }
    }
    assert!(
        worst <= 1.0e-12 * peak,
        "trace accumulation must match the per-pair association to reassociation rounding: worst {worst:.3e} against peak {peak:.3e}"
    );
}

fn element_at(x: f64) -> ApertureElement {
    ApertureElement::new([x, 0.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]).expect("element")
}

fn worst_difference(a: &leto::Array2<f64>, b: &leto::Array2<f64>) -> (f64, f64) {
    let peak = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let worst = a
        .iter()
        .zip(b.iter())
        .fold(0.0_f64, |m, (p, q)| m.max((p - q).abs()));
    (worst, peak)
}

/// ## Theorem
/// One element firing with zero delay and receiving is the monostatic round
/// trip: the trace is `a_s·(h ⊛ h)·dt` at twice the one-way onset, then the
/// pulse. The Field II amplitude model applies no spreading law, so the
/// comparison is against the raw auto-convolution, not a unit-area shape.
///
/// The two sides differ only by floating-point association (`ε`-level per
/// term, a handful of terms per sample), so `1e-12` of the peak bounds them.
#[test]
fn single_element_array_transmit_is_the_round_trip_at_twice_the_onset() {
    let cloud = ScattererCloud::from_points(&[[0.3e-3, 0.0, 7.0e-3]], &[0.8]).expect("cloud");
    let cfg = config(100.0e6, 1400);
    let dt = 1.0 / cfg.sampling_frequency;
    let pulse = [1.0, -0.5, 0.2];
    let elements = [element_at(0.0)];
    let piston = CircularPiston {
        radius: 2.0e-3,
        sound_speed: cfg.sound_speed,
    };

    let rf = cloud
        .synthesize_rf_with_array_transmit(
            &elements,
            &ArrayTransmit::plane(1),
            &pulse,
            &cfg,
            &piston,
        )
        .expect("array transmit");

    let [x, y, z] = elements[0]
        .field_point(cloud.scatterers()[0].position)
        .expect("in front");
    let two_way = piston.response(x, y, z, dt).auto_convolve(dt);
    let mut trace = vec![0.0_f64; cfg.num_samples];
    for (offset, &sample) in two_way.samples.iter().enumerate() {
        trace[two_way.first_sample + offset] = 0.8 * sample;
    }
    let mut expected = vec![0.0_f64; cfg.num_samples];
    convolve_into(&mut expected, &trace, &pulse, dt);

    let peak = expected.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    assert!(peak > 0.0, "the echo must land inside the window");
    let worst = expected
        .iter()
        .enumerate()
        .fold(0.0_f64, |m, (k, &e)| m.max((rf[[0, k]] - e).abs()));
    assert!(
        worst <= 1.0e-12 * peak,
        "single-element array transmit must equal the round trip: worst {worst:.3e} against peak {peak:.3e}"
    );
}

/// ## Theorem
/// The transmit response is linear in the apodization, so the RF of a
/// weighted event is the sum of the RFs of its parts.
#[test]
fn array_transmit_is_linear_in_the_apodization() {
    let cloud = ScattererCloud::from_points(
        &[[0.5e-3, 0.2e-3, 6.0e-3], [-0.8e-3, 0.0, 9.0e-3]],
        &[1.0, -0.6],
    )
    .expect("cloud");
    let cfg = config(100.0e6, 1600);
    let pulse = [0.4, 1.0, -0.7];
    let elements = [element_at(-0.5e-3), element_at(0.5e-3)];
    let piston = CircularPiston {
        radius: 0.4e-3,
        sound_speed: cfg.sound_speed,
    };
    let delays = vec![0.0, 20.0e-9];
    let run = |weights: Vec<f64>| {
        cloud
            .synthesize_rf_with_array_transmit(
                &elements,
                &ArrayTransmit::new(delays.clone(), weights).expect("event"),
                &pulse,
                &cfg,
                &piston,
            )
            .expect("array transmit")
    };
    let both = run(vec![1.0, 0.5]);
    let first = run(vec![1.0, 0.0]);
    let second = run(vec![0.0, 0.5]);
    let sum = leto::Array2::from_shape_fn([elements.len(), cfg.num_samples], |index| {
        first[index] + second[index]
    });
    let (worst, peak) = worst_difference(&both, &sum);
    assert!(peak > 0.0);
    assert!(
        worst <= 1.0e-12 * peak,
        "apodization must enter linearly: worst {worst:.3e} against peak {peak:.3e}"
    );
}

/// ## Theorem
/// Focusing delays make every element's arrival at the focus coincide:
/// `l_m/c + τ_m` is one value, the farthest element fires at `t = 0`, and no
/// delay is negative.
#[test]
fn focusing_delays_align_the_arrivals_at_the_focus() {
    let c = 1540.0;
    let elements = [element_at(-1.0e-3), element_at(0.0), element_at(1.0e-3)];
    let focus = [0.5e-3, 0.0, 6.0e-3];
    let event = ArrayTransmit::focused(&elements, focus, c).expect("focused");
    let arrivals: Vec<f64> = elements
        .iter()
        .zip(event.delays_s())
        .map(|(e, &tau)| {
            let d = [
                focus[0] - e.position[0],
                focus[1] - e.position[1],
                focus[2] - e.position[2],
            ];
            d[0].hypot(d[1]).hypot(d[2]) / c + tau
        })
        .collect();
    for &tau in event.delays_s() {
        assert!(tau >= 0.0);
    }
    assert!(
        event.delays_s().contains(&0.0),
        "the farthest element fires at t = 0"
    );
    for &arrival in &arrivals {
        assert!(
            (arrival - arrivals[0]).abs() <= 1.0e-15 * arrivals[0],
            "arrivals must coincide: {arrivals:?}"
        );
    }
    assert_eq!(event.apodization(), &[1.0, 1.0, 1.0]);
}

/// ## Theorem
/// Attenuation enters once per leg, so a single-element echo scales by the
/// round-trip factor `exp(−α·2l)` at the configured centre frequency.
#[test]
fn attenuation_scales_a_single_element_echo_by_the_round_trip_factor() {
    let scatterer = [0.4e-3, 0.0, 8.0e-3];
    let cloud = ScattererCloud::from_points(&[scatterer], &[1.0]).expect("cloud");
    let lossless = config(100.0e6, 1600);
    let lossy = RfSynthesisConfig {
        attenuation_db_cm_mhz: 0.5,
        ..lossless
    };
    let pulse = [1.0, -1.0];
    let elements = [element_at(0.0)];
    let piston = CircularPiston {
        radius: 1.0e-3,
        sound_speed: lossless.sound_speed,
    };
    let event = ArrayTransmit::plane(1);
    let reference = cloud
        .synthesize_rf_with_array_transmit(&elements, &event, &pulse, &lossless, &piston)
        .expect("lossless");
    let attenuated = cloud
        .synthesize_rf_with_array_transmit(&elements, &event, &pulse, &lossy, &piston)
        .expect("lossy");

    let l = scatterer[0].hypot(scatterer[2]);
    let alpha_np_m =
        lossy.attenuation_db_cm_mhz * (lossy.center_frequency_hz / 1.0e6) * 100.0 / NP_TO_DB;
    let factor = (-alpha_np_m * 2.0 * l).exp();
    assert!(
        factor < 0.99,
        "the case must attenuate measurably, got {factor}"
    );
    let scaled =
        leto::Array2::from_shape_fn([1, lossless.num_samples], |index| reference[index] * factor);
    let (worst, peak) = worst_difference(&attenuated, &scaled);
    assert!(peak > 0.0);
    assert!(
        worst <= 1.0e-12 * peak,
        "attenuation must scale the echo by exp(-2αl): worst {worst:.3e} against peak {peak:.3e}"
    );
}

#[test]
fn array_transmit_rejects_mismatched_lengths_and_non_finite_kernels() {
    assert!(ArrayTransmit::new(vec![0.0], vec![1.0, 1.0]).is_err());
    assert!(ArrayTransmit::new(vec![-1.0e-9], vec![1.0]).is_err());
    assert!(ArrayTransmit::new(vec![f64::NAN], vec![1.0]).is_err());
    assert!(ArrayTransmit::new(vec![0.0], vec![f64::INFINITY]).is_err());

    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, 5.0e-3]], &[1.0]).expect("cloud");
    let cfg = config(100.0e6, 1000);
    let elements = [element_at(0.0)];
    let piston = CircularPiston {
        radius: 1.0e-3,
        sound_speed: cfg.sound_speed,
    };
    let err = cloud
        .synthesize_rf_with_array_transmit(
            &elements,
            &ArrayTransmit::plane(2),
            &[1.0],
            &cfg,
            &piston,
        )
        .expect_err("two delays for one element");
    assert!(format!("{err}").contains("one delay and apodization per element"));

    let poisoned = |_x: f64, _y: f64, _z: f64, _dt: f64| SupportSamples {
        first_sample: 10,
        samples: vec![1.0, f64::NAN],
    };
    let err = cloud
        .synthesize_rf_with_array_transmit(
            &elements,
            &ArrayTransmit::plane(1),
            &[1.0],
            &cfg,
            &poisoned,
        )
        .expect_err("a NaN sample is a provider defect");
    assert!(format!("{err}").contains("non-finite sample"));
}

/// ## Theorem
/// Two elements with distinct non-zero delays produce, on each receiver, the
/// cross-convolution of the delayed, weighted transmit superposition with
/// that receiver's own response, at the kernels' summed onsets.
///
/// The expected rows are assembled from the math primitives and the test
/// double alone — `superpose` with `round(τ_m·fs)`, `convolve` with the
/// receive response, `convolve_into` with the pulse — so a zeroed delay, a
/// delay in the wrong unit, or a receive leg reading the wrong element all
/// move a row. Rounding: a few terms per sample, so `1e-12` of the peak.
#[test]
fn delayed_two_element_transmit_matches_the_primitive_assembly_per_receiver() {
    let scatterers = [[0.4e-3, 0.1e-3, 6.0e-3], [-0.6e-3, 0.0, 8.5e-3]];
    let amplitudes = [1.0, -0.7];
    let cloud = ScattererCloud::from_points(&scatterers, &amplitudes).expect("cloud");
    let cfg = config(100.0e6, 1600);
    let fs = cfg.sampling_frequency;
    let dt = 1.0 / fs;
    let pulse = [0.3, 1.0, -0.6];
    let elements = [element_at(-0.5e-3), element_at(0.5e-3)];
    let piston = CircularPiston {
        radius: 0.4e-3,
        sound_speed: cfg.sound_speed,
    };
    let delays = [35.0e-9, 120.0e-9];
    let weights = [1.0, 0.6];
    let event = ArrayTransmit::new(delays.to_vec(), weights.to_vec()).expect("event");

    let rf = cloud
        .synthesize_rf_with_array_transmit(&elements, &event, &pulse, &cfg, &piston)
        .expect("array transmit");

    let mut traces = vec![vec![0.0_f64; cfg.num_samples]; elements.len()];
    for (scatterer, &amplitude) in scatterers.iter().zip(&amplitudes) {
        let responses: Vec<SupportSamples> = elements
            .iter()
            .map(|e| {
                let [x, y, z] = e.field_point(*scatterer).expect("in front");
                piston.response(x, y, z, dt)
            })
            .collect();
        let mut transmit = SupportSamples::zero();
        for ((response, &tau), &weight) in responses.iter().zip(&delays).zip(&weights) {
            transmit.superpose(response, (tau * fs).round() as usize, weight);
        }
        for (trace, response) in traces.iter_mut().zip(&responses) {
            let echo = transmit.convolve(response, dt);
            for (offset, &sample) in echo.samples.iter().enumerate() {
                if let Some(slot) = trace.get_mut(echo.first_sample + offset) {
                    *slot += amplitude * sample;
                }
            }
        }
    }
    let mut peak = 0.0_f64;
    let mut worst = 0.0_f64;
    for (e, trace) in traces.iter().enumerate() {
        let mut expected = vec![0.0_f64; cfg.num_samples];
        convolve_into(&mut expected, trace, &pulse, dt);
        for (k, &value) in expected.iter().enumerate() {
            peak = peak.max(value.abs());
            worst = worst.max((rf[[e, k]] - value).abs());
        }
    }
    assert!(peak > 0.0, "both receivers must see both echoes");
    // The delays are load-bearing: the shifts differ by 8.5 samples, so an
    // assembly with zero delays lands elsewhere.
    assert!((delays[1] - delays[0]) * fs > 8.0);
    assert!(
        worst <= 1.0e-12 * peak,
        "per-receiver rows must match the primitive assembly: worst {worst:.3e} against peak {peak:.3e}"
    );
}

/// An empty response is a provider defect on either path — on the array path
/// its area is the amplitude, so silence would be the wrong answer.
#[test]
fn an_empty_response_is_an_error_on_both_paths() {
    let cloud = ScattererCloud::from_points(&[[0.0, 0.0, 5.0e-3]], &[1.0]).expect("cloud");
    let cfg = config(100.0e6, 1000);
    let elements = [element_at(0.0)];
    let empty = |_x: f64, _y: f64, _z: f64, _dt: f64| SupportSamples::zero();
    let err = cloud
        .synthesize_rf_with_array_transmit(
            &elements,
            &ArrayTransmit::plane(1),
            &[1.0],
            &cfg,
            &empty,
        )
        .expect_err("empty response on the array path");
    assert!(format!("{err}").contains("empty"));
    let err = cloud
        .synthesize_rf_with_aperture(&elements, &[1.0], &cfg, &empty)
        .expect_err("empty response on the monostatic path");
    assert!(format!("{err}").contains("empty"));

    // A non-empty response with no area is the same defect in another form.
    let dead = |_x: f64, _y: f64, _z: f64, _dt: f64| SupportSamples {
        first_sample: 100,
        samples: vec![0.0],
    };
    let err = cloud
        .synthesize_rf_with_array_transmit(&elements, &ArrayTransmit::plane(1), &[1.0], &cfg, &dead)
        .expect_err("zero-area response on the array path");
    assert!(format!("{err}").contains("no area"));
    let err = cloud
        .synthesize_rf_with_aperture(&elements, &[1.0], &cfg, &dead)
        .expect_err("zero-area response on the monostatic path");
    assert!(format!("{err}").contains("no area"));
}
