use super::{fixture, traced, AMBIENT, FREQUENCY, R0, SIGMA};
use aequitas::systems::si::quantities::{Pressure, Time};
use kwavers_core::constants::GAS_CONSTANT;
use std::f64::consts::{PI, TAU};
use tracing::{info, info_span};

const FIXTURE: &str = include_str!("../fixtures/diffusive_bubble/sensitivity.csv");
const HORIZON: u32 = 96;
const BALANCE_TOLERANCE: f64 = 1e-8;

#[derive(Clone, Copy)]
struct Resolution {
    steps: u32,
    cells: usize,
    extent: f64,
}

const BASE: Resolution = Resolution {
    steps: 1024,
    cells: 768,
    extent: 120e-6,
};
const REFINEMENTS: [(&str, Resolution); 3] = [
    (
        "time step",
        Resolution {
            steps: 2 * BASE.steps,
            ..BASE
        },
    ),
    (
        "radial grid",
        Resolution {
            cells: 2 * BASE.cells,
            ..BASE
        },
    ),
    (
        "reservoir",
        Resolution {
            cells: 2 * BASE.cells,
            extent: 4.0 * BASE.extent,
            ..BASE
        },
    ),
];

#[derive(Clone, Copy)]
struct Sample {
    end: u32,
    phase: Phase,
    amount: f64,
    total: f64,
    radius: f64,
    velocity: f64,
    equilibrium: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    On,
    Off,
}

fn column<T: std::str::FromStr>(row: &mut std::str::Split<'_, char>) -> T {
    row.next().unwrap().parse().ok().unwrap()
}

fn reference(name: &str) -> (u32, u32, f64, Vec<Sample>) {
    let mut config = None;
    let mut samples = Vec::new();
    for line in FIXTURE.lines().skip(1) {
        let mut row = line.split(',');
        if row.next() != Some(name) {
            continue;
        }
        let input = (column(&mut row), column(&mut row), column(&mut row));
        assert_eq!(*config.get_or_insert(input), input);
        let _: u32 = column(&mut row);
        let end = column(&mut row);
        let phase = match row.next() {
            Some("on") => Phase::On,
            Some("off") => Phase::Off,
            unexpected => panic!("invalid reference phase: {unexpected:?}"),
        };
        samples.push(Sample {
            end,
            phase,
            amount: column(&mut row),
            total: column(&mut row),
            radius: column(&mut row),
            velocity: column(&mut row),
            equilibrium: column(&mut row),
        });
    }
    let (on, off, concentration) = config.expect("reference exists");
    assert_eq!(samples.last().unwrap().end, HORIZON);
    (on, off, concentration, samples)
}

fn simulate(resolution: Resolution, on: u32, off: u32, saturation: f64) -> Vec<Sample> {
    let mut bubble = fixture(resolution.cells, resolution.extent, saturation);
    let initial = bubble.state().gas_amount.into_base();
    let dt = Time::from_base((FREQUENCY * f64::from(resolution.steps)).recip());
    let mut peak_speed = 0.0_f64;
    let mut minimum_radius = R0;
    let mut maximum_radius = R0;
    let mut gross = 0.0;
    let mut reservoir = 0.0;
    let mut previous_total = 0.0;
    let mut max_balance = 0.0_f64;
    let mut samples = Vec::new();
    let period = on + off;

    for step in 0..HORIZON * resolution.steps {
        let cycle = f64::from(step % (period * resolution.steps)) / f64::from(resolution.steps)
            + 0.5 / f64::from(resolution.steps);
        let pressure = if cycle < f64::from(on) {
            let ramp = cycle.min(f64::from(on) - cycle).min(1.0);
            let envelope = (PI * ramp / 2.0).sin().powi(2);
            AMBIENT * (1.0 + 0.03 * envelope * (TAU * cycle).sin())
        } else {
            AMBIENT
        };
        let exchange = bubble
            .advance(dt, Pressure::from_base(pressure))
            .unwrap_or_else(|error| panic!("step {step}, cycle {cycle}: {error}"));
        let mol = exchange.into_bubble.into_base();
        let outer = exchange.into_reservoir.into_base();
        gross += mol.abs() + outer.abs();
        reservoir += outer;
        let wall = bubble.state();
        let radius = wall.radius.into_base();
        let velocity = wall.wall_velocity.into_base();
        peak_speed = peak_speed.max(velocity.abs());
        minimum_radius = minimum_radius.min(radius);
        maximum_radius = maximum_radius.max(radius);
        let elapsed = step + 1;
        let phase_end = elapsed % (period * resolution.steps);
        if phase_end == 0 || phase_end == on * resolution.steps {
            let change = bubble.gas_amount_change().into_base();
            let total = 1e6 * change / initial;
            let residual = change + bubble.liquid_excess().into_base() + reservoir;
            max_balance = max_balance.max(residual.abs() / gross);
            let equilibrium = bubble.equilibrium_radius().unwrap().into_base();
            let eos_pressure = wall.gas_amount.into_base() * GAS_CONSTANT * 293.15
                / ((4.0 * PI / 3.0) * equilibrium.powi(3));
            let static_pressure = AMBIENT + 2.0 * SIGMA / equilibrium;
            assert!(
                (eos_pressure - static_pressure).abs() <= 32.0 * f64::EPSILON * static_pressure
            );
            samples.push(Sample {
                end: elapsed / resolution.steps,
                phase: if phase_end == 0 {
                    Phase::Off
                } else {
                    Phase::On
                },
                amount: total - previous_total,
                total,
                radius,
                velocity,
                equilibrium: 1e9 * (equilibrium - R0),
            });
            previous_total = total;
        }
    }
    info!(
        steps = resolution.steps,
        cells = resolution.cells,
        extent = resolution.extent,
        max_balance,
        peak_speed,
        minimum_radius,
        maximum_radius,
        "exposure results"
    );
    assert!(max_balance <= BALANCE_TOLERANCE, "{max_balance:e}");
    samples
}

fn compare(actual: &[Sample], expected: &[Sample], oracle: &[Sample], fraction: f64) {
    assert_eq!(actual.len(), expected.len());
    assert_eq!(actual.len(), oracle.len());
    let mut largest = [0.0_f64; 5];
    for ((actual, expected), oracle) in actual.iter().zip(expected).zip(oracle) {
        assert_eq!((actual.end, actual.phase), (expected.end, expected.phase));
        assert_eq!((actual.end, actual.phase), (oracle.end, oracle.phase));
        let amount_budget = (oracle.amount.abs() * 0.02).max(0.003);
        let total_budget = (oracle.total.abs() * 0.02).max(0.003);
        // dR_eq/dn = R_eq/[n(3−2σ/(R_eq p_d))] ≤ R_eq/(2n).
        // Bound R_eq/n over these fixtures by 1.01 times the initial ratio.
        let equilibrium_budget = 1.01 * R0 * 500.0 * total_budget;
        let quantities = [
            (
                "phase moles (ppm)",
                actual.amount,
                expected.amount,
                amount_budget,
            ),
            (
                "total moles (ppm)",
                actual.total,
                expected.total,
                total_budget,
            ),
            ("wall radius (m)", actual.radius, expected.radius, 1e-10),
            ("wall speed (m/s)", actual.velocity, expected.velocity, 5e-5),
            (
                "equilibrium radius change (nm)",
                actual.equilibrium,
                expected.equilibrium,
                equilibrium_budget,
            ),
        ];
        for ((name, value, target, budget), maximum) in quantities.into_iter().zip(&mut largest) {
            let ratio = (value - target).abs() / budget;
            assert!(
                ratio <= fraction,
                "{name}, cycle {} {:?}: {value:e} vs {target:e}, error/budget={ratio} > {fraction}",
                actual.end,
                actual.phase,
            );
            *maximum = maximum.max(ratio);
        }
    }
    info!(
        ?largest,
        "maximum fractions of the independent acceptance budgets"
    );
}

fn verify(names: &[&str]) {
    traced(|| {
        for name in names {
            let _scenario = info_span!("scenario", name).entered();
            let (on, off, saturation, oracle) = info_span!("arrange").in_scope(|| reference(name));
            let baseline = info_span!("act", axis = "baseline")
                .in_scope(|| simulate(BASE, on, off, saturation));
            info_span!("assert", axis = "independent reference").in_scope(|| {
                compare(&baseline, &oracle, &oracle, 1.0);
                if *name == "baseline" {
                    let reset = reference("reset_control").3;
                    // Same mechanics, inventory and forcing at the restart.
                    // Only the independent control clears the liquid profile.
                    let retained = baseline[2].amount;
                    let reset_uptake = reset[2].amount;
                    assert!(retained + 0.003 < reset_uptake);
                    assert!((retained - oracle[2].amount).abs() <= 0.003);
                }
            });
            for (axis, resolution) in REFINEMENTS {
                let refined =
                    info_span!("act", axis).in_scope(|| simulate(resolution, on, off, saturation));
                info_span!("assert", axis)
                    .in_scope(|| compare(&refined, &baseline, &oracle, 1.0 / 3.0));
            }
        }
    });
}

#[test]
fn retained_history_refines_in_the_reference_schedule() {
    verify(&["baseline"]);
}

#[test]
fn retained_history_refines_across_quiet_intervals() {
    verify(&["short_quiet", "long_quiet"]);
}

#[test]
fn retained_history_refines_across_pulse_durations() {
    verify(&["short_pulse", "long_pulse"]);
}

#[test]
fn retained_history_refines_at_equal_duty_cycle() {
    verify(&["50pct_short", "50pct_long"]);
}

#[test]
fn retained_history_refines_across_bulk_concentrations() {
    verify(&["undersaturated", "supersaturated"]);
}
