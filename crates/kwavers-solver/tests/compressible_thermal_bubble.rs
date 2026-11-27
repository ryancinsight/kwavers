use aequitas::systems::si::quantities::{
    AreaPerTime, DynamicViscosity, Length, MassDensity, MolarConcentration, Pressure, PressureRate,
    SurfaceTension, ThermodynamicTemperature, Time, Velocity,
};
use kwavers_physics::acoustics::bubble_dynamics::{
    gas_transfer::SphericalGasDiffusion, rayleigh_plesset::PolytropicParameters,
};
use kwavers_solver::forward::bubble_dynamics::gas_transfer::{
    BubbleForcing, DiffusiveBubble, DiffusiveBubbleError, DiffusiveBubbleModels, GasHeatTransfer,
    RadialMechanics,
};
use std::f64::consts::{FRAC_PI_2, PI, TAU};
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

const R0: f64 = 30e-6;
const AMBIENT: f64 = 101_325.0;
const AMPLITUDE: f64 = 3039.75;
const ORACLE: &str = include_str!("fixtures/diffusive_bubble/compressible_thermal.csv");

fn traced(run: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, run);
}

fn models(compressible: bool, heat: bool, intervals: usize) -> DiffusiveBubbleModels {
    DiffusiveBubbleModels {
        mechanics: if compressible {
            RadialMechanics::KellerMiksis {
                liquid_sound_speed: Velocity::from_base(1480.0),
            }
        } else {
            RadialMechanics::RayleighPlesset
        },
        heat_transfer: heat.then_some(GasHeatTransfer {
            heat_capacity_ratio: 1.4,
            reference_diffusivity: AreaPerTime::from_base(2.2e-5),
            radial_intervals: intervals,
        }),
    }
}

fn fixture(models: DiffusiveBubbleModels, cells: usize) -> DiffusiveBubble {
    DiffusiveBubble::new_with_models(
        PolytropicParameters {
            ambient_pressure: Pressure::from_base(AMBIENT),
            vapor_pressure: Pressure::from_base(0.0),
            liquid_density: MassDensity::from_base(998.0),
            liquid_viscosity: DynamicViscosity::from_base(1e-3),
            surface_tension: SurfaceTension::from_base(0.072),
            reference_temperature: ThermodynamicTemperature::from_base(293.15),
            exponent: 1.0,
        },
        SphericalGasDiffusion {
            reference_radius: Length::from_base(R0),
            diffusivity: AreaPerTime::from_base(2e-9),
            bulk_concentration: MolarConcentration::from_base(
                (0.26 / AMBIENT) * (AMBIENT + 2.0 * 0.072 / R0),
            ),
        },
        MolarConcentration::from_base(0.26),
        Length::from_base(120e-6),
        cells,
        models,
    )
    .unwrap()
}

fn forcing(frequency: f64, cycle: f64, cycles: f64, on: bool) -> BubbleForcing {
    if !on {
        return BubbleForcing::steady(Pressure::from_base(AMBIENT));
    }
    let ramp = cycle.min(cycles - cycle).min(1.0);
    let slope = if ramp == 1.0 {
        0.0
    } else if cycle < cycles / 2.0 {
        1.0
    } else {
        -1.0
    };
    let envelope = (FRAC_PI_2 * ramp).sin().powi(2);
    let (sine, cosine) = (TAU * cycle).sin_cos();
    BubbleForcing {
        pressure: Pressure::from_base(AMBIENT + AMPLITUDE * envelope * sine),
        pressure_rate: PressureRate::from_base(
            AMPLITUDE
                * frequency
                * (TAU * envelope * cosine + FRAC_PI_2 * slope * (PI * ramp).sin() * sine),
        ),
    }
}

#[derive(Clone, Copy)]
struct Resolution {
    steps: u32,
    liquid: usize,
    heat: usize,
}

const BASE: Resolution = Resolution {
    steps: 64,
    liquid: 384,
    heat: 48,
};

#[derive(Clone, Copy)]
struct PhaseSample {
    phase_ppm: f64,
    total_ppm: f64,
    radius: f64,
    speed: f64,
}

impl PhaseSample {
    fn error_fractions(self, other: Self, target: Self) -> [f64; 4] {
        [
            (self.phase_ppm - other.phase_ppm).abs() / 0.003_f64.max(0.02 * target.phase_ppm.abs()),
            (self.total_ppm - other.total_ppm).abs() / 0.003_f64.max(0.02 * target.total_ppm.abs()),
            (self.radius - other.radius).abs() / 1e-10,
            (self.speed - other.speed).abs() / 5e-5,
        ]
    }

    fn within(self, other: Self, target: Self, fraction: f64) {
        let errors = self.error_fractions(other, target);
        tracing::info!(
            ?errors,
            "fractions of phase gas, total gas, radius and speed budgets"
        );
        assert!(
            errors.iter().all(|error| *error <= fraction),
            "phase error fractions {errors:?}, limit {fraction}"
        );
    }
}

struct ExpectedPhase {
    elapsed: f64,
    sample: PhaseSample,
}

fn oracle(frequency: u32, model: &str) -> Vec<ExpectedPhase> {
    let mut phases = Vec::new();
    for line in ORACLE.lines().skip(1) {
        let [carrier, name, phase, elapsed, phase_ppm, total_ppm, radius, speed]: [&str; 8] =
            line.split(',').collect::<Vec<_>>().try_into().unwrap();
        if carrier.parse::<u32>().unwrap() != frequency || name != model {
            continue;
        }
        assert_eq!(phase.parse::<usize>().unwrap(), phases.len() + 1);
        phases.push(ExpectedPhase {
            elapsed: elapsed.parse().unwrap(),
            sample: PhaseSample {
                phase_ppm: phase_ppm.parse().unwrap(),
                total_ppm: total_ppm.parse().unwrap(),
                radius: radius.parse().unwrap(),
                speed: speed.parse().unwrap(),
            },
        });
    }
    assert_eq!(phases.len(), 4);
    phases
}

fn simulate(
    frequency: u32,
    expected: &[ExpectedPhase],
    compressible: bool,
    heat: bool,
    resolution: Resolution,
) -> Vec<PhaseSample> {
    let mut bubble = info_span!("arrange").in_scope(|| {
        fixture(
            models(compressible, heat, resolution.heat),
            resolution.liquid,
        )
    });
    let n0 = bubble.state().gas_amount.into_base();
    let frequency = f64::from(frequency);
    let dt = (frequency * f64::from(resolution.steps)).recip();
    let mut reservoir = 0.0;
    let mut gross = 0.0;
    let mut elapsed = 0.0;
    let on_cycles = frequency * 0.01;
    let off_cycles = frequency * 0.005625;
    [on_cycles, off_cycles, on_cycles, off_cycles]
        .into_iter()
        .zip(expected)
        .enumerate()
        .map(|(phase, (cycles, target))| {
            let _phase = info_span!("phase", phase, frequency, compressible, heat).entered();
            let start = bubble.gas_amount_change().into_base();
            let steps = (cycles * f64::from(resolution.steps)).round() as u32;
            info_span!("act").in_scope(|| {
                for step in 0..steps {
                    let cycle = (f64::from(step) + 0.5) / f64::from(resolution.steps);
                    let flux = bubble
                        .advance_with_forcing(
                            Time::from_base(dt),
                            forcing(frequency, cycle, cycles, phase % 2 == 0),
                        )
                        .unwrap_or_else(|error| panic!("step {step}: {error}"));
                    let outer = flux.into_reservoir.into_base();
                    reservoir += outer;
                    gross += flux.into_bubble.into_base().abs() + outer.abs();
                }
            });
            info_span!("assert").in_scope(|| {
                elapsed += dt * f64::from(steps);
                assert!((elapsed - target.elapsed).abs() <= 8.0 * f64::EPSILON * elapsed);
                let amount = bubble.gas_amount_change().into_base();
                let state = bubble.state();
                let sample = PhaseSample {
                    phase_ppm: 1e6 * (amount - start) / n0,
                    total_ppm: 1e6 * amount / n0,
                    radius: state.radius.into_base(),
                    speed: state.wall_velocity.into_base(),
                };
                sample.within(target.sample, target.sample, 1.0);
                let residual =
                    (amount + bubble.liquid_excess().into_base() + reservoir).abs() / gross;
                assert!(residual < 1e-8, "gas ledger {residual:e}");
                sample
            })
        })
        .collect()
}

fn verify(frequency: u32, model: &str, compressible: bool, heat: bool) {
    traced(|| {
        simulate(
            frequency,
            &oracle(frequency, model),
            compressible,
            heat,
            BASE,
        );
    });
}

const HALF_STEP: Resolution = Resolution {
    steps: 2 * BASE.steps,
    ..BASE
};
const DOUBLE_LIQUID_GRID: Resolution = Resolution {
    liquid: 2 * BASE.liquid,
    ..BASE
};
const DOUBLE_HEAT_GRID: Resolution = Resolution {
    heat: 2 * BASE.heat,
    ..BASE
};

fn verify_refinement(frequency: u32, axis: &str, refined: Resolution) {
    traced(|| {
        let expected = oracle(frequency, "km_thermal");
        let baseline = simulate(frequency, &expected, true, true, BASE);
        let _axis = info_span!("refinement", axis).entered();
        let actual = simulate(frequency, &expected, true, true, refined);
        info_span!("assert").in_scope(|| {
            for ((sample, base), target) in actual.into_iter().zip(baseline).zip(&expected) {
                sample.within(base, target.sample, 1.0 / 3.0);
            }
        });
    });
}

#[test]
fn km_thermal_time_refines_220() {
    verify_refinement(220_000, "time", HALF_STEP);
}

#[test]
fn km_thermal_time_refines_230() {
    verify_refinement(230_000, "time", HALF_STEP);
}

#[test]
fn km_thermal_liquid_grid_refines_220() {
    verify_refinement(220_000, "liquid grid", DOUBLE_LIQUID_GRID);
}

#[test]
fn km_thermal_liquid_grid_refines_230() {
    verify_refinement(230_000, "liquid grid", DOUBLE_LIQUID_GRID);
}

#[test]
fn km_thermal_heat_grid_refines_220() {
    verify_refinement(220_000, "heat grid", DOUBLE_HEAT_GRID);
}

#[test]
fn km_thermal_heat_grid_refines_230() {
    verify_refinement(230_000, "heat grid", DOUBLE_HEAT_GRID);
}

#[test]
fn rp_isothermal_220() {
    verify(220_000, "rp_isothermal", false, false);
}
#[test]
fn km_isothermal_220() {
    verify(220_000, "km_isothermal", true, false);
}
#[test]
fn rp_thermal_220() {
    verify(220_000, "rp_thermal", false, true);
}
#[test]
fn km_thermal_220() {
    verify(220_000, "km_thermal", true, true);
}
#[test]
fn rp_isothermal_230() {
    verify(230_000, "rp_isothermal", false, false);
}
#[test]
fn km_isothermal_230() {
    verify(230_000, "km_isothermal", true, false);
}
#[test]
fn rp_thermal_230() {
    verify(230_000, "rp_thermal", false, true);
}
#[test]
fn km_thermal_230() {
    verify(230_000, "km_thermal", true, true);
}

#[test]
fn rejected_step_preserves_compressible_thermal_history() {
    traced(|| {
        let (mut actual, mut control, dt, forcing) = info_span!("arrange").in_scope(|| {
            let selected = models(true, true, 32);
            let dt = Time::from_base(2e-8);
            let forcing = forcing(230_000.0, 0.8, 10.0, true);
            let (mut a, mut b) = (fixture(selected, 128), fixture(selected, 128));
            for _ in 0..64 {
                a.advance_with_forcing(dt, forcing).unwrap();
                b.advance_with_forcing(dt, forcing).unwrap();
            }
            (a, b, dt, forcing)
        });
        let before = (
            actual.state(),
            actual.liquid_excess(),
            actual.gas_pressure(),
        );
        let (failed, missing_rate, after) = info_span!("act").in_scope(|| {
            let failed = actual.advance_with_forcing(Time::from_base(1.0), forcing);
            let missing_rate = actual.advance(dt, forcing.pressure);
            (
                failed,
                missing_rate,
                (
                    actual.state(),
                    actual.liquid_excess(),
                    actual.gas_pressure(),
                ),
            )
        });
        info_span!("assert").in_scope(|| {
            assert_eq!(
                failed.err(),
                Some(DiffusiveBubbleError::InvalidState("wall Mach number"))
            );
            assert_eq!(
                missing_rate.err(),
                Some(DiffusiveBubbleError::InvalidState(
                    "Keller–Miksis requires pressure rate"
                ))
            );
            assert_eq!(before, after);
            let a = actual.advance_with_forcing(dt, forcing).unwrap();
            let b = control.advance_with_forcing(dt, forcing).unwrap();
            assert_eq!(a.into_bubble, b.into_bubble);
            assert_eq!(a.into_reservoir, b.into_reservoir);
            assert_eq!(actual.state(), control.state());
            assert_eq!(actual.gas_pressure(), control.gas_pressure());
        });
    });
}
