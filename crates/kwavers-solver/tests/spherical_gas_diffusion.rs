use aequitas::systems::si::quantities::{AreaPerTime, Length, MolarConcentration, Time};
use kwavers_core::constants::cavitation::SURFACE_TENSION_WATER;
use kwavers_core::constants::fundamental::{ATMOSPHERIC_PRESSURE, GAS_CONSTANT};
use kwavers_core::constants::thermodynamic::T_AMBIENT;
use kwavers_physics::acoustics::bubble_dynamics::gas_transfer::{
    GasDiffusionWall, SphericalGasDiffusion,
};
use kwavers_solver::forward::bubble_dynamics::gas_transfer::{
    GasTransferError, TransientGasDiffusion,
};
use std::f64::consts::{PI, TAU};
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

const D: f64 = 2e-9;
const FREQUENCY: f64 = 26_600.0;
const C_SAT: f64 = 0.26;
const TOLERANCE: f64 = 0.01;

fn traced(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_ansi(false)
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

fn wall(radius: f64, concentration: f64) -> GasDiffusionWall {
    GasDiffusionWall {
        radius: Length::from_base(radius),
        concentration: MolarConcentration::from_base(concentration),
    }
}

fn solver(radius: f64, bulk: f64, boundary: f64, cells: usize) -> TransientGasDiffusion {
    TransientGasDiffusion::new(
        SphericalGasDiffusion {
            reference_radius: Length::from_base(radius),
            diffusivity: AreaPerTime::from_base(D),
            bulk_concentration: MolarConcentration::from_base(bulk),
        },
        Length::from_base(120e-6),
        cells,
        MolarConcentration::from_base(boundary),
    )
    .unwrap()
}

fn rate_conversion(radius: f64) -> f64 {
    let dn_dr = 4.0
        * PI
        * radius.powi(2)
        * (ATMOSPHERIC_PRESSURE + 4.0 * SURFACE_TENSION_WATER / (3.0 * radius))
        / (GAS_CONSTANT * T_AMBIENT);
    60e6 / dn_dr
}

struct Fixture {
    radius: f64,
    amplitude: f64,
    kappa: f64,
    reference: &'static str,
}

impl Fixture {
    fn bulk(&self) -> f64 {
        C_SAT * (1.0 + 2.0 * SURFACE_TENSION_WATER / (self.radius * ATMOSPHERIC_PRESSURE))
    }

    fn driven_wall(&self, cycle: f64) -> GasDiffusionWall {
        let on_time = if cycle < 16.0 {
            Some(cycle)
        } else if (32.0..48.0).contains(&cycle) {
            Some(cycle - 32.0)
        } else {
            None
        };
        let envelope = on_time.map_or(0.0, |time| {
            let ramp_progress = time.min(16.0 - time).min(2.0) / 2.0;
            0.5 * (1.0 - (PI * ramp_progress).cos())
        });
        let radius = self.radius * (1.0 - self.amplitude * envelope * (TAU * cycle).sin());
        wall(
            radius,
            self.bulk() * (self.radius / radius).powf(3.0 * self.kappa),
        )
    }
}

#[test]
fn pulse_on_and_off_match_resolved_spherical_reference() {
    traced(|| {
        let fixtures = info_span!("arrange").in_scope(|| {
            [
                Fixture {
                    radius: 30e-6,
                    amplitude: 0.065_342_064_503_02,
                    kappa: 1.049_728_319_340_68,
                    reference: include_str!("fixtures/spherical_gas_diffusion/radius-30.csv"),
                },
                Fixture {
                    radius: 42.5e-6,
                    amplitude: 0.066_557_616_606_386_3,
                    kappa: 1.118_868_535_965_087_4,
                    reference: include_str!("fixtures/spherical_gas_diffusion/radius-42p5.csv"),
                },
            ]
        });
        for fixture in fixtures {
            let span = info_span!("scenario", radius_um = fixture.radius * 1e6);
            let _entered = span.enter();
            let (rows, residual, gross_exchange) =
                info_span!("act").in_scope(|| compare(&fixture, 768, 512));
            let (refined, _, _) =
                info_span!("act", refinement = true).in_scope(|| compare(&fixture, 1536, 1024));
            info_span!("assert").in_scope(|| {
                assert_eq!(rows.len(), 72);
                assert_eq!(rows.len(), refined.len());
                let max_refinement = rows.iter().zip(refined).fold(0.0_f64, |max, (row, fine)| {
                    max.max((row.actual - fine.actual).abs())
                });
                assert!(max_refinement <= TOLERANCE / 2.0, "{max_refinement}");
                tracing::info!(max_refinement, "draft refinement");
                let mut maximum = [0.0_f64; 2];
                for row in rows {
                    let error = (row.actual - row.reference).abs();
                    let phase = usize::from(row.phase == "off");
                    maximum[phase] = maximum[phase].max(error);
                    tracing::debug!(
                        cycle = row.cycle,
                        phase = row.phase,
                        reference_rate = row.reference,
                        draft_rate = row.actual,
                        difference = error,
                        "cycle rate"
                    );
                }
                tracing::info!(
                    max_on = maximum[0],
                    max_off = maximum[1],
                    mass_residual = residual,
                    gross_exchange,
                    "history comparison"
                );
                assert!(
                    maximum.iter().all(|&error| error <= TOLERANCE),
                    "{maximum:?}"
                );
                assert!(residual.abs() <= 1e-8 * gross_exchange, "{residual:e}");
            });
        }
    });
}

struct Comparison<'a> {
    cycle: usize,
    phase: &'a str,
    reference: f64,
    actual: f64,
}

fn compare(fixture: &Fixture, cells: usize, steps: u32) -> (Vec<Comparison<'_>>, f64, f64) {
    let dt = Time::from_base((FREQUENCY * f64::from(steps)).recip());
    let mut solution = solver(fixture.radius, fixture.bulk(), fixture.bulk(), cells);
    let mut transferred = 0.0;
    let mut gross_exchange = 0.0;
    let mut result = Vec::new();
    for (index, line) in fixture.reference.lines().skip(1).enumerate() {
        let mut fields = line.split(',');
        let cycle: usize = fields.next().unwrap().parse().unwrap();
        let phase = fields.next().unwrap();
        let reference: f64 = fields.next().unwrap().parse().unwrap();
        assert_eq!(cycle, index + 1);
        let mut cycle_amount = 0.0;
        for step in 1..=steps {
            let time =
                f64::from(u32::try_from(index).unwrap()) + f64::from(step) / f64::from(steps);
            let exchange = solution.advance(dt, fixture.driven_wall(time)).unwrap();
            let bubble = exchange.into_bubble.into_base();
            let reservoir = exchange.into_reservoir.into_base();
            cycle_amount += bubble;
            transferred += bubble + reservoir;
            gross_exchange += bubble.abs() + reservoir.abs();
        }
        result.push(Comparison {
            cycle,
            phase,
            reference,
            actual: rate_conversion(fixture.radius) * FREQUENCY * cycle_amount,
        });
    }
    (
        result,
        solution.liquid_excess().into_base() + transferred,
        gross_exchange,
    )
}

#[test]
fn constant_spherical_wall_matches_diffusive_step_solution() {
    traced(|| {
        let (radius, initial, boundary, elapsed, steps) =
            info_span!("arrange").in_scope(|| (30e-6, 0.26, 0.27, 1e-3, 10_000));
        let transferred = info_span!("act").in_scope(|| {
            let mut solution = solver(radius, initial, boundary, 768);
            let dt = Time::from_base(elapsed / f64::from(steps));
            (0..steps).fold(0.0, |sum, _| {
                sum + solution
                    .advance(dt, wall(radius, boundary))
                    .unwrap()
                    .into_bubble
                    .into_base()
            })
        });
        info_span!("assert").in_scope(|| {
            // Exact for a fixed sphere in an infinite reservoir. Its flux
            // contains the 1/R and 1/sqrt(pi D t) terms.
            let expected = 4.0
                * PI
                * D
                * radius
                * (initial - boundary)
                * (elapsed + 2.0 * radius * (elapsed / (PI * D)).sqrt());
            let error = rate_conversion(radius) * (transferred - expected).abs() / elapsed;
            tracing::info!(error, "constant wall average-rate difference");
            assert!(error <= TOLERANCE, "{error}");
        });
    });
}

#[test]
fn failed_step_preserves_computed_history() {
    traced(|| {
        let (mut trial, mut control, dt) = info_span!("arrange").in_scope(|| {
            let mut trial = solver(30e-6, 0.26, 0.26, 256);
            let mut control = solver(30e-6, 0.26, 0.26, 256);
            let dt = Time::from_base(1e-6);
            for state in [&mut trial, &mut control] {
                assert_eq!(
                    state
                        .advance(dt, wall(30e-6, 0.26))
                        .unwrap()
                        .into_bubble
                        .into_base(),
                    0.0
                );
                for _ in 0..16 {
                    assert!(
                        state
                            .advance(dt, wall(30e-6, 0.27))
                            .unwrap()
                            .into_bubble
                            .into_base()
                            < 0.0
                    );
                }
            }
            (trial, control, dt)
        });
        let (invalid, before, after, trial_amount, control_amount) =
            info_span!("act").in_scope(|| {
                let before = trial.liquid_excess().into_base();
                let invalid = trial.advance(dt, wall(0.0, 0.27));
                let after = trial.liquid_excess().into_base();
                // The first interval returns the wall concentration to bulk. The
                // next holds it there while the previously enriched liquid diffuses.
                for state in [&mut trial, &mut control] {
                    state.advance(dt, wall(30e-6, 0.26)).unwrap();
                }
                let trial_amount = trial
                    .advance(dt, wall(30e-6, 0.26))
                    .unwrap()
                    .into_bubble
                    .into_base();
                let control_amount = control
                    .advance(dt, wall(30e-6, 0.26))
                    .unwrap()
                    .into_bubble
                    .into_base();
                (invalid, before, after, trial_amount, control_amount)
            });
        info_span!("assert").in_scope(|| {
            assert!(matches!(
                invalid,
                Err(GasTransferError::InvalidInput("wall radius"))
            ));
            assert!(before > 0.0);
            assert_eq!(before, after);
            assert!(trial_amount > 0.0);
            assert_eq!(trial_amount, control_amount);
        });
    });
}

#[test]
fn rejects_nonphysical_step_without_discarding_liquid() {
    traced(|| {
        let (mut solution, short, wall_at_bulk) = info_span!("arrange").in_scope(|| {
            let mut solution = solver(30e-6, 0.0, 0.26, 96);
            let short = Time::from_base(1e-8);
            let wall_at_bulk = wall(30e-6, 0.0);
            solution
                .advance(Time::from_base(1e-3), wall(30e-6, 0.26))
                .unwrap();
            solution.advance(short, wall_at_bulk).unwrap();
            (solution, short, wall_at_bulk)
        });
        let (before, rejected, after, recovered) = info_span!("act").in_scope(|| {
            let before = solution.liquid_excess().into_base();
            let rejected = solution.advance(Time::from_base(1.0), wall_at_bulk);
            let after = solution.liquid_excess().into_base();
            let recovered = solution.advance(short, wall_at_bulk).unwrap();
            (before, rejected, after, recovered)
        });
        info_span!("assert").in_scope(|| {
            assert!(matches!(
                rejected,
                Err(GasTransferError::NegativeConcentration)
            ));
            assert!(before > 0.0);
            assert_eq!(before, after);
            assert!(recovered.into_bubble.into_base() > 0.0);
            assert!(solution.liquid_excess().into_base() < before);
        });
    });
}
