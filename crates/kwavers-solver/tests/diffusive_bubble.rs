use aequitas::systems::si::quantities::{
    AreaPerTime, DynamicViscosity, Length, MassDensity, MolarConcentration, Pressure,
    SurfaceTension, ThermodynamicTemperature, Time,
};
use eunomia::RealField;
use kwavers_physics::acoustics::bubble_dynamics::{
    gas_transfer::SphericalGasDiffusion, rayleigh_plesset::PolytropicParameters,
};
use kwavers_solver::forward::bubble_dynamics::gas_transfer::{
    DiffusiveBubble, DiffusiveBubbleError,
};
use std::f64::consts::{PI, TAU};
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

const FREQUENCY: f64 = 26_600.0;
const DRIVE: f64 = 20_000.0;
const AMBIENT: f64 = 101_325.0;
const VAPOR: f64 = 2_338.0;
const TENSION: f64 = 0.0728;
const SATURATION: f64 = 0.26;
const RATE_TOLERANCE: f64 = 0.01;
const STEPS_PER_CYCLE: usize = 8_192;
const RADIAL_CELLS: usize = 1_536;

fn traced(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

fn bubble<T: RealField>(
    radius: T,
    exponent: T,
    bulk_ratio: T,
    extent: T,
    cells: usize,
) -> DiffusiveBubble<T> {
    let ambient = T::from_f64(AMBIENT);
    let vapor = T::from_f64(VAPOR);
    let tension = T::from_f64(TENSION);
    let saturation = T::from_f64(SATURATION);
    let dry = ambient - vapor;
    let gas_pressure = dry + T::from_count(2) * tension / radius;
    let henry = saturation / dry;
    DiffusiveBubble::new(
        PolytropicParameters {
            ambient_pressure: Pressure::from_base(ambient),
            vapor_pressure: Pressure::from_base(vapor),
            liquid_density: MassDensity::from_base(T::from_f64(1_000.0)),
            liquid_viscosity: DynamicViscosity::from_base(T::from_f64(0.001_002)),
            surface_tension: SurfaceTension::from_base(tension),
            reference_temperature: ThermodynamicTemperature::from_base(T::from_f64(293.15)),
            exponent,
        },
        SphericalGasDiffusion {
            diffusivity: AreaPerTime::from_base(T::from_f64(2e-9)),
            reference_radius: Length::from_base(radius),
            bulk_concentration: MolarConcentration::from_base(henry * gas_pressure * bulk_ratio),
        },
        MolarConcentration::from_base(saturation),
        Length::from_base(extent),
        cells,
    )
    .unwrap()
}

fn drive(cycle: f64) -> f64 {
    let on_time = if cycle < 16.0 { cycle } else { cycle - 32.0 };
    if !(0.0..16.0).contains(&on_time) {
        return AMBIENT;
    }
    let ramp = on_time.min(16.0 - on_time).min(2.0);
    let envelope = 0.5 * (1.0 - (PI * ramp / 2.0).cos());
    AMBIENT + DRIVE * envelope * (TAU * cycle).sin()
}

#[test]
fn pulse_history_at_30_micrometres() {
    traced(|| {
        compare(
            30e-6,
            1.05,
            include_str!("fixtures/diffusive_bubble/radius-30.csv"),
        );
    });
}

#[test]
fn pulse_history_at_42_point_5_micrometres() {
    traced(|| {
        compare(
            42.5e-6,
            1.12,
            include_str!("fixtures/diffusive_bubble/radius-42p5.csv"),
        );
    });
}

fn compare(radius: f64, exponent: f64, reference: &str) {
    let (mut solution, reference_amount, conversion, dt) = info_span!("arrange").in_scope(|| {
        let solution = bubble(radius, exponent, 1.0, 60e-6, RADIAL_CELLS);
        let reference_amount = solution.state().gas_amount.into_base();
        let p0 = AMBIENT - VAPOR + 2.0 * TENSION / radius;
        // Linearize the static Rayleigh–Plesset balance about R0.
        let dn_dr = reference_amount * (3.0 - 2.0 * TENSION / (exponent * radius * p0)) / radius;
        let dt = Time::from_base(1.0 / (FREQUENCY * STEPS_PER_CYCLE as f64));
        (solution, reference_amount, 60e6 * FREQUENCY / dn_dr, dt)
    });

    let mut reservoir = 0.0;
    let mut gross = 0.0;
    let mut worst_balance = 0.0_f64;
    let mut phase_max = [0.0_f64; 2];
    let mut phase_count = [0; 2];
    let mut radius_error = 0.0_f64;
    let mut velocity_error = 0.0_f64;
    let mut minimum_radius = radius;
    let mut maximum_radius = radius;
    let mut peak_speed = 0.0_f64;
    for (index, row) in reference.lines().skip(1).enumerate() {
        let cycle = index + 1;
        let on = cycle <= 16 || (33..=48).contains(&cycle);
        let fields: Vec<_> = row.split(',').collect();
        let (rate, radius_sample, velocity) = info_span!("act", cycle, on).in_scope(|| {
            let before = solution.gas_amount_change().into_base();
            for step in 0..STEPS_PER_CYCLE {
                let midpoint_cycle = index as f64 + (step as f64 + 0.5) / STEPS_PER_CYCLE as f64;
                let transfer = solution
                    .advance(dt, Pressure::from_base(drive(midpoint_cycle)))
                    .unwrap_or_else(|error| panic!("cycle {cycle}, substep {step}: {error}"));
                let incoming = transfer.into_bubble.into_base();
                let outer = transfer.into_reservoir.into_base();
                reservoir += outer;
                gross += incoming.abs() + outer.abs();
                let state = solution.state();
                let wall_radius = state.radius.into_base();
                minimum_radius = minimum_radius.min(wall_radius);
                maximum_radius = maximum_radius.max(wall_radius);
                peak_speed = peak_speed.max(state.wall_velocity.into_base().abs());
                let balance = solution.gas_amount_change().into_base()
                    + solution.liquid_excess().into_base()
                    + reservoir;
                worst_balance = worst_balance.max(balance.abs());
            }
            let delta = solution.gas_amount_change().into_base() - before;
            let state = solution.state();
            (
                delta * conversion,
                state.radius.into_base(),
                state.wall_velocity.into_base(),
            )
        });
        info_span!("assert", cycle, on).in_scope(|| {
            assert_eq!(fields.len(), 6);
            assert_eq!(fields[0].parse::<usize>().unwrap(), cycle);
            assert_eq!(fields[1], if on { "on" } else { "off" });
            let reference_rate = fields[2].parse::<f64>().unwrap();
            let error = (rate - reference_rate).abs();
            assert!(error.is_finite());
            phase_max[usize::from(!on)] = phase_max[usize::from(!on)].max(error);
            phase_count[usize::from(!on)] += 1;
            let reference_radius = fields[4].parse::<f64>().unwrap();
            let reference_velocity = fields[5].parse::<f64>().unwrap();
            radius_error = radius_error.max((radius_sample - reference_radius).abs());
            velocity_error = velocity_error.max((velocity - reference_velocity).abs());
            tracing::info!(
                cycle,
                on,
                rate,
                reference_rate,
                error,
                radius_sample,
                velocity,
                "cycle comparison"
            );
        });
    }
    info_span!("assert").in_scope(|| {
        let relative_balance = worst_balance / gross;
        tracing::info!(
            radius,
            on_error = phase_max[0],
            off_error = phase_max[1],
            worst_balance,
            reference_amount,
            gross,
            relative_balance,
            radius_error,
            velocity_error,
            minimum_radius,
            maximum_radius,
            peak_speed,
            "pulse history summary"
        );
        assert_eq!(phase_count, [32, 40]);
        assert!(
            gross > 0.0 && relative_balance < 5e-9,
            "{relative_balance:e}"
        );
        assert!(
            phase_max.into_iter().all(|error| error <= RATE_TOLERANCE),
            "{phase_max:?}"
        );
        assert!(radius_error <= 1e-10, "{radius_error:e}");
        assert!(velocity_error <= 5e-5, "{velocity_error:e}");
    });
}

#[test]
fn saturated_static_state_preserves_equilibrium_at_each_precision() {
    traced(|| {
        equilibrium::<f64>();
        equilibrium::<f32>();
    });
}

fn equilibrium<T: RealField>() {
    let (mut solution, dt, ambient) = info_span!("arrange").in_scope(|| {
        (
            bubble(
                T::from_f64(30e-6),
                T::from_f64(1.05),
                T::ONE,
                T::from_f64(120e-6),
                128,
            ),
            Time::from_base(T::from_f64(1e-7)),
            Pressure::from_base(T::from_f64(AMBIENT)),
        )
    });
    let initial = solution.state();
    let transfer = info_span!("act").in_scope(|| solution.advance(dt, ambient).unwrap());
    info_span!("assert").in_scope(|| {
        assert_eq!(solution.state(), initial);
        assert_eq!(solution.liquid_excess().into_base(), T::ZERO);
        assert_eq!(transfer.into_bubble.into_base(), T::ZERO);
        assert_eq!(transfer.into_reservoir.into_base(), T::ZERO);
    });
}

#[test]
fn finite_reservoir_closes_budget_and_gas_uptake_expands_wall() {
    traced(|| {
        let (mut solution, dt, ambient) = info_span!("arrange").in_scope(|| {
            (
                bubble(30e-6, 1.05, 1.1, 1e-6, 128),
                Time::from_base(1e-7),
                Pressure::from_base(AMBIENT),
            )
        });
        let initial = solution.state();
        let (bubble_flux, reservoir, excess, state) = info_span!("act").in_scope(|| {
            let mut bubble_flux = 0.0;
            let mut reservoir = 0.0;
            for _ in 0..4_000 {
                let transfer = solution.advance(dt, ambient).unwrap();
                bubble_flux += transfer.into_bubble.into_base();
                reservoir += transfer.into_reservoir.into_base();
            }
            (
                bubble_flux,
                reservoir,
                solution.liquid_excess().into_base(),
                solution.state(),
            )
        });
        info_span!("assert").in_scope(|| {
            let balance =
                state.gas_amount.into_base() - initial.gas_amount.into_base() + excess + reservoir;
            tracing::info!(
                bubble_flux,
                reservoir,
                excess,
                balance,
                radius = state.radius.into_base(),
                "finite reservoir budget"
            );
            assert!(bubble_flux > 0.0 && excess < 0.0 && reservoir < -0.1 * bubble_flux);
            assert!(balance.abs() < 5e-9 * bubble_flux, "{balance:e}");
            // The ambient pressure never changes. With fixed n the wall would remain at R0.
            assert!(state.gas_amount > initial.gas_amount && state.radius > initial.radius);
        });
    });
}

#[test]
fn rejected_coupled_step_preserves_wall_inventory_and_diffusion_history() {
    traced(|| {
        let (mut solution, mut control, dt, pressure) = info_span!("arrange").in_scope(|| {
            // In pure solvent, the large step from a 100 nm sphere would
            // dissolve more gas than its finite initial inventory.
            let solution = bubble(100e-9, 1.05, 0.0, 10e-6, 128);
            let control = bubble(100e-9, 1.05, 0.0, 10e-6, 128);
            let dt = Time::from_base(1e-12);
            let pressure = Pressure::from_base(AMBIENT);
            (solution, control, dt, pressure)
        });
        let before = solution.state();
        let initial_liquid = solution.liquid_excess();
        let rejected =
            info_span!("act").in_scope(|| solution.advance(Time::from_base(1e-3), pressure));
        info_span!("assert").in_scope(|| {
            assert!(matches!(
                rejected,
                Err(DiffusiveBubbleError::InvalidState("trial gas amount"))
            ));
            assert_eq!(solution.state(), before);
            assert_eq!(solution.liquid_excess(), initial_liquid);
            let retry = solution.advance(dt, pressure).unwrap();
            let expected = control.advance(dt, pressure).unwrap();
            assert!(retry.into_bubble.into_base() < 0.0);
            assert_eq!(retry.into_bubble, expected.into_bubble);
            assert_eq!(retry.into_reservoir, expected.into_reservoir);
            assert_eq!(solution.state(), control.state());
            assert_eq!(solution.liquid_excess(), control.liquid_excess());
        });
    });
}
