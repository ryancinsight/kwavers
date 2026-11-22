//! Independent acceptance checks for gas inventory feedback.

use aequitas::systems::si::quantities::{
    AmountOfSubstance, AreaPerTime, DynamicViscosity, Length, MassDensity, MolarConcentration,
    Pressure, SurfaceTension, ThermodynamicTemperature, Time,
};
use kwavers_physics::acoustics::bubble_dynamics::{
    gas_transfer::SphericalGasDiffusion,
    rayleigh_plesset::{PolytropicGas, PolytropicParameters},
};
use kwavers_solver::forward::bubble_dynamics::gas_transfer::DiffusiveBubble;
use tracing::info_span;
use tracing_subscriber::fmt::format::FmtSpan;

#[path = "diffusive_bubble/sensitivity.rs"]
mod sensitivity;

const R0: f64 = 30e-6;
const AMBIENT: f64 = 101_325.0;
const FREQUENCY: f64 = 26_600.0;
const SIGMA: f64 = 0.072;
const HENRY_SATURATION: f64 = 0.26;
const DIFFUSIVITY: f64 = 2e-9;

fn traced(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_span_events(FmtSpan::ENTER | FmtSpan::EXIT)
        .with_test_writer()
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

fn mechanics(exponent: f64) -> PolytropicParameters<f64> {
    PolytropicParameters {
        ambient_pressure: Pressure::from_base(AMBIENT),
        vapor_pressure: Pressure::from_base(0.0),
        liquid_density: MassDensity::from_base(998.0),
        liquid_viscosity: DynamicViscosity::from_base(1e-3),
        surface_tension: SurfaceTension::from_base(SIGMA),
        reference_temperature: ThermodynamicTemperature::from_base(293.15),
        exponent,
    }
}

fn fixture(cells: usize, extent: f64, bulk_fraction: f64) -> DiffusiveBubble {
    let initial_wall = HENRY_SATURATION / AMBIENT * (AMBIENT + 2.0 * SIGMA / R0);
    DiffusiveBubble::new(
        mechanics(1.0),
        SphericalGasDiffusion {
            reference_radius: Length::from_base(R0),
            diffusivity: AreaPerTime::from_base(DIFFUSIVITY),
            bulk_concentration: MolarConcentration::from_base(bulk_fraction * initial_wall),
        },
        MolarConcentration::from_base(HENRY_SATURATION),
        Length::from_base(extent),
        cells,
    )
    .expect("finite fixture")
}

#[test]
fn rejected_coupled_step_preserves_wall_and_liquid() {
    traced(|| {
        let (mut actual, mut control, dt, pressure) = info_span!("arrange").in_scope(|| {
            let mut actual = fixture(128, 120e-6, 1.0);
            let mut control = fixture(128, 120e-6, 1.0);
            let dt = Time::from_base(1e-8);
            let pressure = Pressure::from_base(AMBIENT - 2000.0);
            for _ in 0..32 {
                actual.advance(dt, pressure).unwrap();
                control.advance(dt, pressure).unwrap();
            }
            assert_ne!(actual.liquid_excess().into_base(), 0.0);
            (actual, control, dt, pressure)
        });
        let (before, liquid_before, outcome, after, liquid_after, a, b) = info_span!("act")
            .in_scope(|| {
                let before = actual.state();
                let liquid_before = actual.liquid_excess().into_base();
                let outcome = actual.advance(Time::from_base(1.0), pressure);
                let after = actual.state();
                let liquid_after = actual.liquid_excess().into_base();
                let a = actual
                    .advance(dt, pressure)
                    .unwrap()
                    .into_bubble
                    .into_base();
                let b = control
                    .advance(dt, pressure)
                    .unwrap()
                    .into_bubble
                    .into_base();
                (before, liquid_before, outcome, after, liquid_after, a, b)
            });
        info_span!("assert").in_scope(|| {
            assert!(outcome.is_err(), "the excessive coupled step must fail");
            assert_eq!(before, after);
            assert_eq!(liquid_before, liquid_after);
            assert_eq!(a, b);
            assert_eq!(actual.state(), control.state());
        });
    });
}

#[test]
fn equilibrium_radius_round_trips_the_matching_gas_law() {
    traced(|| {
        let cases = info_span!("arrange")
            .in_scope(|| [(1.0, SIGMA, 2338.0), (1.4, SIGMA, 2338.0), (1.0, 0.0, 0.0)]);
        for (exponent, sigma, vapor) in cases {
            let _case = info_span!("scenario", exponent, sigma, vapor).entered();
            let gas = info_span!("arrange").in_scope(|| {
                let mut parameters = mechanics(exponent);
                parameters.surface_tension = SurfaceTension::from_base(sigma);
                parameters.vapor_pressure = Pressure::from_base(vapor);
                PolytropicGas::new(Length::from_base(R0), parameters).unwrap()
            });
            let roots = info_span!("act").in_scope(|| {
                [0.37_f64, 0.88, 1.0, 1.33, 2.8].map(|scale| {
                    let radius = R0 * scale;
                    let dry = AMBIENT - vapor;
                    let pressure_ratio = (dry + 2.0 * sigma / radius) / (dry + 2.0 * sigma / R0);
                    let amount = gas.reference_amount().into_base()
                        * scale.powi(3)
                        * pressure_ratio.powf(exponent.recip());
                    (
                        radius,
                        gas.equilibrium_radius(AmountOfSubstance::from_base(amount)),
                    )
                })
            });
            info_span!("assert").in_scope(|| {
                for (expected, equilibrium) in roots {
                    let actual = equilibrium.unwrap().into_base();
                    assert!((actual - expected).abs() <= 32.0 * f64::EPSILON * expected);
                }
                assert!(gas
                    .equilibrium_radius(AmountOfSubstance::from_base(0.0))
                    .is_none());
            });
        }
    });
}

#[test]
fn rejects_exponents_outside_the_static_closure_domain() {
    traced(|| {
        let exponents =
            info_span!("arrange").in_scope(|| [1.0_f64.next_down(), 0.0, f64::NAN, f64::INFINITY]);
        let accepted = info_span!("act").in_scope(|| {
            exponents.map(|exponent| PolytropicGas::new(Length::from_base(R0), mechanics(exponent)))
        });
        info_span!("assert").in_scope(|| {
            assert!(accepted.into_iter().all(|result| result.is_none()));
            assert!(PolytropicGas::new(Length::from_base(R0), mechanics(1.0)).is_some());
        });
    });
}
