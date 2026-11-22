//! Fourier eigenmodes of the production three-dimensional absorption solver.
use aequitas::systems::si::quantities::{
    Dimensionless, Frequency, Length, ReciprocalLength, Time, Velocity,
};
use kwavers_math::fft::Complex64;
use kwavers_medium::absorption::FractionalLaplacian;
use kwavers_solver::forward::viscoacoustic::FractionalAbsorptionOperator;
use leto::Array3;
use std::f64::consts::TAU;

fn trace(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

#[test]
fn periodic_modes_have_the_calibrated_amplitude_and_phase() {
    trace(|| {
        let shape = [8, 4, 2];
        for order in [1.0, 1.5, 2.0] {
            let op = tracing::info_span!("arrange", order).in_scope(|| {
                // c=2 m/s, t=1/2 s. k=1 rad/m is the reference frequency.
                let law = FractionalLaplacian::new(
                    ReciprocalLength::from_base(0.25),
                    Frequency::from_base(2. / TAU),
                    Dimensionless::from_base(order),
                )
                .unwrap();
                let step = law
                    .step(Velocity::from_base(2.), Time::from_base(0.5))
                    .unwrap();
                FractionalAbsorptionOperator::new(
                    shape,
                    shape.map(|n| Length::from_base(TAU / n as f64)),
                    step,
                )
                .unwrap()
            });
            for mode in [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [-1, 0, 0],
                [1, -1, 1],
                [4, 0, 0],
            ] {
                let initial = Array3::from_shape_fn(shape, |idx| {
                    let cycles: f64 = (0..3)
                        .map(|a| f64::from(mode[a]) * idx[a] as f64 / shape[a] as f64)
                        .sum();
                    Complex64::from_polar(1., TAU * cycles)
                });
                let mut pressure = initial.clone();
                tracing::info_span!("act", ?mode).in_scope(|| op.apply(&mut pressure).unwrap());
                tracing::info_span!("assert").in_scope(|| {
                    let squared: f64 = mode.iter().map(|&i| f64::from(i * i)).sum();
                    let gain = (-0.25 * squared.powf(order / 2.)).exp();
                    // Two radix-two FFTs, six butterfly stages each; allow four
                    // scalar roundoffs per stage plus the input phasor evaluation.
                    for (output, input) in pressure.iter().zip(initial.iter()) {
                        assert!(
                            (*output - *input * gain).norm() <= 64. * f64::EPSILON,
                            "order={order} mode={mode:?} gain={gain} output={output}"
                        );
                    }
                });
            }
            let mut wrong = Array3::from_elem([2, 4, 8], Complex64::new(1., 2.));
            let before = wrong.clone();
            assert!(op.apply(&mut wrong).is_err());
            assert_eq!(wrong, before);
        }
    });
}

#[test]
fn catalog_runs_legacy_and_reference_calibrations_at_the_requested_timestep() {
    use kwavers_boundary::DomainPMLBoundary;
    use kwavers_grid::Grid;
    use kwavers_medium::HomogeneousMedium;
    use kwavers_physics::factory::{PhysicsConfig, PhysicsModelConfig};
    use kwavers_solver::plugin::PhysicsCatalog;
    trace(|| {
        for malformed in [
            serde_json::json!({"reference_loss":1.0,"exponent":1.0}),
            serde_json::json!({"alpha0":0.25,"reference_frequency":10.0,"exponent":1.0}),
        ] {
            assert!(
                serde_json::from_value::<kwavers_physics::factory::PhysicsModelType>(
                    serde_json::json!({"FractionalViscoacoustic":malformed})
                )
                .is_err()
            );
        }
        // α(ω)=ω/4. The new calibration takes α=1/2 at ω=2.
        for material in [
            serde_json::json!({"alpha0":0.25,"exponent":1.0}),
            serde_json::json!({"reference_loss":0.5,"reference_frequency":2./TAU,"exponent":1.0}),
        ] {
            let (grid, medium, mut manager, mut fields, mut boundary) =
                tracing::info_span!("arrange").in_scope(|| {
                    let grid = Grid::new(8, 2, 2, TAU / 8., 1., 1.).unwrap();
                    let medium = HomogeneousMedium::new(1000., 2., 0., 0., &grid);
                    let mut config = PhysicsConfig::new();
                    config.models = vec![PhysicsModelConfig {
                        model_type: serde_json::from_value(
                            serde_json::json!({"FractionalViscoacoustic":material}),
                        )
                        .unwrap(),
                        enabled: true,
                        parameters: Default::default(),
                    }];
                    let mut manager = PhysicsCatalog::build(&config, &grid, &medium, 0.5).unwrap();
                    manager.initialize(&grid, &medium).unwrap();
                    let fields = leto::Array4::from_shape_fn([1, 8, 2, 2], |[_, x, _, _]| {
                        (TAU * x as f64 / 8.).cos()
                    });
                    (
                        grid,
                        medium,
                        manager,
                        fields,
                        DomainPMLBoundary::new(Default::default()).unwrap(),
                    )
                });
            let initial = fields.clone();
            let mut heterogeneous =
                kwavers_medium::heterogeneous::HeterogeneousMedium::from_homogeneous(
                    &medium, &grid,
                )
                .unwrap();
            heterogeneous.sound_speed[[1, 0, 0]] = 3.;
            assert!(manager
                .execute(
                    &mut fields,
                    &grid,
                    &heterogeneous,
                    &[],
                    &mut boundary,
                    0.5,
                    0.
                )
                .is_err());
            assert_eq!(fields, initial);
            manager
                .execute(&mut fields, &grid, &medium, &[], &mut boundary, 0., 0.)
                .unwrap();
            assert_eq!(fields, initial);
            tracing::info_span!("act").in_scope(|| {
                manager
                    .execute(&mut fields, &grid, &medium, &[], &mut boundary, 0.5, 0.)
                    .unwrap();
                manager
                    .execute(&mut fields, &grid, &medium, &[], &mut boundary, 0.25, 0.5)
                    .unwrap();
            });
            tracing::info_span!("assert").in_scope(|| {
                // k=1, c=2, total path=1.5 m, α=0.5/m, log loss=0.75.
                for (&got, &input) in fields.iter().zip(initial.iter()) {
                    assert!((got - input * (-0.75_f64).exp()).abs() <= 64. * f64::EPSILON);
                }
            });
        }
    });
}
