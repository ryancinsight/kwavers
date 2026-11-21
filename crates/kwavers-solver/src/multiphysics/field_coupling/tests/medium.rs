use aequitas::systems::si::quantities::{
    Frequency, MassDensity, ReciprocalLength, ReciprocalTime, SpecificHeatCapacity, Time, Velocity,
};
use hyperion::coefficient::InteractionCoefficient;
use kwavers_core::error::KwaversError;
use kwavers_field::indices::{LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX};
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use leto::Array3;
use std::sync::Arc;

use crate::multiphysics::{
    CoupledMultiPhysicsSolver, FieldCouplingMaterial, FieldCouplingStrategy,
    MultiphysicsFieldCoupler, PhotoelasticCoefficient,
};

const SHAPE: [usize; 3] = [2, 1, 1];

fn with_trace(test: impl FnOnce()) {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .finish();
    tracing::subscriber::with_default(subscriber, test);
}

fn local_material([x, ..]: [usize; 3], frequency: Frequency) -> FieldCouplingMaterial {
    let first = x == 0;
    let frequency_ratio = frequency / Frequency::from_base(1.0);
    FieldCouplingMaterial {
        photoelastic_coefficient: PhotoelasticCoefficient::from_base(if first {
            0.125
        } else {
            0.0
        }),
        photoelastic_response_rate: ReciprocalTime::from_base(1.0),
        optical_absorption: InteractionCoefficient::new(ReciprocalLength::from_base(if first {
            3.0
        } else {
            6.0
        }))
        .expect("finite positive optical absorption"),
        acoustic_intensity_absorption: ReciprocalLength::from_base(if first { 1.0 } else { 2.0 })
            * frequency_ratio,
        density: MassDensity::from_base(2.0),
        sound_speed: Velocity::from_base(4.0),
        specific_heat: SpecificHeatCapacity::from_base(4.0),
    }
}

#[test]
fn coupled_solver_applies_spatial_and_frequency_resolved_material() {
    with_trace(|| {
        for (hertz, expected_temperatures) in [(2.0, [12.5, 13.5]), (4.0, [12.75, 14.0])] {
            let frequency = Frequency::from_base(hertz);
            let mut solver = tracing::info_span!("arrange").in_scope(|| {
                let grid = Grid::new(2, 1, 1, 0.001, 0.001, 0.001).expect("valid grid");
                let medium = Arc::new(HomogeneousMedium::from_minimal(2.0, 4.0, &grid));
                let mut solver =
                    CoupledMultiPhysicsSolver::new(grid, medium, FieldCouplingStrategy::Weak)
                        .expect("solver initializes");
                for (field, value) in [
                    (PRESSURE_IDX, 4.0),
                    (TEMPERATURE_IDX, 10.0),
                    (LIGHT_IDX, 4.0),
                ] {
                    solver.fields_mut()[field] = Array3::from_elem(SHAPE, value);
                }
                solver
            });
            tracing::info_span!("act").in_scope(|| {
                solver
                    .step_with_medium(
                        Time::from_base(1.0),
                        frequency,
                        &|cell, sampled_frequency| Ok(local_material(cell, sampled_frequency)),
                    )
                    .expect("each grid cell has a valid material");
            });
            tracing::info_span!("assert").in_scope(|| {
                // I' = I(1 + β p k dt). Since ρ c_p = 8 and p²/(2ρc) = 1,
                // ΔT = (μ_a I' + α) dt / 8 for each cell.
                for (x, (light, temperature)) in [6.0, 4.0]
                    .into_iter()
                    .zip(expected_temperatures)
                    .enumerate()
                {
                    let cell = [x, 0, 0];
                    assert_eq!(solver.acoustic_pressure()[cell], 4.0);
                    assert_eq!(solver.optical_intensity()[cell], light);
                    assert_eq!(solver.temperature()[cell], temperature);
                }
            });
        }
    });
}

#[test]
fn invalid_material_in_later_cell_preserves_all_fields() {
    with_trace(|| {
        let (coupler, original) = tracing::info_span!("arrange").in_scope(|| {
            (
                MultiphysicsFieldCoupler::new(FieldCouplingStrategy::Weak),
                [4.0, 10.0, 4.0].map(|value| Array3::from_elem(SHAPE, value)),
            )
        });
        let mut fields = original.clone();
        let result = tracing::info_span!("act").in_scope(|| {
            coupler.couple_fields_with_medium(
                &mut fields,
                Time::from_base(1.0),
                Frequency::from_base(2.0),
                &|cell: [usize; 3], frequency| {
                    let mut material = local_material(cell, frequency);
                    if cell[0] == 1 {
                        // A finite density that would overflow the derived volumetric capacity.
                        material.density = MassDensity::from_base(f64::MAX);
                    }
                    Ok(material)
                },
            )
        });
        tracing::info_span!("assert").in_scope(|| {
            assert!(matches!(result, Err(KwaversError::InvalidInput(_))));
            for (actual, expected) in fields.iter().zip(&original) {
                assert!(actual.iter().eq(expected.iter()));
            }
        });
    });
}
