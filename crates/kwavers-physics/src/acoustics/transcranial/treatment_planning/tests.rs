use super::*;
use kwavers_grid::Grid;
use leto::Array3;

#[test]
fn test_treatment_planner_creation() {
    let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
    let ct_data = Array3::from_elem((64, 64, 64), 0.0); // Air

    let _planner = TreatmentPlanner::new(&grid, &ct_data).unwrap();
}

#[test]
fn test_treatment_plan_generation() {
    let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002).unwrap();
    let ct_data = Array3::from_elem((32, 32, 32), 400.0); // Bone-like HU
    let planner = TreatmentPlanner::new(&grid, &ct_data).unwrap();

    let target = TranscranialTargetVolume {
        center: [0.016, 0.016, 0.016],
        dimensions: [0.004, 0.004, 0.004],
        shape: TranscranialTargetShape::Ellipsoidal,
        priority: 8,
        max_temperature: 45.0,
        required_intensity: 100.0,
    };

    let spec = TranscranialTransducerSpecification::default();
    let plan = planner
        .generate_plan("test_patient", &[target], &spec)
        .unwrap();
    assert_eq!(plan.patient_id, "test_patient");
}

/// Grid + spec + target sized so the focused bowl (radius = focal_distance)
/// lies entirely inside the CT volume, so the aberration ray-trace samples real
/// voxels rather than clamped edges.
fn aberration_fixture() -> (
    Grid,
    TranscranialTransducerSpecification,
    TranscranialTargetVolume,
) {
    let grid = Grid::new(40, 40, 40, 1e-3, 1e-3, 1e-3).unwrap(); // 40 mm cube
    let spec = TranscranialTransducerSpecification {
        num_elements: 64,
        frequency: 650e3,
        focal_distance: 12.0, // mm → 12 mm radius of curvature
        radius: 12.0,
        sound_speed: 1500.0,
    };
    let target = TranscranialTargetVolume {
        center: [0.020, 0.020, 0.020], // grid centre, metres
        dimensions: [0.004, 0.004, 0.004],
        shape: TranscranialTargetShape::Ellipsoidal,
        priority: 8,
        max_temperature: 45.0,
        required_intensity: 100.0,
    };
    (grid, spec, target)
}

fn phase_span(phases: &[f64]) -> f64 {
    let max = phases.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min = phases.iter().copied().fold(f64::INFINITY, f64::min);
    max - min
}

/// A homogeneous CT volume leaves the equidistant bowl exactly in phase: every
/// element ray accrues the same aberration `Δφ = (k_local−k_water)·R`, so the
/// correction is a constant offset and the phase span is ~0.
#[test]
fn homogeneous_ct_keeps_bowl_in_phase() {
    let (grid, spec, target) = aberration_fixture();
    let ct = Array3::from_elem((40, 40, 40), 0.0); // uniform soft-tissue HU
    let planner = TreatmentPlanner::new(&grid, &ct).unwrap();

    let setup = planner.optimize_transducer_setup(&[target], &spec).unwrap();
    assert_eq!(setup.element_phases.len(), spec.num_elements);
    assert!(
        phase_span(&setup.element_phases) < 1e-6,
        "uniform medium must leave equidistant bowl in phase; span = {}",
        phase_span(&setup.element_phases)
    );
}

/// A bone slab between the array and the target makes the per-ray aberration
/// path-length-dependent: polar elements cross the full slab, equatorial ones
/// skim below it. The CT aberration correction must therefore (a) spread the
/// element phases and (b) differ element-wise from the homogeneous plan —
/// proving the corrector is wired in, not dead.
#[test]
fn skull_slab_induces_ray_dependent_aberration_correction() {
    let (grid, spec, target) = aberration_fixture();

    let homogeneous = Array3::from_elem((40, 40, 40), 0.0);
    let mut skull = homogeneous.clone();
    // Cortical-bone slab (≈1500 HU) at z ∈ [26,30) mm, between the +z bowl and
    // the focus at z = 20 mm.
    for iz in 26..30 {
        for ix in 0..40 {
            for iy in 0..40 {
                skull[[ix, iy, iz]] = 1500.0;
            }
        }
    }

    let phases_homog = TreatmentPlanner::new(&grid, &homogeneous)
        .unwrap()
        .optimize_transducer_setup(std::slice::from_ref(&target), &spec)
        .unwrap()
        .element_phases;
    let phases_skull = TreatmentPlanner::new(&grid, &skull)
        .unwrap()
        .optimize_transducer_setup(std::slice::from_ref(&target), &spec)
        .unwrap()
        .element_phases;

    // (a) The skull induces a real ray-dependent phase spread.
    assert!(
        phase_span(&phases_skull) > 0.1,
        "skull slab must spread element phases; span = {}",
        phase_span(&phases_skull)
    );
    // (b) The correction genuinely changed the phases versus the homogeneous plan.
    let altered = phases_homog
        .iter()
        .zip(phases_skull.iter())
        .any(|(h, s)| (h - s).abs() > 1e-6);
    assert!(
        altered,
        "aberration correction must alter element phases through a skull"
    );
}

#[test]
fn test_skull_properties_analysis() {
    let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
    let ct_data = Array3::from_elem((16, 16, 16), 800.0); // Dense bone
    let planner = TreatmentPlanner::new(&grid, &ct_data).unwrap();

    let props = planner.analyze_skull_properties().unwrap();
    assert_eq!(props.sound_speed.shape(), [16, 16, 16]);
    assert_eq!(props.density.shape(), [16, 16, 16]);
    assert_eq!(props.attenuation.shape(), [16, 16, 16]);
}
