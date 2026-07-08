use approx::assert_abs_diff_eq;
use leto::Array3;

use crate::traits::BoundaryFieldType;
use crate::BoundaryCondition;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_grid::topology::{GridTopology, TopologyDimension};

use super::{PeriodicBoundaryCondition, PeriodicConfig};
use kwavers_core::constants::numerical::TWO_PI;

struct MockGrid;
impl GridTopology for MockGrid {
    fn dimensionality(&self) -> TopologyDimension {
        TopologyDimension::Three
    }
    fn size(&self) -> usize {
        1000
    }
    fn dimensions(&self) -> [usize; 3] {
        [10, 10, 10]
    }
    fn spacing(&self) -> [f64; 3] {
        [0.001, 0.001, 0.001]
    }
    fn extents(&self) -> [f64; 3] {
        [0.01, 0.01, 0.01]
    }
    fn indices_to_coordinates(&self, indices: [usize; 3]) -> [f64; 3] {
        let spacing = self.spacing();
        [
            indices[0] as f64 * spacing[0],
            indices[1] as f64 * spacing[1],
            indices[2] as f64 * spacing[2],
        ]
    }
    fn coordinates_to_indices(&self, coords: [f64; 3]) -> Option<[usize; 3]> {
        let spacing = self.spacing();
        let dims = self.dimensions();
        let i = (coords[0] / spacing[0]).floor() as usize;
        let j = (coords[1] / spacing[1]).floor() as usize;
        let k = (coords[2] / spacing[2]).floor() as usize;
        if i < dims[0] && j < dims[1] && k < dims[2] {
            Some([i, j, k])
        } else {
            None
        }
    }
    fn metric_coefficient(&self, _indices: [usize; 3]) -> f64 {
        1.0
    }
    fn is_uniform(&self) -> bool {
        true
    }
    fn k_max(&self) -> f64 {
        let spacing = self.spacing();
        std::f64::consts::PI / spacing[0]
    }
}

#[test]
fn test_periodic_wrapping_x() {
    let config = PeriodicConfig::new(true, false, false);
    let boundary = PeriodicBoundaryCondition::new(config).unwrap();

    let mut field = Array3::<f64>::zeros((10, 5, 5));

    for i in 1..9 {
        for j in 0..5 {
            for k in 0..5 {
                field[[i, j, k]] = i as f64;
            }
        }
    }

    boundary.wrap_x(field.view_mut());

    assert_abs_diff_eq!(field[[0, 2, 2]], 8.0, epsilon = 1e-12);
    assert_abs_diff_eq!(field[[9, 2, 2]], 1.0, epsilon = 1e-12);
}

#[test]
fn test_periodic_all_directions() {
    let config = PeriodicConfig::all();
    let mut boundary = PeriodicBoundaryCondition::new(config).unwrap();

    let mut field = Array3::<f64>::ones((8, 8, 8));
    field[[4, 4, 4]] = 42.0;

    boundary
        .apply_scalar_spatial(field.view_mut(), &MockGrid, 0, 1e-7)
        .unwrap();

    assert_abs_diff_eq!(field[[4, 4, 4]], 42.0, epsilon = 1e-12);
}

#[test]
fn test_standing_wave_resonance() {
    let config = PeriodicConfig::new(true, false, false);
    let _boundary = PeriodicBoundaryCondition::new(config).unwrap();

    let nx = 10;
    let dx = 0.001;
    let length = (nx as f64) * dx;
    let k = TWO_PI / length;
    let amplitude = 1.0;

    let mut field = Array3::<f64>::zeros((nx, 1, 1));
    for i in 0..nx {
        let x = (i as f64) * dx;
        field[[i, 0, 0]] = amplitude * (k * x).sin();
    }

    assert_abs_diff_eq!(field[[0, 0, 0]], 0.0, epsilon = 1e-12);

    let expected_at_boundary = (k * ((nx - 1) as f64) * dx).sin();
    assert_abs_diff_eq!(field[[nx - 1, 0, 0]], expected_at_boundary, epsilon = 1e-12);

    let node_idx = nx / 2;
    assert_abs_diff_eq!(field[[node_idx, 0, 0]].abs(), 0.0, epsilon = 1e-12);
}

#[test]
fn test_bloch_periodic() {
    let phase_x = std::f64::consts::PI / 4.0;
    let config = PeriodicConfig::new(true, false, false).with_bloch_phase([phase_x, 0.0, 0.0]);
    let boundary = PeriodicBoundaryCondition::new(config).unwrap();

    assert!(boundary.is_bloch());
    assert_abs_diff_eq!(boundary.bloch_phase()[0], phase_x, epsilon = 1e-12);

    let mut field = Array3::<f64>::zeros((10, 5, 5));
    field[[1, 2, 2]] = 1.0;

    boundary.wrap_x(field.view_mut());

    let expected = phase_x.cos();
    assert_abs_diff_eq!(field[[9, 2, 2]], expected, epsilon = 1e-12);
}

#[test]
fn test_boundary_condition_trait() {
    let config = PeriodicConfig::all();
    let boundary = PeriodicBoundaryCondition::new(config).unwrap();

    assert_eq!(boundary.name(), "Periodic Boundary");
    assert!(boundary.active_directions().x_min);
    assert!(boundary.active_directions().x_max);
    assert!(boundary.supports_field_type(BoundaryFieldType::Pressure));
    assert_abs_diff_eq!(
        boundary.reflection_coefficient(0.0, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM),
        0.0
    );
    assert!(!boundary.is_stateful());
}

#[test]
fn test_energy_conservation() {
    let config = PeriodicConfig::all();
    let mut boundary = PeriodicBoundaryCondition::new(config).unwrap();

    let mut field = Array3::<f64>::zeros((16, 16, 16));
    for i in 2..14 {
        for j in 2..14 {
            for k in 2..14 {
                let x = (i as f64) / 16.0;
                let y = (j as f64) / 16.0;
                let z = (k as f64) / 16.0;
                field[[i, j, k]] = (TWO_PI * x).sin() * (TWO_PI * y).sin() * (TWO_PI * z).sin();
            }
        }
    }

    let energy_before = field.iter().map(|&p| p * p).sum::<f64>();

    boundary
        .apply_scalar_spatial(field.view_mut(), &MockGrid, 0, 1e-7)
        .unwrap();

    let energy_after = field.iter().map(|&p| p * p).sum::<f64>();

    let relative_error = (energy_after - energy_before).abs() / energy_before;
    assert!(
        relative_error < 0.01,
        "Energy not conserved: before={}, after={}, error={}",
        energy_before,
        energy_after,
        relative_error
    );
}
