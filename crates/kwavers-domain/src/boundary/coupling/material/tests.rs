use super::*;
use kwavers_core::constants::fundamental::{
    DENSITY_TISSUE,
    DENSITY_WATER_NOMINAL,
    SOUND_SPEED_AIR,
    SOUND_SPEED_TISSUE,
    SOUND_SPEED_WATER_SIM,
};
use kwavers_core::constants::tissue_acoustics::{B_OVER_A_SOFT_TISSUE, B_OVER_A_WATER, DENSITY_AIR};
use crate::boundary::traits::BoundaryCondition;
use kwavers_grid::GridTopologyExt;
use kwavers_medium::properties::AcousticPropertyData;
use ndarray::Array3;

#[test]
fn test_material_interface_coefficients() {
    let material_1 = AcousticPropertyData {
        density: DENSITY_WATER_NOMINAL,
        sound_speed: SOUND_SPEED_WATER_SIM,
        absorption_coefficient: 0.1,
        absorption_power: 2.0,
        nonlinearity: B_OVER_A_WATER,
    };

    let material_2 = AcousticPropertyData {
        density: 1600.0,
        sound_speed: SOUND_SPEED_TISSUE,
        absorption_coefficient: 0.5,
        absorption_power: 1.1,
        nonlinearity: B_OVER_A_SOFT_TISSUE,
    };

    let interface = MaterialInterface::new(
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        material_1,
        material_2,
        0.0,
    );

    let r = interface.reflection_coefficient();
    let t = interface.transmission_coefficient();

    let z1 = interface.material_1.impedance();
    let z2 = interface.material_2.impedance();
    let energy_conservation = r * r + (z1 / z2) * t * t;
    assert!(
        (energy_conservation - 1.0).abs() < 1e-10,
        "Energy conservation violated: R² + (Z₁/Z₂)T² = {}, expected 1.0",
        energy_conservation
    );

    let incident = 1e5;
    let transmitted = interface.transmitted_pressure(incident);
    let reflected = interface.reflected_pressure(incident);

    assert!(transmitted > 0.0);
    assert!(reflected.abs() < incident.abs());
}

#[test]
fn test_material_interface_normal_incidence_water_tissue() {
    let material_water = AcousticPropertyData {
        density: DENSITY_WATER_NOMINAL,
        sound_speed: SOUND_SPEED_WATER_SIM,
        absorption_coefficient: 0.002,
        absorption_power: 2.0,
        nonlinearity: B_OVER_A_WATER,
    };

    let material_tissue = AcousticPropertyData {
        density: DENSITY_TISSUE,
        sound_speed: SOUND_SPEED_TISSUE,
        absorption_coefficient: 0.5,
        absorption_power: 1.1,
        nonlinearity: B_OVER_A_SOFT_TISSUE,
    };

    let grid = kwavers_grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001)
        .expect("Failed to create grid");
    let grid_adapter = grid.as_topology();

    let interface = MaterialInterface::new(
        [0.016, 0.016, 0.016],
        [1.0, 0.0, 0.0],
        material_water,
        material_tissue,
        0.001,
    );

    let mut field = Array3::<f64>::zeros((32, 32, 32));
    for i in 0..16 {
        for j in 0..32 {
            for k in 0..32 {
                field[[i, j, k]] = 1.0;
            }
        }
    }

    let mut interface_bc = interface;
    interface_bc
        .apply_scalar_spatial(field.view_mut(), &grid_adapter, 0, 1e-6)
        .unwrap();

    let r_expected = interface_bc.reflection_coefficient();
    let t_expected = interface_bc.transmission_coefficient();

    let water_side_value = field[[8, 16, 16]];
    let expected_water = 1.0 + r_expected * 1.0;
    assert!(
        (water_side_value - expected_water).abs() < 0.1,
        "Water side: got {}, expected {} (R={})",
        water_side_value,
        expected_water,
        r_expected
    );

    let tissue_side_value = field[[24, 16, 16]];
    let expected_tissue = t_expected * 1.0;
    assert!(
        (tissue_side_value - expected_tissue).abs() < 0.1,
        "Tissue side: got {}, expected {} (T={})",
        tissue_side_value,
        expected_tissue,
        t_expected
    );
}

#[test]
fn test_material_interface_energy_conservation() {
    let material_1 = AcousticPropertyData {
        density: DENSITY_WATER_NOMINAL,
        sound_speed: SOUND_SPEED_WATER_SIM,
        absorption_coefficient: 0.1,
        absorption_power: 2.0,
        nonlinearity: B_OVER_A_WATER,
    };

    let material_2 = AcousticPropertyData {
        density: 2000.0,
        sound_speed: 2000.0,
        absorption_coefficient: 0.3,
        absorption_power: 1.5,
        nonlinearity: B_OVER_A_SOFT_TISSUE,
    };

    let interface = MaterialInterface::new(
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0],
        material_1,
        material_2,
        0.0,
    );

    let r = interface.reflection_coefficient();
    let t = interface.transmission_coefficient();
    let z1 = material_1.impedance();
    let z2 = material_2.impedance();

    let energy_balance = r * r + (z1 / z2) * t * t;
    assert!(
        (energy_balance - 1.0).abs() < 1e-12,
        "Energy conservation violated: |R|² + (Z₁/Z₂)|T|² = {}, expected 1.0",
        energy_balance
    );
}

#[test]
fn test_material_interface_matched_impedance() {
    let material_1 = AcousticPropertyData {
        density: DENSITY_WATER_NOMINAL,
        sound_speed: SOUND_SPEED_WATER_SIM,
        absorption_coefficient: 0.1,
        absorption_power: 2.0,
        nonlinearity: B_OVER_A_WATER,
    };

    let material_2 = AcousticPropertyData {
        density: DENSITY_WATER_NOMINAL,
        sound_speed: SOUND_SPEED_WATER_SIM,
        absorption_coefficient: 0.2,
        absorption_power: 1.8,
        nonlinearity: B_OVER_A_WATER, // matched impedance: B/A irrelevant for R/T
    };

    let interface = MaterialInterface::new(
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0],
        material_1,
        material_2,
        0.0,
    );

    let r = interface.reflection_coefficient();
    let t = interface.transmission_coefficient();

    assert!(
        r.abs() < 1e-12,
        "Matched impedance should give R=0, got {}",
        r
    );
    assert!(
        (t - 1.0).abs() < 1e-12,
        "Matched impedance should give T=1, got {}",
        t
    );
}

#[test]
fn test_material_interface_large_impedance_mismatch() {
    let material_air = AcousticPropertyData {
        density: DENSITY_AIR,         // 1.204 kg/m³ (Duck 1990)
        sound_speed: SOUND_SPEED_AIR, // 343 m/s at room temperature
        absorption_coefficient: 0.01,
        absorption_power: 2.0,
        nonlinearity: 0.4, // air B/A (empirical; ideal gas: 2(γ-1)=0.8)
    };

    let material_water = AcousticPropertyData {
        density: DENSITY_WATER_NOMINAL,
        sound_speed: SOUND_SPEED_WATER_SIM,
        absorption_coefficient: 0.002,
        absorption_power: 2.0,
        nonlinearity: B_OVER_A_WATER,
    };

    let interface = MaterialInterface::new(
        [0.5, 0.5, 0.5],
        [1.0, 0.0, 0.0],
        material_air,
        material_water,
        0.0,
    );

    let r = interface.reflection_coefficient();
    let t = interface.transmission_coefficient();

    assert!(r > 0.99, "Air-water interface should have R ≈ 1, got {}", r);
    assert!(
        t > 1.99 && t < 2.01,
        "Air-water pressure transmission should be T ≈ 2, got {}",
        t
    );

    let z1 = material_air.impedance();
    let z2 = material_water.impedance();
    let energy = r * r + (z1 / z2) * t * t;
    assert!(
        (energy - 1.0).abs() < 1e-10,
        "Energy not conserved for extreme mismatch: {}",
        energy
    );
}

#[test]
fn test_material_interface_field_continuity() {
    let material_1 = AcousticPropertyData {
        density: DENSITY_WATER_NOMINAL,
        sound_speed: SOUND_SPEED_WATER_SIM,
        absorption_coefficient: 0.1,
        absorption_power: 2.0,
        nonlinearity: B_OVER_A_WATER,
    };

    let material_2 = AcousticPropertyData {
        density: 1500.0,
        sound_speed: 1800.0,
        absorption_coefficient: 0.3,
        absorption_power: 1.5,
        nonlinearity: B_OVER_A_SOFT_TISSUE,
    };

    let grid = kwavers_grid::Grid::new(64, 64, 64, 0.001, 0.001, 0.001)
        .expect("Failed to create grid");
    let grid_adapter = grid.as_topology();

    let interface = MaterialInterface::new(
        [0.032, 0.032, 0.032],
        [1.0, 0.0, 0.0],
        material_1,
        material_2,
        0.002,
    );
    let mut field = Array3::<f64>::zeros((64, 64, 64));

    for i in 0..32 {
        for j in 0..64 {
            for k in 0..64 {
                field[[i, j, k]] = 1.0;
            }
        }
    }

    let mut interface_bc = interface;
    interface_bc
        .apply_scalar_spatial(field.view_mut(), &grid_adapter, 0, 1e-6)
        .unwrap();

    let left_of_interface = field[[31, 32, 32]];
    let right_of_interface = field[[32, 32, 32]];
    let jump = (left_of_interface - right_of_interface).abs();
    assert!(
        jump < 0.5,
        "Sharp discontinuity at interface: left={}, right={}, jump={}",
        left_of_interface,
        right_of_interface,
        jump
    );
}

#[test]
fn test_material_interface_zero_thickness() {
    let material_1 = AcousticPropertyData {
        density: DENSITY_WATER_NOMINAL,
        sound_speed: SOUND_SPEED_WATER_SIM,
        absorption_coefficient: 0.1,
        absorption_power: 2.0,
        nonlinearity: B_OVER_A_WATER,
    };

    let material_2 = AcousticPropertyData {
        density: 1200.0,
        sound_speed: 1600.0,
        absorption_coefficient: 0.2,
        absorption_power: 1.8,
        nonlinearity: B_OVER_A_SOFT_TISSUE,
    };

    let grid = kwavers_grid::Grid::new(32, 32, 32, 0.001, 0.001, 0.001)
        .expect("Failed to create grid");
    let grid_adapter = grid.as_topology();

    let interface = MaterialInterface::new(
        [0.016, 0.016, 0.016],
        [1.0, 0.0, 0.0],
        material_1,
        material_2,
        0.0,
    );
    let mut field = Array3::<f64>::ones((32, 32, 32));

    let mut interface_bc = interface;
    interface_bc
        .apply_scalar_spatial(field.view_mut(), &grid_adapter, 0, 1e-6)
        .unwrap();
}
