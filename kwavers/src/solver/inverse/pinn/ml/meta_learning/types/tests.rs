use super::pde_type::PdeType;
use super::physics::MetaLearningPhysicsParameters;
use super::task::{TaskData, TaskDataStatistics};
use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;

#[test]
fn test_pde_type_complexity() {
    assert_eq!(PdeType::Wave.complexity(), 0.2);
    assert_eq!(PdeType::Diffusion.complexity(), 0.3);
    assert_eq!(PdeType::NavierStokes.complexity(), 1.0);
    assert!(PdeType::Wave.complexity() < PdeType::NavierStokes.complexity());
}

#[test]
fn test_pde_type_num_equations() {
    assert_eq!(PdeType::Wave.num_equations(), 1);
    assert_eq!(PdeType::Elastic.num_equations(), 2);
    assert_eq!(PdeType::Electromagnetic.num_equations(), 6);
    assert_eq!(PdeType::NavierStokes.num_equations(), 4);
}

#[test]
fn test_pde_type_linearity() {
    assert!(PdeType::Wave.is_linear());
    assert!(PdeType::Diffusion.is_linear());
    assert!(!PdeType::NavierStokes.is_linear());
}

#[test]
fn test_physics_parameters_default() {
    let params = MetaLearningPhysicsParameters::default();
    assert_eq!(params.wave_speed, 343.0);
    assert_eq!(params.density, 1.2);
    assert!(params.viscosity.is_none());
}

#[test]
fn test_physics_parameters_presets() {
    let air = MetaLearningPhysicsParameters::acoustic_air();
    assert_eq!(air.wave_speed, 343.0);

    let water = MetaLearningPhysicsParameters::acoustic_water();
    assert_eq!(water.wave_speed, SOUND_SPEED_WATER_SIM);

    let tissue = MetaLearningPhysicsParameters::acoustic_tissue();
    assert_eq!(tissue.wave_speed, 1540.0);
    assert!(tissue.nonlinearity.unwrap() > 0.0);
}

#[test]
fn test_physics_parameters_fluid() {
    let fluid = MetaLearningPhysicsParameters::fluid(1000.0, 0.001);
    assert_eq!(fluid.density, 1000.0);
    assert_eq!(fluid.viscosity, Some(0.001));
}

#[test]
fn test_task_data_default() {
    let data = TaskData::default();
    assert!(data.is_empty());
    assert_eq!(data.total_points(), 0);
}

#[test]
fn test_task_data_with_capacity() {
    let data = TaskData::with_capacity(1000, 100, 50);
    assert!(data.is_empty());
    assert_eq!(data.collocation_points.capacity(), 1000);
    assert_eq!(data.boundary_data.capacity(), 100);
    assert_eq!(data.initial_data.capacity(), 50);
}

#[test]
fn test_task_data_statistics() {
    let mut data = TaskData::default();
    data.collocation_points.push((0.0, 0.0, 0.0));
    data.boundary_data.push((0.0, 0.0, 0.0, 0.0));
    data.initial_data.push((0.0, 0.0, 0.0, 0.0, 0.0));

    let stats: TaskDataStatistics = data.statistics();
    assert_eq!(stats.num_collocation, 1);
    assert_eq!(stats.num_boundary, 1);
    assert_eq!(stats.num_initial, 1);
    assert_eq!(stats.total, 3);
}

#[test]
fn test_task_data_not_empty() {
    let mut data = TaskData::default();
    data.collocation_points.push((0.0, 0.0, 0.0));
    assert!(!data.is_empty());
    assert_eq!(data.total_points(), 1);
}
