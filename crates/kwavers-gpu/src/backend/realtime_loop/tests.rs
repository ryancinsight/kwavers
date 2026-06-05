//! Tests for `RealtimeSimulationOrchestrator`.

use std::collections::HashMap;

use ndarray::Array3;

use crate::backend::physics_kernels::{
    GpuKernelPhysicsDomain, PhysicsKernel, PhysicsKernelRegistry, WorkgroupConfig,
};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;

use super::orchestrator::RealtimeSimulationOrchestrator;
use super::types::{RealtimeConfig, StepResult};

#[test]
fn test_config_default() {
    let config = RealtimeConfig::default();
    assert_eq!(config.budget_ms, 10.0);
    assert!(config.adaptive_timestepping);
    assert_eq!(config.cfl_safety_factor, 0.9);
}

#[test]
fn test_orchestrator_creation() -> KwaversResult<()> {
    let config = RealtimeConfig::default();
    let registry = PhysicsKernelRegistry::new();
    let orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;

    assert_eq!(orchestrator.step_count(), 0);
    Ok(())
}

#[test]
fn test_step_result_creation() {
    let result = StepResult {
        dt: 1e-6,
        time: 1e-5,
        wall_time_ms: 5.0,
        within_budget: true,
        kernels_executed: 3,
    };

    assert_eq!(result.dt, 1e-6);
    assert!(result.within_budget);
}

#[test]
fn test_budget_enforcement() -> KwaversResult<()> {
    let config = RealtimeConfig {
        budget_ms: 5.0,
        ..Default::default()
    };
    let registry = PhysicsKernelRegistry::new();
    let mut orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;

    let mut fields = HashMap::new();
    let grid = Grid::new(64, 64, 64, 0.1, 0.1, 0.1)?;

    let result = orchestrator.step(&mut fields, 1e-6, 0.0, &grid)?;

    assert_eq!(result.kernels_executed, 0);
    assert!(result.time == 0.0);
    assert_eq!(orchestrator.step_count(), 1);

    Ok(())
}

#[test]
fn test_nonempty_fields_require_registered_kernel() -> KwaversResult<()> {
    let config = RealtimeConfig::default();
    let registry = PhysicsKernelRegistry::new();
    let mut orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;
    let grid = Grid::new(4, 4, 4, 0.1, 0.1, 0.1)?;
    let mut fields = HashMap::from([("pressure".to_string(), Array3::zeros((4, 4, 4)))]);

    let error = orchestrator
        .step(&mut fields, 1e-6, 0.0, &grid)
        .unwrap_err();

    assert!(format!("{error}").contains("requires at least one registered physics kernel"));
    Ok(())
}

#[test]
fn test_registered_kernel_step_records_execution_metadata() -> KwaversResult<()> {
    let config = RealtimeConfig::default();
    let mut registry = PhysicsKernelRegistry::new();
    registry.register(PhysicsKernel::new(
        GpuKernelPhysicsDomain::AcousticFDTD,
        "@compute @workgroup_size(1) fn compute_main() {}".to_string(),
        "compute_main".to_string(),
        25,
        WorkgroupConfig::new(4, 4, 4),
    ))?;
    let mut orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;
    let grid = Grid::new(4, 4, 4, 0.1, 0.1, 0.1)?;
    let mut fields = HashMap::from([("pressure".to_string(), Array3::zeros((4, 4, 4)))]);

    let result = orchestrator.step(&mut fields, 1e-6, 1e-5, &grid)?;
    let metrics = orchestrator.get_metrics();

    assert_eq!(result.kernels_executed, 1);
    assert_eq!(result.dt, 1e-6);
    assert_eq!(result.time, 1e-5);
    assert!(metrics.avg_step_time_ms >= 0.0);
    assert_eq!(orchestrator.step_count(), 1);
    Ok(())
}

#[test]
fn test_timestep_adjustment() -> KwaversResult<()> {
    let config = RealtimeConfig {
        cfl_safety_factor: 0.8,
        ..Default::default()
    };
    let registry = PhysicsKernelRegistry::new();
    let orchestrator = RealtimeSimulationOrchestrator::new(config, registry)?;

    let dt = 1e-5;
    let adjusted = orchestrator.adjust_timestep(dt, 0.0, 1.0);

    assert!(adjusted <= dt);
    assert!((adjusted - dt * 0.8).abs() < 1e-15);

    Ok(())
}
