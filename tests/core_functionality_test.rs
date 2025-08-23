//! Core functionality tests for Kwavers
//! 
//! These tests verify that the fundamental operations work correctly.

use kwavers::{Grid, Time, error::KwaversResult};
use kwavers::solver::fdtd::{FdtdConfig, FdtdSolver};
use kwavers::solver::pstd::{PstdConfig, PstdSolver};
use kwavers::medium::homogeneous::HomogeneousMedium;
use kwavers::source::{PointSource, Signal};
use kwavers::boundary::{BoundaryCondition, PmlConfig};
use ndarray::Array3;

#[test]
fn test_grid_creation_valid() {
    // Test valid grid creation
    let grid = Grid::try_new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    assert!(grid.is_ok());
    
    let grid = grid.unwrap();
    assert_eq!(grid.nx, 64);
    assert_eq!(grid.ny, 64);
    assert_eq!(grid.nz, 64);
}

#[test]
fn test_grid_creation_invalid() {
    // Test invalid grid dimensions
    let grid = Grid::try_new(0, 64, 64, 1e-3, 1e-3, 1e-3);
    assert!(grid.is_err());
    
    // Test invalid spacing
    let grid = Grid::try_new(64, 64, 64, -1e-3, 1e-3, 1e-3);
    assert!(grid.is_err());
}

#[test]
fn test_fdtd_solver_initialization() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let config = FdtdConfig {
        cfl: 0.5,
        boundary_condition: BoundaryCondition::Pml(PmlConfig::default()),
    };
    
    let solver = FdtdSolver::new(config, &grid)?;
    assert!(solver.validate_stability(&grid)?);
    Ok(())
}

#[test]
fn test_pstd_solver_initialization() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let config = PstdConfig {
        spectral_radius: 0.5,
        k_max_ratio: 0.25,
    };
    
    let solver = PstdSolver::new(config, &grid)?;
    // Solver should initialize without panicking
    Ok(())
}

#[test]
fn test_homogeneous_medium() {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 1e-3, 0.072, &grid);
    
    // Test properties at various points
    assert_eq!(medium.density(0.0, 0.0, 0.0, &grid), 1000.0);
    assert_eq!(medium.sound_speed(15e-3, 15e-3, 15e-3, &grid), 1500.0);
    assert!(medium.is_homogeneous());
}

#[test]
fn test_point_source() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let signal = Signal::Sine {
        frequency: 1e6,
        amplitude: 1.0,
        phase: 0.0,
    };
    
    let source = PointSource::new([16e-3, 16e-3, 16e-3], signal);
    
    // Test source field generation
    let mut field = Array3::zeros((32, 32, 32));
    source.add_to_field(&mut field, 0.0, &grid)?;
    
    // Source should have added something to the field
    let sum: f64 = field.iter().map(|x| x.abs()).sum();
    assert!(sum > 0.0);
    
    Ok(())
}

#[test]
fn test_boundary_conditions() -> KwaversResult<()> {
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    
    // Test PML creation
    let pml_config = PmlConfig {
        thickness: 10,
        max_damping: 1.0,
        power: 2.0,
    };
    
    let boundary = BoundaryCondition::Pml(pml_config);
    
    // Should be able to create fields for the boundary
    match boundary {
        BoundaryCondition::Pml(config) => {
            assert_eq!(config.thickness, 10);
            assert_eq!(config.power, 2.0);
        }
        _ => panic!("Expected PML boundary"),
    }
    
    Ok(())
}

#[test]
fn test_wave_propagation_basic() -> KwaversResult<()> {
    // Simple test that wave propagation doesn't panic
    let grid = Grid::new(32, 32, 32, 1e-3, 1e-3, 1e-3);
    let time = Time::new(1e-7, 10);
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 1e-3, 0.072, &grid);
    
    let config = FdtdConfig {
        cfl: 0.5,
        boundary_condition: BoundaryCondition::Pml(PmlConfig::default()),
    };
    
    let solver = FdtdSolver::new(config, &grid)?;
    
    // Initialize fields
    let mut pressure = grid.create_field();
    let mut velocity_x = grid.create_field();
    let mut velocity_y = grid.create_field();
    let mut velocity_z = grid.create_field();
    
    // Add a source
    let signal = Signal::Gaussian {
        center_frequency: 1e6,
        bandwidth: 0.5e6,
        peak_time: 5e-6,
    };
    let source = PointSource::new([16e-3, 16e-3, 16e-3], signal);
    
    // Run a few time steps - this should not panic
    for step in 0..time.num_steps {
        let t = step as f64 * time.dt;
        source.add_to_field(&mut pressure, t, &grid)?;
        
        // Basic stability check
        let max_pressure: f64 = pressure.iter().map(|x| x.abs()).fold(0.0, f64::max);
        assert!(max_pressure.is_finite(), "Pressure became infinite at step {}", step);
        assert!(max_pressure < 1e10, "Pressure exploded at step {}", step);
    }
    
    Ok(())
}

#[test]
fn test_grid_field_creation() {
    let grid = Grid::new(10, 20, 30, 1e-3, 1e-3, 1e-3);
    let field = grid.create_field();
    
    assert_eq!(field.shape(), &[10, 20, 30]);
    assert_eq!(field.iter().sum::<f64>(), 0.0); // Should be zero-initialized
}

#[test]
fn test_position_to_indices() {
    let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3);
    
    // Test valid position
    let indices = grid.position_to_indices(5e-3, 5e-3, 5e-3);
    assert_eq!(indices, Some((5, 5, 5)));
    
    // Test boundary
    let indices = grid.position_to_indices(9.5e-3, 9.5e-3, 9.5e-3);
    assert_eq!(indices, Some((9, 9, 9)));
    
    // Test out of bounds
    let indices = grid.position_to_indices(11e-3, 5e-3, 5e-3);
    assert_eq!(indices, None);
    
    // Test negative
    let indices = grid.position_to_indices(-1e-3, 5e-3, 5e-3);
    assert_eq!(indices, None);
}