//! Minimal unit tests for SRS NFR-002 compliance (<30s execution)
//! 
//! These tests focus on core functionality without expensive computations
//! to ensure CI/CD pipeline efficiency and production deployment readiness.

#[cfg(test)]
mod minimal_tests {
    use crate::grid::Grid;
    use crate::medium::HomogeneousMedium;
    use crate::physics::constants::{DENSITY_WATER, SOUND_SPEED_WATER};

    #[test]
    fn test_grid_creation_minimal() {
        let grid = Grid::new(10, 10, 10, 0.001, 0.001, 0.001).expect("Grid creation");
        assert_eq!(grid.nx, 10);
        assert_eq!(grid.ny, 10);
        assert_eq!(grid.nz, 10);
        assert_eq!(grid.size(), 1000);
    }

    #[test]
    fn test_medium_basic_properties() {
        let grid = Grid::new(5, 5, 5, 0.001, 0.001, 0.001).expect("Grid creation");
        let medium = HomogeneousMedium::new(DENSITY_WATER, SOUND_SPEED_WATER, 0.0, 0.0, &grid);
        
        assert!(medium.is_homogeneous());
        assert!((medium.sound_speed(0, 0, 0) - SOUND_SPEED_WATER).abs() < 1e-6);
        assert!((medium.density(0, 0, 0) - DENSITY_WATER).abs() < 1e-6);
    }

    #[test]
    fn test_physics_constants_validation() {
        // Basic sanity checks for physics constants
        assert!(DENSITY_WATER > 0.0);
        assert!(SOUND_SPEED_WATER > 0.0);
        assert!(DENSITY_WATER > 900.0 && DENSITY_WATER < 1100.0); // Water density range
        assert!(SOUND_SPEED_WATER > 1400.0 && SOUND_SPEED_WATER < 1600.0); // Water sound speed range
    }

    #[test]
    fn test_grid_spacing_consistency() {
        let grid = Grid::new(20, 30, 40, 0.001, 0.002, 0.003).expect("Grid creation");
        assert!((grid.dx - 0.001).abs() < 1e-10);
        assert!((grid.dy - 0.002).abs() < 1e-10);
        assert!((grid.dz - 0.003).abs() < 1e-10);
    }

    #[test]
    fn test_cfl_calculation_basic() {
        let grid = Grid::new(16, 16, 16, 1e-3, 1e-3, 1e-3).expect("Grid creation");
        let sound_speed = 1500.0;
        let cfl = 0.5;
        let min_dx = grid.dx.min(grid.dy).min(grid.dz);
        let dt = cfl * min_dx / sound_speed;
        
        assert!(dt > 0.0);
        assert!(dt < 1e-6); // Reasonable timestep for acoustics
        
        // CFL stability condition: c*dt/dx <= CFL_max
        let actual_cfl = sound_speed * dt / min_dx;
        assert!((actual_cfl - cfl).abs() < 1e-10);
    }
}