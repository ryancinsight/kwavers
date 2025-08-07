//! Tests for multi-rate time integration

#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::grid::Grid;
    use ndarray::Array3;
    use std::collections::HashMap;
    use crate::solver::time_integration::time_stepper::AdamsBashforthConfig;
    use crate::solver::time_integration::adaptive_stepping::RichardsonErrorEstimator;
    
    /// Mock physics component for testing
    #[derive(Debug)]
    struct MockPhysics {
        wave_speed: f64,
        frequency: f64,
    }
    
    impl PhysicsComponent for MockPhysics {
        fn component_id(&self) -> &str {
            "mock_physics"
        }
        
        fn required_fields(&self) -> Vec<crate::physics::composable::FieldType> {
            vec![]
        }
        
        fn provided_fields(&self) -> Vec<crate::physics::composable::FieldType> {
            vec![crate::physics::composable::FieldType::Custom("test".to_string())]
        }
        
        fn update(
            &mut self,
            _fields: &mut ndarray::Array4<f64>,
            _grid: &Grid,
            _medium: &dyn crate::medium::Medium,
            _dt: f64,
            _t: f64,
        ) -> KwaversResult<()> {
            Ok(())
        }
        
        fn clone_component(&self) -> Box<dyn PhysicsComponent> {
            Box::new(MockPhysics {
                wave_speed: self.wave_speed,
                frequency: self.frequency,
            })
        }
        
        fn max_wave_speed(&self, _field: &Array3<f64>, _grid: &Grid) -> f64 {
            self.wave_speed
        }
        
        fn evaluate(&self, field: &Array3<f64>, _grid: &Grid) -> KwaversResult<Array3<f64>> {
            // Simple oscillatory behavior: d/dt u = -omega^2 * u
            let omega = 2.0 * std::f64::consts::PI * self.frequency;
            Ok(field.mapv(|v| -omega * omega * v))
        }
    }
    
    #[test]
    fn test_runge_kutta_4() {
        let config = time_stepper::RK4Config::default();
        let mut stepper = RungeKutta4::new(config);
        
        // Test on simple exponential decay
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let field = Array3::ones((8, 8, 8));
        let dt = 0.01;
        
        let rhs_fn = |f: &Array3<f64>| -> KwaversResult<Array3<f64>> {
            Ok(f.mapv(|v| -v)) // du/dt = -u
        };
        
        let result = stepper.step(&field, rhs_fn, dt, &grid).unwrap();
        
        // Check that values decreased (exponential decay)
        for &val in result.iter() {
            assert!(val < 1.0);
            assert!(val > 0.0);
        }
    }
    
    #[test]
    fn test_adams_bashforth() {
        let config = AdamsBashforthConfig {
            order: 2,
            startup_steps: 2,
        };
        let mut stepper = AdamsBashforth::new(config);
        
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let mut field = Array3::ones((4, 4, 4));
        let dt = 0.01;
        
        let rhs_fn = |f: &Array3<f64>| -> KwaversResult<Array3<f64>> {
            Ok(f.mapv(|v| -v))
        };
        
        // Take multiple steps to test multi-step behavior
        for _ in 0..5 {
            field = stepper.step(&field, rhs_fn, dt, &grid).unwrap();
        }
        
        // Check that values decreased
        for &val in field.iter() {
            assert!(val < 1.0);
            assert!(val > 0.0);
        }
    }
    
    #[test]
    fn test_adaptive_time_stepping() {
        let base_stepper = RungeKutta4::new(time_stepper::RK4Config::default());
        let low_order_stepper = RungeKutta4::new(time_stepper::RK4Config::default());
        let error_estimator = Box::new(RichardsonErrorEstimator::new(4));
        
        let mut adaptive = AdaptiveTimeStepper::new(
            base_stepper,
            low_order_stepper,
            error_estimator,
            0.01,    // initial dt
            1e-6,    // min dt
            1.0,     // max dt
            1e-4,    // tolerance
        );
        
        let grid = Grid::new(8, 8, 8, 1.0, 1.0, 1.0);
        let field = Array3::ones((8, 8, 8));
        
        let rhs_fn = |f: &Array3<f64>| -> KwaversResult<Array3<f64>> {
            Ok(f.mapv(|v| -v))
        };
        
        let (result, actual_dt) = adaptive.adaptive_step(&field, rhs_fn, &grid).unwrap();
        
        assert!(actual_dt > 0.0);
        assert!(result.iter().all(|&v| v < 1.0 && v > 0.0));
    }
    
    #[test]
    fn test_multi_rate_controller() {
        let config = MultiRateConfig::default();
        let mut controller = MultiRateController::new(config);
        
        let mut component_time_steps = HashMap::new();
        component_time_steps.insert("fast".to_string(), 0.001);
        component_time_steps.insert("medium".to_string(), 0.01);
        component_time_steps.insert("slow".to_string(), 0.1);
        
        let (global_dt, subcycles) = controller
            .determine_time_steps(&component_time_steps, 1.0)
            .unwrap();
        
        assert_eq!(global_dt, 0.001); // Should be the minimum
        assert_eq!(subcycles["fast"], 1);
        assert!(subcycles["medium"] >= 1);
        assert!(subcycles["slow"] >= 1);
    }
    
    #[test]
    fn test_stability_analyzer() {
        let analyzer = StabilityAnalyzer::new(0.9);
        let grid = Grid::new(10, 10, 10, 0.1, 0.1, 0.1);
        
        let physics = MockPhysics {
            wave_speed: 1.0,
            frequency: 1.0,
        };
        
        let field = Array3::ones((10, 10, 10));
        let max_dt = analyzer.compute_stable_dt(&physics, &field, &grid).unwrap();
        
        assert!(max_dt > 0.0);
        assert!(max_dt < 0.1); // Should be limited by CFL
        
        // Check stability
        assert!(analyzer.is_stable(max_dt * 0.9, 1.0, &grid));
        assert!(!analyzer.is_stable(max_dt * 1.5, 1.0, &grid));
    }
    
    #[test]
    fn test_subcycling_strategy() {
        let strategy = SubcyclingStrategy::new(10);
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        
        let mut fields = HashMap::new();
        fields.insert("fast".to_string(), Array3::ones((4, 4, 4)));
        fields.insert("slow".to_string(), Array3::ones((4, 4, 4)) * 2.0);
        
        let mut physics_components: HashMap<String, Box<dyn PhysicsComponent>> = HashMap::new();
        physics_components.insert(
            "fast".to_string(),
            Box::new(MockPhysics { wave_speed: 10.0, frequency: 10.0 })
        );
        physics_components.insert(
            "slow".to_string(),
            Box::new(MockPhysics { wave_speed: 1.0, frequency: 1.0 })
        );
        
        let mut subcycles = HashMap::new();
        subcycles.insert("fast".to_string(), 10);
        subcycles.insert("slow".to_string(), 1);
        
        strategy.advance_coupled_system(
            &mut fields,
            &physics_components,
            &subcycles,
            0.01,
            &grid,
        ).unwrap();
        
        // Check that fields were updated
        assert!(fields["fast"].iter().any(|&v| v != 1.0));
        assert!(fields["slow"].iter().any(|&v| v != 2.0));
    }
    
    #[test]
    fn test_multi_rate_integrator() {
        let mut config = MultiRateConfig::default();
        config.max_subcycles = 5;
        
        let grid = Grid::new(4, 4, 4, 1.0, 1.0, 1.0);
        let mut integrator = MultiRateTimeIntegrator::new(config, &grid);
        
        let mut fields = HashMap::new();
        fields.insert("component1".to_string(), Array3::ones((4, 4, 4)));
        
        let mut physics_components: HashMap<String, Box<dyn PhysicsComponent>> = HashMap::new();
        physics_components.insert(
            "component1".to_string(),
            Box::new(MockPhysics { wave_speed: 1.0, frequency: 1.0 })
        );
        
        let final_time = integrator.advance(
            &mut fields,
            &physics_components,
            0.0,
            0.1,
            &grid,
        ).unwrap();
        
        assert_eq!(final_time, 0.1);
        
        let stats = integrator.get_statistics();
        assert!(stats.total_steps > 0);
    }
}