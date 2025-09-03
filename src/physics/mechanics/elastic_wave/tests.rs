// src/physics/mechanics/elastic_wave/tests.rs

#[cfg(test)]
mod tests {
    // Note: Removed unused import following YAGNI principle
    use crate::grid::Grid;
    use crate::medium::homogeneous::HomogeneousMedium;
    use crate::physics::field_mapping::UnifiedFieldType;
    use crate::physics::mechanics::elastic_wave::ElasticWave;
    use crate::physics::traits::AcousticWaveModel;
    use crate::source::NullSource;
    use ndarray::{Array3, Array4};

    #[test]
    fn test_elastic_wave_constructor() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let _elastic_wave = ElasticWave::new(&grid).unwrap();
        // Test that the constructor completes successfully
        assert!(true);
    }

    #[test]
    fn test_elastic_wave_single_step() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let mut elastic_wave = ElasticWave::new(&grid).unwrap();
        let medium = HomogeneousMedium::from_minimal(1000.0, 1500.0, &grid);
        let source = NullSource::new();

        let mut fields = Array4::<f64>::zeros((crate::solver::TOTAL_FIELDS, 32, 32, 32));

        // Set initial conditions
        fields[[UnifiedFieldType::VelocityX.index(), 16, 16, 16]] = 1.0;

        let prev_pressure = Array3::<f64>::zeros((32, 32, 32));

        // This should run without panicking
        elastic_wave.update_wave(
            &mut fields,
            &prev_pressure,
            &source,
            &grid,
            &medium,
            1e-6,
            0.0,
        );
        // Test passes if no panic occurs
        assert!(true);
    }
}
