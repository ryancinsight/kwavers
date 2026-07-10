use super::AcousticSolverBackend;
use kwavers_core::constants::fundamental::ACOUSTIC_IMPEDANCE_WATER_NOMINAL;
use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::KwaversResult;
use kwavers_source::Source;
use leto::Array3;
use std::sync::Arc;

/// Mock backend for testing trait interface
#[derive(Debug)]
struct MockBackend {
    nx: usize,
    ny: usize,
    nz: usize,
    dt: f64,
    time: f64,
    pressure: Array3<f64>,
    vx: Array3<f64>,
    vy: Array3<f64>,
    vz: Array3<f64>,
}

impl MockBackend {
    fn new(nx: usize, ny: usize, nz: usize, dt: f64) -> Self {
        Self {
            nx,
            ny,
            nz,
            dt,
            time: 0.0,
            pressure: Array3::zeros((nx, ny, nz)),
            vx: Array3::zeros((nx, ny, nz)),
            vy: Array3::zeros((nx, ny, nz)),
            vz: Array3::zeros((nx, ny, nz)),
        }
    }
}

impl AcousticSolverBackend for MockBackend {
    fn step(&mut self) -> KwaversResult<()> {
        self.time += self.dt;
        Ok(())
    }

    fn get_pressure_field(&self) -> &Array3<f64> {
        &self.pressure
    }

    fn get_velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.vx, &self.vy, &self.vz)
    }

    fn get_intensity_field(&self) -> KwaversResult<Array3<f64>> {
        // I = p²/(ρc) with ρc = 1.5 MRayl for water
        let rho_c = ACOUSTIC_IMPEDANCE_WATER_NOMINAL;
        Ok(self.pressure.mapv(|p| p * p / rho_c))
    }

    fn get_impedance_field(&self) -> KwaversResult<Array3<f64>> {
        // Return constant impedance for mock backend (1.5 MRayl)
        Ok(Array3::from_elem(
            (self.nx, self.ny, self.nz),
            ACOUSTIC_IMPEDANCE_WATER_NOMINAL,
        ))
    }

    fn get_dt(&self) -> f64 {
        self.dt
    }

    fn add_source(&mut self, _source: Arc<dyn Source>) -> KwaversResult<()> {
        Ok(())
    }

    fn get_current_time(&self) -> f64 {
        self.time
    }

    fn get_grid_dimensions(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz)
    }
}

#[test]
fn test_backend_trait_basic_operations() {
    let mut backend = MockBackend::new(10, 10, 10, 1e-7);

    // Test initial state
    assert_eq!(backend.get_current_time(), 0.0);
    assert_eq!(backend.get_dt(), 1e-7);
    assert_eq!(backend.get_grid_dimensions(), (10, 10, 10));

    // Test stepping
    backend.step().unwrap();
    assert_eq!(backend.get_current_time(), 1e-7);

    backend.step().unwrap();
    assert_eq!(backend.get_current_time(), 2e-7);
}

#[test]
fn test_backend_field_access() {
    let backend = MockBackend::new(5, 5, 5, 1e-7);

    // Test field dimensions
    let p = backend.get_pressure_field();
    assert_eq!(p.shape(), [5, 5, 5]);

    let (vx, vy, vz) = backend.get_velocity_fields();
    assert_eq!(vx.shape(), [5, 5, 5]);
    assert_eq!(vy.shape(), [5, 5, 5]);
    assert_eq!(vz.shape(), [5, 5, 5]);
}

#[test]
fn test_backend_intensity_computation() {
    let mut backend = MockBackend::new(3, 3, 3, 1e-7);

    // Set non-zero pressure
    backend.pressure[[1, 1, 1]] = MPA_TO_PA; // 1 MPa

    // Compute intensity
    let intensity = backend.get_intensity_field().unwrap();

    // Expected: I = p²/(ρc) = (1 MPa)² / (1.5 MRayl) ≈ 666.7 kW/m²
    let expected = (MPA_TO_PA * MPA_TO_PA) / ACOUSTIC_IMPEDANCE_WATER_NOMINAL;
    assert!((intensity[[1, 1, 1]] - expected).abs() < 1.0);
}

#[test]
fn test_backend_as_trait_object() {
    // Verify trait object usage
    let backend: Box<dyn AcousticSolverBackend> = Box::new(MockBackend::new(8, 8, 8, 1e-7));

    assert_eq!(backend.get_grid_dimensions(), (8, 8, 8));
    assert_eq!(backend.get_dt(), 1e-7);
}
