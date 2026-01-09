use ndarray::Array3;

/// Photoacoustic simulation results
#[derive(Debug)]
pub struct PhotoacousticResult {
    /// Time-resolved pressure fields (Pa)
    pub pressure_fields: Vec<Array3<f64>>,
    /// Time vector (s)
    pub time: Vec<f64>,
    /// Final reconstructed image
    pub reconstructed_image: Array3<f64>,
    /// Signal-to-noise ratio
    pub snr: f64,
}

/// Photoacoustic simulation parameters
#[derive(Debug, Clone)]
pub struct PhotoacousticParameters {
    /// Optical wavelengths for multi-spectral imaging (nm)
    pub wavelengths: Vec<f64>,
    /// Optical absorption coefficients for each wavelength (m⁻¹)
    pub absorption_coefficients: Vec<f64>,
    /// Scattering coefficients (m⁻¹)
    pub scattering_coefficients: Vec<f64>,
    /// Anisotropy factors (g = <cosθ>)
    pub anisotropy_factors: Vec<f64>,
    /// Grüneisen parameters (thermoelastic efficiency)
    pub gruneisen_parameters: Vec<f64>,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Laser fluence (J/m²)
    pub laser_fluence: f64,
    /// Speed of sound used for time-reversal reconstruction (m/s)
    pub speed_of_sound: f64,
    /// Center frequency for phase correction in reconstruction (Hz)
    pub center_frequency: f64,
    /// Time step for simulation (s)
    pub dt: f64,
}

impl Default for PhotoacousticParameters {
    fn default() -> Self {
        Self {
            wavelengths: vec![532.0, 650.0, 750.0, 850.0], // Common PAI wavelengths
            absorption_coefficients: vec![10.0, 5.0, 2.0, 1.0], // Example values
            scattering_coefficients: vec![100.0, 80.0, 60.0, 40.0],
            anisotropy_factors: vec![0.9, 0.85, 0.8, 0.75],
            gruneisen_parameters: vec![0.12, 0.12, 0.12, 0.12], // Typical for soft tissue
            pulse_duration: 10e-9,                              // 10 ns pulses
            laser_fluence: 10.0,                                // 10 mJ/cm²
            speed_of_sound: 1500.0, // Match homogeneous medium used in tests
            center_frequency: 5e6,  // 5 MHz center frequency
            dt: 1e-9,               // 1 ns time step for photoacoustic simulation
        }
    }
}

/// Optical properties for tissue types
#[derive(Debug, Clone)]
pub struct OpticalProperties {
    /// Absorption coefficient (m⁻¹)
    pub absorption: f64,
    /// Scattering coefficient (m⁻¹)
    pub scattering: f64,
    /// Anisotropy factor
    pub anisotropy: f64,
    /// Refractive index
    pub refractive_index: f64,
}

impl OpticalProperties {
    /// Blood optical properties (wavelength-dependent)
    pub fn blood(wavelength: f64) -> Self {
        // Simplified wavelength dependence for hemoglobin
        let absorption = if wavelength < 600.0 {
            100.0 + (wavelength - 400.0) * 0.5 // Oxy-Hb peak ~400-600nm
        } else {
            50.0 + (wavelength - 600.0) * (-0.1) // Deoxy-Hb ~600-1000nm
        };

        Self {
            absorption,
            scattering: 150.0,
            anisotropy: 0.95,
            refractive_index: 1.4,
        }
    }

    /// Soft tissue optical properties
    pub fn soft_tissue(wavelength: f64) -> Self {
        Self {
            absorption: 0.1 + wavelength * 0.001, // Low absorption
            scattering: 100.0 + wavelength * 0.1, // High scattering
            anisotropy: 0.8,
            refractive_index: 1.4,
        }
    }

    /// Tumor tissue (enhanced absorption)
    pub fn tumor(wavelength: f64) -> Self {
        Self {
            absorption: 5.0 + wavelength * 0.01, // Higher absorption
            scattering: 120.0 + wavelength * 0.15,
            anisotropy: 0.85,
            refractive_index: 1.4,
        }
    }
}

/// Photoacoustic initial pressure distribution
#[derive(Debug)]
pub struct InitialPressure {
    /// Pressure field (Pa)
    pub pressure: Array3<f64>,
    /// Maximum pressure amplitude
    pub max_pressure: f64,
    /// Optical fluence distribution
    pub fluence: Array3<f64>,
}
