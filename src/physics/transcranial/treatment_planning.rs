//! Treatment Planning for Transcranial Focused Ultrasound
//!
//! Patient-specific treatment planning using CT scans for skull characterization
//! and optimal trajectory calculation for brain targets.

use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;
use num_complex::Complex;

/// Treatment target volume in brain coordinates
#[derive(Debug, Clone)]
pub struct TargetVolume {
    /// Target center (x, y, z) in mm relative to brain origin
    pub center: [f64; 3],
    /// Target dimensions (width, height, depth) in mm
    pub dimensions: [f64; 3],
    /// Target shape (ellipsoidal, rectangular, etc.)
    pub shape: TargetShape,
    /// Clinical priority (1-10, higher = more critical)
    pub priority: u8,
    /// Maximum allowable temperature (°C)
    pub max_temperature: f64,
    /// Required acoustic intensity (W/cm²)
    pub required_intensity: f64,
}

#[derive(Debug, Clone)]
pub enum TargetShape {
    /// Ellipsoidal target
    Ellipsoidal,
    /// Rectangular target
    Rectangular,
    /// Custom shape defined by mask
    Custom(Array3<bool>),
}

/// Complete treatment plan for tFUS session
#[derive(Debug)]
pub struct TreatmentPlan {
    /// Patient identifier
    pub patient_id: String,
    /// Treatment targets
    pub targets: Vec<TargetVolume>,
    /// Skull CT data (Hounsfield units)
    pub skull_ct: Array3<f64>,
    /// Optimal transducer positions and phases
    pub transducer_setup: TransducerSetup,
    /// Predicted acoustic field in brain
    pub acoustic_field: Array3<f64>,
    /// Predicted temperature field
    pub temperature_field: Array3<f64>,
    /// Safety margins and constraints
    pub safety_constraints: SafetyConstraints,
    /// Estimated treatment time (seconds)
    pub treatment_time: f64,
}

/// Transducer array configuration
#[derive(Debug, Clone)]
pub struct TransducerSetup {
    /// Number of transducer elements
    pub num_elements: usize,
    /// Element positions (x, y, z) in mm
    pub element_positions: Vec<[f64; 3]>,
    /// Element phases for aberration correction (radians)
    pub element_phases: Vec<f64>,
    /// Element amplitudes (normalized)
    pub element_amplitudes: Vec<f64>,
    /// Operating frequency (Hz)
    pub frequency: f64,
    /// Focal distance (mm)
    pub focal_distance: f64,
}

/// Safety constraints for treatment
#[derive(Debug, Clone)]
pub struct SafetyConstraints {
    /// Maximum skull surface temperature (°C)
    pub max_skull_temp: f64,
    /// Maximum brain temperature (°C)
    pub max_brain_temp: f64,
    /// Maximum mechanical index
    pub max_mi: f64,
    /// Maximum thermal dose (CEM43)
    pub max_thermal_dose: f64,
    /// Minimum distance from skull-brain interface (mm)
    pub min_skull_distance: f64,
}

impl Default for SafetyConstraints {
    fn default() -> Self {
        Self {
            max_skull_temp: 42.0,    // °C
            max_brain_temp: 43.0,    // °C
            max_mi: 1.9,             // Mechanical index limit
            max_thermal_dose: 240.0, // CEM43 for brain tissue
            min_skull_distance: 5.0, // mm
        }
    }
}

/// Treatment planner for tFUS procedures
#[derive(Debug)]
pub struct TreatmentPlanner {
    /// Computational grid for brain volume
    brain_grid: Grid,
    /// Skull model from CT data
    skull_model: crate::physics::skull::CTBasedSkullModel,
    /// Aberration correction calculator
    _aberration_corrector:
        crate::physics::transcranial::aberration_correction::TranscranialAberrationCorrection,
}

impl TreatmentPlanner {
    /// Create new treatment planner
    pub fn new(brain_grid: &Grid, skull_ct_data: &Array3<f64>) -> KwaversResult<Self> {
        let skull_model = crate::physics::skull::CTBasedSkullModel::from_ct_data(skull_ct_data)?;
        let aberration_corrector = crate::physics::transcranial::aberration_correction::TranscranialAberrationCorrection::new(brain_grid)?;

        Ok(Self {
            brain_grid: brain_grid.clone(),
            skull_model,
            _aberration_corrector: aberration_corrector,
        })
    }

    /// Generate treatment plan for target volumes
    pub fn generate_plan(
        &self,
        patient_id: &str,
        targets: &[TargetVolume],
        transducer_spec: &TransducerSpecification,
    ) -> KwaversResult<TreatmentPlan> {
        println!("Generating tFUS treatment plan for patient: {}", patient_id);
        println!("Planning for {} target volumes", targets.len());

        // Step 1: Analyze skull properties
        let _skull_properties = self.analyze_skull_properties()?;

        // Step 2: Calculate optimal transducer configuration
        let transducer_setup = self.optimize_transducer_setup(targets, transducer_spec)?;

        // Step 3: Simulate acoustic field through skull
        let acoustic_field = self.simulate_acoustic_field(&transducer_setup)?;

        // Step 4: Calculate thermal response
        let temperature_field = self.calculate_thermal_response(&acoustic_field)?;

        // Step 5: Validate safety constraints
        self.validate_safety(
            &temperature_field,
            &acoustic_field,
            transducer_spec.frequency,
        )?;

        // Step 6: Estimate treatment time
        let treatment_time = self.estimate_treatment_time(targets, &acoustic_field);

        Ok(TreatmentPlan {
            patient_id: patient_id.to_string(),
            targets: targets.to_vec(),
            skull_ct: self.skull_model.ct_data().clone(),
            transducer_setup,
            acoustic_field,
            temperature_field,
            safety_constraints: SafetyConstraints::default(),
            treatment_time,
        })
    }

    /// Analyze skull acoustic properties from CT data
    fn analyze_skull_properties(&self) -> KwaversResult<SkullProperties> {
        let (nx, ny, nz) = self.skull_model.ct_data().dim();

        // Convert Hounsfield units to acoustic properties
        let mut speed_map = Array3::zeros((nx, ny, nz));
        let mut density_map = Array3::zeros((nx, ny, nz));
        let mut attenuation_map = Array3::zeros((nx, ny, nz));

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let hu = self.skull_model.ct_data()[[i, j, k]];

                    // Empirical relationships from literature
                    // Reference: Pinton et al. (2012)
                    if hu > 300.0 {
                        // Bone threshold
                        speed_map[[i, j, k]] = 3000.0 + (hu - 300.0) * 2.0; // m/s
                        density_map[[i, j, k]] = 1800.0 + (hu - 300.0) * 0.5; // kg/m³
                        attenuation_map[[i, j, k]] = 5.0 + (hu - 300.0) * 0.01; // dB/MHz/cm
                    } else if hu > -200.0 {
                        // Tissue
                        speed_map[[i, j, k]] = 1500.0;
                        density_map[[i, j, k]] = 1000.0;
                        attenuation_map[[i, j, k]] = 0.5;
                    } else {
                        // Air
                        speed_map[[i, j, k]] = 340.0;
                        density_map[[i, j, k]] = 1.2;
                        attenuation_map[[i, j, k]] = 0.0;
                    }
                }
            }
        }

        Ok(SkullProperties {
            sound_speed: speed_map,
            density: density_map,
            attenuation: attenuation_map,
        })
    }

    /// Optimize transducer setup for targets
    fn optimize_transducer_setup(
        &self,
        targets: &[TargetVolume],
        spec: &TransducerSpecification,
    ) -> KwaversResult<TransducerSetup> {
        // Simplified optimization - place transducer elements in hemisphere
        let num_elements = spec.num_elements;
        let mut element_positions = Vec::with_capacity(num_elements);
        let mut element_phases = vec![0.0; num_elements];
        let element_amplitudes = vec![1.0; num_elements];

        // Arrange elements in hemispherical array
        let radius = spec.radius;
        let center = [0.0, 0.0, radius]; // Above skull

        for i in 0..num_elements {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / num_elements as f64;
            let phi = std::f64::consts::PI / 4.0; // 45 degrees from vertical

            let x = center[0] + radius * phi.sin() * theta.cos();
            let y = center[1] + radius * phi.sin() * theta.sin();
            let z = center[2] + radius * phi.cos();

            element_positions.push([x, y, z]);
        }

        // Calculate aberration correction phases
        for (i, &pos) in element_positions.iter().enumerate() {
            // Simplified phase calculation - would need full wave propagation
            let target_center = targets[0].center; // Focus on first target
            let distance = ((pos[0] - target_center[0]).powi(2)
                + (pos[1] - target_center[1]).powi(2)
                + (pos[2] - target_center[2]).powi(2))
            .sqrt();

            let k = 2.0 * std::f64::consts::PI * spec.frequency / spec.sound_speed;
            element_phases[i] = -k * distance; // Phase conjugation
        }

        Ok(TransducerSetup {
            num_elements,
            element_positions,
            element_phases,
            element_amplitudes,
            frequency: spec.frequency,
            focal_distance: spec.focal_distance,
        })
    }

    /// Simulate acoustic field through skull
    fn simulate_acoustic_field(&self, setup: &TransducerSetup) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = self.brain_grid.dimensions();
        let mut acoustic_field = Array3::zeros((nx, ny, nz));

        // Simplified field calculation - would need full wave propagation
        // Compute acoustic field for transcranial therapy planning
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let x = i as f64 * self.brain_grid.dx;
                    let y = j as f64 * self.brain_grid.dy;
                    let z = k as f64 * self.brain_grid.dz;

                    // Calculate field contribution from each element
                    let mut total_field: f64 = 0.0;

                    for element in &setup.element_positions {
                        let dx = x - element[0];
                        let dy = y - element[1];
                        let dz = z - element[2];
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                        if distance > 0.0 {
                            let k = 2.0 * std::f64::consts::PI * setup.frequency / 1500.0;
                            let phase = k * distance;
                            let complex_phase = Complex::new(0.0, phase);
                            total_field += complex_phase.exp().norm_sqr().sqrt() / distance;
                        }
                    }

                    acoustic_field[[i, j, k]] = total_field * total_field; // Intensity
                }
            }
        }

        Ok(acoustic_field)
    }

    /// Calculate thermal response to acoustic field
    fn calculate_thermal_response(
        &self,
        acoustic_field: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = acoustic_field.dim();
        let mut temperature_field = Array3::zeros((nx, ny, nz));

        // Simplified bioheat equation solution
        // T = T0 + (α * I * t) / (ρ * c)
        // where α = absorption coefficient, I = intensity, t = time

        let absorption_coeff = 0.5; // dB/MHz/cm converted to appropriate units
        let perfusion_rate = 0.01; // 1/s (brain perfusion)
        let specific_heat = 3600.0; // J/kg/K
        let density = 1000.0; // kg/m³
        let exposure_time = 10.0; // seconds

        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let intensity = acoustic_field[[i, j, k]];
                    let absorbed_power = absorption_coeff * intensity;

                    // Steady-state temperature rise calculation
                    let temp_rise = absorbed_power * exposure_time
                        / (density * specific_heat * (perfusion_rate + absorption_coeff));

                    temperature_field[[i, j, k]] = 37.0 + temp_rise; // Body temp + rise
                }
            }
        }

        Ok(temperature_field)
    }

    /// Validate safety constraints
    fn validate_safety(
        &self,
        temperature: &Array3<f64>,
        acoustic_field: &Array3<f64>,
        frequency_hz: f64,
    ) -> KwaversResult<()> {
        let constraints = SafetyConstraints::default();

        // Check temperature limits
        for &temp in temperature.iter() {
            if temp > constraints.max_brain_temp {
                return Err(crate::error::KwaversError::Validation(
                    crate::error::ValidationError::ConstraintViolation {
                        message: format!(
                            "Brain temperature {:.1}°C exceeds limit {:.1}°C",
                            temp, constraints.max_brain_temp
                        ),
                    },
                ));
            }
        }

        // Check mechanical index using MI = P_neg(MPa) / sqrt(frequency_MHz)
        // Convert intensity to pressure using p = sqrt(I * rho * c)
        let max_intensity = acoustic_field.iter().fold(0.0_f64, |a, &b| a.max(b));
        let rho = 1000.0; // kg/m^3
        let c = 1500.0; // m/s
        let p_pa = (max_intensity * rho * c).sqrt();
        let p_mpa = p_pa / 1_000_000.0; // Pa -> MPa
        let freq_mhz = frequency_hz / 1_000_000.0;
        let mi = if freq_mhz > 0.0 {
            p_mpa / freq_mhz.sqrt()
        } else {
            f64::INFINITY
        };

        if mi > constraints.max_mi {
            return Err(crate::error::KwaversError::Validation(
                crate::error::ValidationError::ConstraintViolation {
                    message: format!(
                        "Mechanical index {:.2} exceeds limit {:.2}",
                        mi, constraints.max_mi
                    ),
                },
            ));
        }

        Ok(())
    }

    /// Estimate treatment time
    fn estimate_treatment_time(
        &self,
        _targets: &[TargetVolume],
        acoustic_field: &Array3<f64>,
    ) -> f64 {
        // Estimate based on required thermal dose
        let thermal_dose_target = 240.0; // CEM43
        let max_intensity = acoustic_field.iter().fold(0.0_f64, |a, &b| a.max(b));

        if max_intensity > 0.0 {
            // Simplified: t = thermal_dose / (absorption_rate * intensity)
            thermal_dose_target / (0.5 * max_intensity)
        } else {
            0.0
        }
    }
}

/// Skull acoustic properties derived from CT
#[derive(Debug)]
pub struct SkullProperties {
    pub sound_speed: Array3<f64>,
    pub density: Array3<f64>,
    pub attenuation: Array3<f64>,
}

/// Transducer array specifications
#[derive(Debug, Clone)]
pub struct TransducerSpecification {
    pub num_elements: usize,
    pub frequency: f64,
    pub focal_distance: f64,
    pub radius: f64,
    pub sound_speed: f64,
}

impl Default for TransducerSpecification {
    fn default() -> Self {
        Self {
            num_elements: 1024,
            frequency: 650e3,      // 650 kHz for brain therapy
            focal_distance: 120.0, // mm
            radius: 80.0,          // mm
            sound_speed: 1500.0,   // m/s
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_treatment_planner_creation() {
        let grid = Grid::new(64, 64, 64, 0.001, 0.001, 0.001).unwrap();
        let ct_data = Array3::from_elem((64, 64, 64), 0.0); // Air

        let planner = TreatmentPlanner::new(&grid, &ct_data);
        assert!(planner.is_ok());
    }

    #[test]
    fn test_treatment_plan_generation() {
        let grid = Grid::new(32, 32, 32, 0.002, 0.002, 0.002).unwrap();
        let ct_data = Array3::from_elem((32, 32, 32), 400.0); // Bone-like HU
        let planner = TreatmentPlanner::new(&grid, &ct_data).unwrap();

        let target = TargetVolume {
            center: [0.016, 0.016, 0.016],
            dimensions: [0.004, 0.004, 0.004],
            shape: TargetShape::Ellipsoidal,
            priority: 8,
            max_temperature: 45.0,
            required_intensity: 100.0,
        };

        let spec = TransducerSpecification::default();
        let plan = planner.generate_plan("test_patient", &[target], &spec);

        assert!(plan.is_ok());
    }

    #[test]
    fn test_skull_properties_analysis() {
        let grid = Grid::new(16, 16, 16, 0.005, 0.005, 0.005).unwrap();
        let ct_data = Array3::from_elem((16, 16, 16), 800.0); // Dense bone
        let planner = TreatmentPlanner::new(&grid, &ct_data).unwrap();

        let properties = planner.analyze_skull_properties();
        assert!(properties.is_ok());

        let props = properties.unwrap();
        assert_eq!(props.sound_speed.dim(), (16, 16, 16));
        assert_eq!(props.density.dim(), (16, 16, 16));
        assert_eq!(props.attenuation.dim(), (16, 16, 16));
    }
}
