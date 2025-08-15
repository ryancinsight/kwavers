//! Transducer Design Module
//! 
//! Comprehensive transducer design parameters including element size, kerf,
//! bandwidth, impedance matching, and coupling considerations.
//! 
//! References:
//! - Szabo (2014): "Diagnostic Ultrasound Imaging: Inside Out"
//! - Shung (2015): "Diagnostic Ultrasound: Imaging and Blood Flow Measurements"
//! - Cobbold (2007): "Foundations of Biomedical Ultrasound"
//! - Kino (1987): "Acoustic Waves: Devices, Imaging, and Analog Signal Processing"
//! - Hunt et al. (1983): "Ultrasound transducers for pulse-echo medical imaging"

use crate::error::{KwaversError, KwaversResult, ConfigError};
use ndarray::{Array1, Array2};
use num_complex::Complex64;
use std::f64::consts::PI;

// Transducer design constants based on literature
/// Typical piezoelectric coupling coefficient (PZT-5H)
const PIEZO_COUPLING_K33: f64 = 0.75;

/// Typical mechanical quality factor
const MECHANICAL_Q: f64 = 80.0;

/// Typical electrical quality factor  
const ELECTRICAL_Q: f64 = 50.0;

/// Standard acoustic impedance of PZT (MRayl)
const PZT_IMPEDANCE: f64 = 30.0;

/// Acoustic impedance of water/tissue (MRayl)
const TISSUE_IMPEDANCE: f64 = 1.5;

/// Acoustic impedance of backing material (MRayl)
const BACKING_IMPEDANCE: f64 = 5.0;

/// Minimum kerf width as fraction of element width
const MIN_KERF_RATIO: f64 = 0.05;

/// Maximum kerf width as fraction of element width
const MAX_KERF_RATIO: f64 = 0.3;

/// Typical matching layer thickness (quarter wavelength)
const MATCHING_LAYER_FACTOR: f64 = 0.25;

/// Bandwidth threshold (-6 dB) for fractional bandwidth calculation
const BANDWIDTH_THRESHOLD_DB: f64 = -6.0;

/// Minimum element aspect ratio (width/thickness)
const MIN_ASPECT_RATIO: f64 = 0.5;

/// Maximum element aspect ratio to avoid lateral modes
const MAX_ASPECT_RATIO: f64 = 20.0;

/// Lateral mode suppression factor
const LATERAL_MODE_FACTOR: f64 = 0.3;

/// Cross-coupling coefficient between adjacent elements
const CROSS_COUPLING_COEFFICIENT: f64 = 0.05;

/// Element geometry and dimensions
#[derive(Debug, Clone)]
pub struct ElementGeometry {
    /// Element width [m]
    pub width: f64,
    /// Element height [m]
    pub height: f64,
    /// Element thickness [m]
    pub thickness: f64,
    /// Kerf width between elements [m]
    pub kerf: f64,
    /// Element pitch (width + kerf) [m]
    pub pitch: f64,
    /// Aspect ratio (width/thickness)
    pub aspect_ratio: f64,
    /// Element area [m²]
    pub area: f64,
    /// Element volume [m³]
    pub volume: f64,
}

impl ElementGeometry {
    /// Create new element geometry with validation
    pub fn new(width: f64, height: f64, thickness: f64, kerf: f64) -> KwaversResult<Self> {
        // Validate dimensions
        if width <= 0.0 || height <= 0.0 || thickness <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "dimensions".to_string(),
                value: format!("{}x{}x{}", width, height, thickness),
                constraint: "Must be positive".to_string(),
            }));
        }
        
        if kerf < 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf".to_string(),
                value: kerf.to_string(),
                constraint: "Cannot be negative".to_string(),
            }));
        }
        
        // Check kerf ratio
        let kerf_ratio = kerf / width;
        if kerf_ratio < MIN_KERF_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf_ratio".to_string(),
                value: format!("{:.3}", kerf_ratio),
                constraint: format!("Must be >= {:.3}", MIN_KERF_RATIO),
            }));
        }
        
        if kerf_ratio > MAX_KERF_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "kerf_ratio".to_string(),
                value: format!("{:.3}", kerf_ratio),
                constraint: format!("must be <= {:.3}", MAX_KERF_RATIO),
            }));
        }
        
        let aspect_ratio = width / thickness;
        
        // Check aspect ratio for lateral mode suppression
        if aspect_ratio < MIN_ASPECT_RATIO || aspect_ratio > MAX_ASPECT_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "aspect_ratio".to_string(),
                value: format!("{:.1}", aspect_ratio),
                constraint: format!("must be in range [{:.1}, {:.1}]", MIN_ASPECT_RATIO, MAX_ASPECT_RATIO),
            }));
        }
        
        Ok(Self {
            width,
            height,
            thickness,
            kerf,
            pitch: width + kerf,
            aspect_ratio,
            area: width * height,
            volume: width * height * thickness,
        })
    }
    
    /// Calculate element capacitance (Farads)
    pub fn calculate_capacitance(&self, permittivity: f64) -> f64 {
        // C = ε₀ * εᵣ * A / t
        const EPSILON_0: f64 = 8.854e-12; // F/m
        EPSILON_0 * permittivity * self.area / self.thickness
    }
    
    /// Calculate lateral mode frequency
    pub fn calculate_lateral_mode_frequency(&self, sound_speed: f64) -> f64 {
        // Lateral resonance frequency
        sound_speed / (2.0 * self.width)
    }
    
    /// Calculate thickness mode frequency (fundamental)
    pub fn calculate_thickness_mode_frequency(&self, sound_speed: f64) -> f64 {
        // Thickness resonance frequency
        sound_speed / (2.0 * self.thickness)
    }
}

/// Frequency response and bandwidth characteristics
#[derive(Debug, Clone)]
pub struct FrequencyResponse {
    /// Center frequency [Hz]
    pub center_frequency: f64,
    /// Lower -6dB frequency [Hz]
    pub lower_frequency: f64,
    /// Upper -6dB frequency [Hz]
    pub upper_frequency: f64,
    /// Fractional bandwidth (%)
    pub fractional_bandwidth: f64,
    /// Quality factor Q
    pub quality_factor: f64,
    /// Frequency samples [Hz]
    pub frequencies: Array1<f64>,
    /// Magnitude response [dB]
    pub magnitude: Array1<f64>,
    /// Phase response [rad]
    pub phase: Array1<f64>,
}

impl FrequencyResponse {
    /// Calculate from transducer parameters using KLM model
    pub fn from_klm_model(
        geometry: &ElementGeometry,
        material: &PiezoMaterial,
        backing: &BackingLayer,
        matching: &MatchingLayer,
        num_points: usize,
    ) -> KwaversResult<Self> {
        let fc = material.sound_speed / (2.0 * geometry.thickness);
        
        // Frequency range (0.1fc to 2fc)
        let frequencies = Array1::linspace(0.1 * fc, 2.0 * fc, num_points);
        let mut magnitude = Array1::zeros(num_points);
        let mut phase = Array1::zeros(num_points);
        
        // Calculate response at each frequency
        for (i, &f) in frequencies.iter().enumerate() {
            let response = Self::calculate_klm_response(
                f, fc, geometry, material, backing, matching
            );
            magnitude[i] = 20.0 * response.norm().log10();
            phase[i] = response.arg();
        }
        
        // Find -6dB points
        let max_mag = magnitude.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let threshold = max_mag + BANDWIDTH_THRESHOLD_DB;
        
        let mut lower_idx = 0;
        let mut upper_idx = num_points - 1;
        
        for (i, &mag) in magnitude.iter().enumerate() {
            if mag >= threshold {
                lower_idx = i;
                break;
            }
        }
        
        for (i, &mag) in magnitude.iter().rev().enumerate() {
            if mag >= threshold {
                upper_idx = num_points - 1 - i;
                break;
            }
        }
        
        let lower_frequency = frequencies[lower_idx];
        let upper_frequency = frequencies[upper_idx];
        let bandwidth = upper_frequency - lower_frequency;
        let center_frequency = (lower_frequency + upper_frequency) / 2.0;
        let fractional_bandwidth = 100.0 * bandwidth / center_frequency;
        let quality_factor = center_frequency / bandwidth;
        
        Ok(Self {
            center_frequency,
            lower_frequency,
            upper_frequency,
            fractional_bandwidth,
            quality_factor,
            frequencies,
            magnitude,
            phase,
        })
    }
    
    /// Calculate KLM model response at a frequency
    fn calculate_klm_response(
        f: f64,
        fc: f64,
        geometry: &ElementGeometry,
        material: &PiezoMaterial,
        backing: &BackingLayer,
        matching: &MatchingLayer,
    ) -> Complex64 {
        let omega = 2.0 * PI * f;
        let k = omega / material.sound_speed;
        
        // Acoustic impedances
        let z_piezo = material.acoustic_impedance;
        let z_back = backing.acoustic_impedance;
        let z_match = matching.acoustic_impedance;
        let z_load = TISSUE_IMPEDANCE;
        
        // Transmission line model
        let gamma = Complex64::new(0.0, k * geometry.thickness);
        
        // Input impedance at backing interface
        let z_in_back = z_back;
        
        // Transform through piezo layer
        let z_in_front = z_piezo * (z_in_back + z_piezo * gamma.tanh()) /
                        (z_piezo + z_in_back * gamma.tanh());
        
        // Through matching layer
        let gamma_match = Complex64::new(0.0, k * matching.thickness);
        let z_in_match = z_match * (z_load + z_match * gamma_match.tanh()) /
                        (z_match + z_load * gamma_match.tanh());
        
        // Total transfer function
        let h = 2.0 * z_load / (z_in_front + z_in_match);
        
        // Add resonance effects
        let resonance = 1.0 / Complex64::new(
            1.0 - (f / fc).powi(2),
            f / (fc * material.mechanical_q)
        );
        
        h * resonance
    }
}

/// Piezoelectric material properties
#[derive(Debug, Clone)]
pub struct PiezoMaterial {
    /// Material type (e.g., PZT-5H, PZT-4)
    pub material_type: PiezoType,
    /// Density [kg/m³]
    pub density: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
    /// Acoustic impedance [MRayl]
    pub acoustic_impedance: f64,
    /// Relative permittivity
    pub relative_permittivity: f64,
    /// Piezoelectric coupling coefficient k33
    pub coupling_k33: f64,
    /// Piezoelectric coupling coefficient kt
    pub coupling_kt: f64,
    /// Mechanical quality factor
    pub mechanical_q: f64,
    /// Electrical quality factor
    pub electrical_q: f64,
    /// Dielectric loss tangent
    pub loss_tangent: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PiezoType {
    PZT5H,  // High sensitivity
    PZT4,   // High power
    PZT8,   // High temperature
    PVDF,   // Polymer
    PMN_PT, // Single crystal
    LiNbO3, // Lithium niobate
}

impl PiezoMaterial {
    /// Create standard PZT-5H material
    pub fn pzt_5h() -> Self {
        Self {
            material_type: PiezoType::PZT5H,
            density: 7500.0,
            sound_speed: 4600.0,
            acoustic_impedance: 34.5,
            relative_permittivity: 3400.0,
            coupling_k33: 0.75,
            coupling_kt: 0.50,
            mechanical_q: 65.0,
            electrical_q: 50.0,
            loss_tangent: 0.02,
        }
    }
    
    /// Create standard PZT-4 material
    pub fn pzt_4() -> Self {
        Self {
            material_type: PiezoType::PZT4,
            density: 7500.0,
            sound_speed: 4600.0,
            acoustic_impedance: 34.5,
            relative_permittivity: 1300.0,
            coupling_k33: 0.70,
            coupling_kt: 0.47,
            mechanical_q: 500.0,
            electrical_q: 100.0,
            loss_tangent: 0.004,
        }
    }
    
    /// Create PVDF polymer
    pub fn pvdf() -> Self {
        Self {
            material_type: PiezoType::PVDF,
            density: 1780.0,
            sound_speed: 2200.0,
            acoustic_impedance: 3.9,
            relative_permittivity: 12.0,
            coupling_k33: 0.15,
            coupling_kt: 0.12,
            mechanical_q: 10.0,
            electrical_q: 5.0,
            loss_tangent: 0.18,
        }
    }
    
    /// Create PMN-PT single crystal
    pub fn pmn_pt() -> Self {
        Self {
            material_type: PiezoType::PMN_PT,
            density: 8100.0,
            sound_speed: 4600.0,
            acoustic_impedance: 37.3,
            relative_permittivity: 5000.0,
            coupling_k33: 0.90,
            coupling_kt: 0.58,
            mechanical_q: 100.0,
            electrical_q: 80.0,
            loss_tangent: 0.01,
        }
    }
}

/// Backing layer design
#[derive(Debug, Clone)]
pub struct BackingLayer {
    /// Backing material type
    pub material: BackingMaterial,
    /// Thickness [m]
    pub thickness: f64,
    /// Acoustic impedance [MRayl]
    pub acoustic_impedance: f64,
    /// Attenuation [dB/cm/MHz]
    pub attenuation: f64,
    /// Scattering particles included
    pub has_scatterers: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackingMaterial {
    Epoxy,           // Low impedance
    TungstenEpoxy,   // High impedance
    Air,             // Air backing
    Composite,       // Composite backing
}

impl BackingLayer {
    /// Create tungsten-loaded epoxy backing
    pub fn tungsten_epoxy(thickness: f64) -> Self {
        Self {
            material: BackingMaterial::TungstenEpoxy,
            thickness,
            acoustic_impedance: 10.0,
            attenuation: 50.0,
            has_scatterers: true,
        }
    }
    
    /// Create air backing for high Q
    pub fn air_backed() -> Self {
        Self {
            material: BackingMaterial::Air,
            thickness: 0.0,
            acoustic_impedance: 0.0004,
            attenuation: 0.0,
            has_scatterers: false,
        }
    }
}

/// Matching layer design
#[derive(Debug, Clone)]
pub struct MatchingLayer {
    /// Number of matching layers
    pub num_layers: usize,
    /// Layer thicknesses [m]
    pub thickness: f64,
    /// Acoustic impedance [MRayl]
    pub acoustic_impedance: f64,
    /// Sound speed [m/s]
    pub sound_speed: f64,
}

impl MatchingLayer {
    /// Design quarter-wave matching layer
    pub fn quarter_wave(frequency: f64, z_piezo: f64, z_load: f64) -> Self {
        // Optimal impedance for single layer: Z = sqrt(Z_piezo * Z_load)
        let acoustic_impedance = (z_piezo * z_load).sqrt();
        
        // Typical matching layer sound speed
        let sound_speed = 3000.0; // m/s
        
        // Quarter wavelength thickness
        let wavelength = sound_speed / frequency;
        let thickness = wavelength / 4.0;
        
        Self {
            num_layers: 1,
            thickness,
            acoustic_impedance,
            sound_speed,
        }
    }
    
    /// Design dual matching layers
    pub fn dual_layer(frequency: f64, z_piezo: f64, z_load: f64) -> Vec<Self> {
        // Optimal impedances for two layers (Kino 1987)
        let z1 = z_piezo.powf(0.67) * z_load.powf(0.33);
        let z2 = z_piezo.powf(0.33) * z_load.powf(0.67);
        
        let sound_speed = 3000.0;
        let wavelength = sound_speed / frequency;
        let thickness = wavelength / 4.0;
        
        vec![
            Self {
                num_layers: 2,
                thickness,
                acoustic_impedance: z1,
                sound_speed,
            },
            Self {
                num_layers: 2,
                thickness,
                acoustic_impedance: z2,
                sound_speed,
            },
        ]
    }
}

/// Acoustic lens design for focusing
#[derive(Debug, Clone)]
pub struct AcousticLens {
    /// Lens material
    pub material: LensMaterial,
    /// Radius of curvature [m]
    pub radius_of_curvature: f64,
    /// Center thickness [m]
    pub center_thickness: f64,
    /// Aperture diameter [m]
    pub aperture: f64,
    /// Focal length [m]
    pub focal_length: f64,
    /// F-number
    pub f_number: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LensMaterial {
    Silicone,    // Defocusing
    Polystyrene, // Focusing
    Epoxy,       // Focusing
}

impl AcousticLens {
    /// Design focusing lens
    pub fn focusing_lens(
        focal_length: f64,
        aperture: f64,
        frequency: f64,
    ) -> KwaversResult<Self> {
        // Lens equation: 1/f = (n-1)/R
        // where n = c_medium/c_lens
        
        // Polystyrene lens in water
        let c_lens = 2350.0;  // m/s in polystyrene
        let c_medium = 1540.0; // m/s in water
        let n = c_medium / c_lens;
        
        let radius_of_curvature = focal_length * (n - 1.0);
        
        if radius_of_curvature <= 0.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "radius_of_curvature".to_string(),
                value: format!("{:.3}", radius_of_curvature),
                constraint: "must be > 0.0 for valid focusing".to_string(),
            }));
        }
        
        // Calculate center thickness
        let sagitta = radius_of_curvature - 
                     (radius_of_curvature.powi(2) - (aperture / 2.0).powi(2)).sqrt();
        let center_thickness = sagitta + aperture / 10.0; // Add margin
        
        let f_number = focal_length / aperture;
        
        Ok(Self {
            material: LensMaterial::Polystyrene,
            radius_of_curvature,
            center_thickness,
            aperture,
            focal_length,
            f_number,
        })
    }
}

/// Element directivity pattern
#[derive(Debug, Clone)]
pub struct DirectivityPattern {
    /// Angles [rad]
    pub angles: Array1<f64>,
    /// Directivity values [normalized]
    pub directivity: Array1<f64>,
    /// -3dB beamwidth [rad]
    pub beamwidth_3db: f64,
    /// -6dB beamwidth [rad]
    pub beamwidth_6db: f64,
    /// Main lobe to side lobe ratio [dB]
    pub sidelobe_level: f64,
}

impl DirectivityPattern {
    /// Calculate rectangular element directivity
    pub fn rectangular_element(
        width: f64,
        height: f64,
        frequency: f64,
        num_points: usize,
    ) -> Self {
        let wavelength = 1540.0 / frequency; // Assume water/tissue
        let angles = Array1::linspace(-PI / 2.0, PI / 2.0, num_points);
        let mut directivity = Array1::zeros(num_points);
        
        let kw = 2.0 * PI * width / wavelength;
        let kh = 2.0 * PI * height / wavelength;
        
        for (i, &theta) in angles.iter().enumerate() {
            // Directivity of rectangular piston
            let arg_w = kw * theta.sin() / 2.0;
            let arg_h = kh * theta.sin() / 2.0;
            
            let sinc_w = if arg_w.abs() < 1e-10 {
                1.0
            } else {
                arg_w.sin() / arg_w
            };
            
            let sinc_h = if arg_h.abs() < 1e-10 {
                1.0
            } else {
                arg_h.sin() / arg_h
            };
            
            directivity[i] = (sinc_w * sinc_h).abs();
        }
        
        // Find beamwidths
        let max_val = directivity.iter().cloned().fold(0.0, f64::max);
        let threshold_3db = max_val / 2.0_f64.sqrt();
        let threshold_6db = max_val / 2.0;
        
        let mut beamwidth_3db = 0.0;
        let mut beamwidth_6db = 0.0;
        
        for i in 0..num_points / 2 {
            if directivity[num_points / 2 + i] < threshold_3db && beamwidth_3db == 0.0 {
                beamwidth_3db = 2.0 * angles[num_points / 2 + i].abs();
            }
            if directivity[num_points / 2 + i] < threshold_6db && beamwidth_6db == 0.0 {
                beamwidth_6db = 2.0 * angles[num_points / 2 + i].abs();
            }
        }
        
        // Find first sidelobe level
        let mut sidelobe_level = -60.0; // Default if no sidelobe
        let center = num_points / 2;
        for i in center + 1..num_points - 1 {
            if directivity[i] > directivity[i - 1] && directivity[i] > directivity[i + 1] {
                sidelobe_level = 20.0 * (directivity[i] / max_val).log10();
                break;
            }
        }
        
        Self {
            angles,
            directivity,
            beamwidth_3db,
            beamwidth_6db,
            sidelobe_level,
        }
    }
}

/// Cross-coupling and crosstalk between elements
#[derive(Debug, Clone)]
pub struct ElementCoupling {
    /// Coupling matrix (symmetric)
    pub coupling_matrix: Array2<f64>,
    /// Number of elements
    pub num_elements: usize,
    /// Maximum coupling coefficient
    pub max_coupling: f64,
    /// Coupling decay with distance
    pub decay_factor: f64,
}

impl ElementCoupling {
    /// Calculate coupling matrix for linear array
    pub fn linear_array(
        num_elements: usize,
        pitch: f64,
        frequency: f64,
    ) -> Self {
        let mut coupling_matrix = Array2::eye(num_elements);
        let wavelength = 1540.0 / frequency;
        let decay_factor = (-2.0 * PI * pitch / wavelength).exp();
        
        for i in 0..num_elements {
            for j in 0..num_elements {
                if i != j {
                    let distance = ((i as i32 - j as i32).abs() as f64) * pitch;
                    let coupling = CROSS_COUPLING_COEFFICIENT * 
                                  (-distance / wavelength).exp();
                    coupling_matrix[(i, j)] = coupling;
                }
            }
        }
        
        let max_coupling = CROSS_COUPLING_COEFFICIENT;
        
        Self {
            coupling_matrix,
            num_elements,
            max_coupling,
            decay_factor,
        }
    }
    
    /// Apply coupling effects to element signals
    pub fn apply_coupling(&self, signals: &Array1<Complex64>) -> Array1<Complex64> {
        let mut coupled = Array1::zeros(self.num_elements);
        
        for i in 0..self.num_elements {
            let mut sum = Complex64::new(0.0, 0.0);
            for j in 0..self.num_elements {
                sum += signals[j] * self.coupling_matrix[(i, j)];
            }
            coupled[i] = sum;
        }
        
        coupled
    }
}

/// Transducer sensitivity and efficiency
#[derive(Debug, Clone)]
pub struct TransducerSensitivity {
    /// Transmit sensitivity [Pa/V]
    pub transmit_sensitivity: f64,
    /// Receive sensitivity [V/Pa]
    pub receive_sensitivity: f64,
    /// Electroacoustic efficiency [%]
    pub efficiency: f64,
    /// Insertion loss [dB]
    pub insertion_loss: f64,
    /// Round-trip sensitivity [V/V]
    pub round_trip_sensitivity: f64,
}

impl TransducerSensitivity {
    /// Calculate from transducer parameters
    pub fn calculate(
        geometry: &ElementGeometry,
        material: &PiezoMaterial,
        frequency: f64,
    ) -> Self {
        // Transmit sensitivity (Mason model)
        let area_factor = geometry.area.sqrt();
        let coupling_factor = material.coupling_kt;
        let transmit_sensitivity = 
            area_factor * coupling_factor * material.acoustic_impedance * 1e6;
        
        // Receive sensitivity (reciprocity)
        let capacitance = geometry.calculate_capacitance(material.relative_permittivity);
        let receive_sensitivity = 
            coupling_factor / (2.0 * PI * frequency * capacitance * material.acoustic_impedance);
        
        // Efficiency (based on coupling and Q factors)
        let efficiency = 100.0 * coupling_factor.powi(2) * 
                        material.mechanical_q / (material.mechanical_q + material.electrical_q);
        
        // Insertion loss
        let z_ratio = material.acoustic_impedance / TISSUE_IMPEDANCE;
        let reflection_coeff = (z_ratio - 1.0) / (z_ratio + 1.0);
        let transmission_coeff = 1.0 - reflection_coeff.abs();
        let insertion_loss = -20.0 * transmission_coeff.log10();
        
        // Round-trip sensitivity
        let round_trip_sensitivity = transmit_sensitivity * receive_sensitivity;
        
        Self {
            transmit_sensitivity,
            receive_sensitivity,
            efficiency,
            insertion_loss,
            round_trip_sensitivity,
        }
    }
}

/// Complete transducer design
pub struct TransducerDesign {
    /// Element geometry
    pub geometry: ElementGeometry,
    /// Piezo material
    pub material: PiezoMaterial,
    /// Backing layer
    pub backing: BackingLayer,
    /// Matching layers
    pub matching: Vec<MatchingLayer>,
    /// Frequency response
    pub frequency_response: FrequencyResponse,
    /// Directivity pattern
    pub directivity: DirectivityPattern,
    /// Element coupling
    pub coupling: ElementCoupling,
    /// Sensitivity
    pub sensitivity: TransducerSensitivity,
    /// Optional acoustic lens
    pub lens: Option<AcousticLens>,
}

impl TransducerDesign {
    /// Design optimized transducer for given specifications
    pub fn design_for_application(
        center_frequency: f64,
        num_elements: usize,
        aperture: f64,
        focal_length: Option<f64>,
    ) -> KwaversResult<Self> {
        // Calculate element dimensions
        let wavelength = 1540.0 / center_frequency;
        let element_width = wavelength * 0.95; // Just under λ for grating lobe suppression
        let kerf = element_width * 0.1; // 10% kerf
        let pitch = element_width + kerf;
        
        // Determine array height
        let array_width = num_elements as f64 * pitch;
        let element_height = aperture / array_width * element_width * 10.0; // Elevation dimension
        
        // Material selection based on frequency
        let material = if center_frequency < 5e6 {
            PiezoMaterial::pzt_5h() // Lower frequency, high sensitivity
        } else {
            PiezoMaterial::pmn_pt() // Higher frequency, wide bandwidth
        };
        
        // Calculate thickness for half-wave resonance
        let thickness = material.sound_speed / (2.0 * center_frequency);
        
        // Create geometry
        let geometry = ElementGeometry::new(
            element_width,
            element_height,
            thickness,
            kerf,
        )?;
        
        // Design backing for bandwidth
        let backing = BackingLayer::tungsten_epoxy(thickness * 10.0);
        
        // Design matching layers
        let matching = MatchingLayer::dual_layer(
            center_frequency,
            material.acoustic_impedance,
            TISSUE_IMPEDANCE,
        );
        
        // Calculate frequency response
        let frequency_response = FrequencyResponse::from_klm_model(
            &geometry,
            &material,
            &backing,
            &matching[0],
            100,
        )?;
        
        // Calculate directivity
        let directivity = DirectivityPattern::rectangular_element(
            element_width,
            element_height,
            center_frequency,
            180,
        );
        
        // Calculate coupling
        let coupling = ElementCoupling::linear_array(
            num_elements,
            pitch,
            center_frequency,
        );
        
        // Calculate sensitivity
        let sensitivity = TransducerSensitivity::calculate(
            &geometry,
            &material,
            center_frequency,
        );
        
        // Optional lens design
        let lens = if let Some(fl) = focal_length {
            Some(AcousticLens::focusing_lens(fl, aperture, center_frequency)?)
        } else {
            None
        };
        
        Ok(Self {
            geometry,
            material,
            backing,
            matching,
            frequency_response,
            directivity,
            coupling,
            sensitivity,
            lens,
        })
    }
    
    /// Validate design against specifications
    pub fn validate(&self) -> KwaversResult<()> {
        // Check bandwidth
        if self.frequency_response.fractional_bandwidth < 20.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "fractional_bandwidth".to_string(),
                value: format!("{:.1}%", self.frequency_response.fractional_bandwidth),
                constraint: "must be >= 20%".to_string(),
            }));
        }
        
        // Check efficiency
        if self.sensitivity.efficiency < 30.0 {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "efficiency".to_string(),
                value: format!("{:.1}%", self.sensitivity.efficiency),
                constraint: "must be >= 30%".to_string(),
            }));
        }
        
        // Check aspect ratio
        if self.geometry.aspect_ratio > MAX_ASPECT_RATIO {
            return Err(KwaversError::Config(ConfigError::InvalidValue {
                parameter: "aspect_ratio".to_string(),
                value: format!("{:.1}", self.geometry.aspect_ratio),
                constraint: format!("must be <= {:.1}", MAX_ASPECT_RATIO),
            }));
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_element_geometry() {
        let geometry = ElementGeometry::new(
            0.3e-3,  // 0.3mm width
            10e-3,   // 10mm height
            0.5e-3,  // 0.5mm thickness
            0.05e-3, // 0.05mm kerf
        ).unwrap();
        
        assert!((geometry.pitch - 0.35e-3).abs() < 1e-6);
        assert!((geometry.aspect_ratio - 0.6).abs() < 0.01);
    }
    
    #[test]
    fn test_piezo_materials() {
        let pzt5h = PiezoMaterial::pzt_5h();
        assert_eq!(pzt5h.coupling_k33, 0.75);
        
        let pmn_pt = PiezoMaterial::pmn_pt();
        assert_eq!(pmn_pt.coupling_k33, 0.90);
    }
    
    #[test]
    fn test_matching_layer_design() {
        let matching = MatchingLayer::quarter_wave(
            2.5e6,  // 2.5 MHz
            34.5,   // PZT impedance
            1.5,    // Tissue impedance
        );
        
        let expected_z = (34.5 * 1.5_f64).sqrt();
        assert!((matching.acoustic_impedance - expected_z).abs() < 0.1);
    }
    
    #[test]
    fn test_transducer_design() {
        let design = TransducerDesign::design_for_application(
            2.5e6,     // 2.5 MHz
            64,        // 64 elements
            20e-3,     // 20mm aperture
            Some(50e-3), // 50mm focal length
        ).unwrap();
        
        assert!(design.validate().is_ok());
        assert!(design.frequency_response.fractional_bandwidth > 20.0);
    }
    
    #[test]
    fn test_directivity_pattern() {
        let pattern = DirectivityPattern::rectangular_element(
            0.5e-3,  // 0.5mm width
            10e-3,   // 10mm height
            2.5e6,   // 2.5 MHz
            180,
        );
        
        assert!(pattern.beamwidth_3db > 0.0);
        assert!(pattern.beamwidth_6db > pattern.beamwidth_3db);
        assert!(pattern.sidelobe_level < -10.0);
    }
}