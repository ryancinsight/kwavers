//! Fundamental physical constants

/// Speed of sound in water at 20°C (m/s) — k-Wave simulation default
///
/// Value: 1500.0 m/s — the standard nominal reference used in k-Wave and most
/// ultrasound simulation/beamforming literature. This is a round-number approximation
/// suitable for simulation defaults and signal processing defaults.
/// Note: the physically precise value at 20°C is 1482 m/s (see `SOUND_SPEED_WATER_PRECISE`).
/// Reference: Treeby & Cox (2010), k-Wave Toolbox default; Duck (1990)
pub const SOUND_SPEED_WATER_SIM: f64 = 1500.0;

/// Speed of sound in water at 20°C (m/s) — physically precise value
///
/// Value: 1482.0 m/s at 20°C, 1 atm.
///
/// For temperature-dependent speed of sound use
/// [`crate::core::constants::water::WaterProperties::sound_speed`], which
/// implements the Del Grosso & Mader (1972) 5th-order polynomial accurate to
/// ±0.1 m/s over 0–100 °C.
///
/// For temperature-dependent viscosity use
/// [`crate::core::constants::state_dependent::StateDependentConstants::dynamic_viscosity_water`]
/// (Dortmund Data Bank VFT formula, < 2% vs NIST over 0–100 °C) or
/// [`crate::core::constants::state_dependent::StateDependentConstants::viscosity_arrhenius`]
/// for generic Arrhenius fluids.
///
/// Reference: National Physical Laboratory acoustic properties database.
pub const SOUND_SPEED_WATER: f64 = 1482.0;

/// Speed of sound in soft tissue (m/s)
pub const SOUND_SPEED_TISSUE: f64 = 1540.0;

/// Speed of sound in air at 20°C (m/s)
pub const SOUND_SPEED_AIR: f64 = 343.0;

/// Density of water at 20°C (kg/m³)
/// Value: 998.2 kg/m³ (precise value)
/// Reference: NIST Chemistry WebBook
pub const DENSITY_WATER: f64 = 998.2;

/// Nominal density of water (kg/m³) — round-number simulation default
///
/// Value: 1000.0 kg/m³ — the standard round-number approximation used throughout
/// ultrasound simulation, HIFU planning, and acoustic modelling literature when
/// sub-percent density accuracy is not required.
///
/// Use `DENSITY_WATER = 998.2 kg/m³` when physical precision is needed.
///
/// Reference: k-Wave toolbox default; Duck, F. A. (1990). Physical Properties of
/// Tissue. Academic Press, London.
pub const DENSITY_WATER_NOMINAL: f64 = 1000.0;

/// Speed of sound in water at 37°C / body temperature (m/s)
///
/// Value: 1524.0 m/s — Del Grosso & Mader (1972), measured at 37°C.
/// Water sound speed increases monotonically with temperature up to ≈74°C.
/// At 25°C: ≈1497 m/s; at 37°C: 1524 m/s; at 20°C: 1482 m/s.
///
/// References:
/// - Del Grosso VA, Mader CW (1972). J. Acoust. Soc. Am. **52**(5):1442–1446.
/// - Duck FA (1990). Physical Properties of Tissue. Academic Press, Table 2.1.
pub const SOUND_SPEED_WATER_37C: f64 = 1524.0;

/// Density of water at 37°C (kg/m³)
///
/// Value: 993.3 kg/m³ — NIST thermophysical data at 310.15 K, 1 atm.
/// Reference: NIST Chemistry WebBook SRD 69.
pub const DENSITY_WATER_37C: f64 = 993.3;

/// Sound speed in water at 20°C (m/s) - Alias for compatibility
pub const C_WATER: f64 = SOUND_SPEED_WATER;

/// Bulk modulus of water at 20°C (Pa)
/// K = ρ * c², where ρ = 998.2 kg/m³, c = 1482 m/s
pub const BULK_MODULUS_WATER: f64 = 2.19e9;

/// Nominal acoustic impedance of water / water-like tissue (Pa·s/m = Rayl).
///
/// Derived from simulation-default values: Z = ρ·c = `DENSITY_WATER_NOMINAL` × `SOUND_SPEED_WATER_SIM`
/// = 1000 kg/m³ × 1500 m/s = 1.5 × 10⁶ Pa·s/m = 1.5 MRayl.
///
/// This is the canonical simulation impedance used for water-coupling, tissue-path
/// intensity estimation (`I = p²/Z`), and reflection-coefficient calculations where
/// precise tissue heterogeneity is not modelled.
///
/// Reference: Duck FA (1990). Physical Properties of Tissue. Academic Press; k-Wave defaults.
pub const ACOUSTIC_IMPEDANCE_WATER_NOMINAL: f64 = DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM;

/// Density of soft tissue (kg/m³)
pub const DENSITY_TISSUE: f64 = 1050.0;

// ── Tissue-specific acoustic SSOT ────────────────────────────────────────────
//
// Consensus values from the standard tissue-property compilations. Use these
// in tissue-specific factory methods (e.g. [`super::super::acoustic_parameters`],
// [`crate::physics::acoustics::wave_propagation::nonlinear::parameters`]) so
// liver / kidney / brain / fat simulations remain consistent across modules.
//
// Primary references:
// - Duck FA (1990). *Physical Properties of Tissue: A Comprehensive Reference
//   Book*. Academic Press, London. ISBN 0-12-222800-6.
// - Goss SA, Johnston RL, Dunn F (1978). "Comprehensive compilation of
//   empirical ultrasonic properties of mammalian tissues."
//   J. Acoust. Soc. Am. 64(2), 423–457. DOI: 10.1121/1.382016.
// - Bjørnø L (2002). "Forty years of nonlinear ultrasound."
//   Ultrasonics 40(1–8), 11–17. DOI: 10.1016/S0041-624X(02)00084-7.

/// Density of human liver parenchyma at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; ICRP-23 Reference Man Table 22.
pub const DENSITY_LIVER: f64 = 1060.0;

/// Density of human renal cortex at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1.
pub const DENSITY_KIDNEY: f64 = 1050.0;

/// Density of human brain (mean of white + grey matter) at body temperature
/// (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; ICRP-89 (2002) Table 4.4.
pub const DENSITY_BRAIN: f64 = 1040.0;

/// Density of human adipose tissue at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; Goss et al. (1978) Table II.
pub const DENSITY_FAT: f64 = 928.0;

/// Small-signal sound speed in human liver parenchyma at body temperature
/// (m/s).
///
/// Reference: Duck (1990) Table 4.6; Goss et al. (1978) Table V.
pub const SOUND_SPEED_LIVER: f64 = 1578.0;

/// Small-signal sound speed in human renal cortex at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_KIDNEY: f64 = 1560.0;

/// Small-signal sound speed in human brain (mean of grey + white matter) at
/// body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6; consistent with brain-mean values
/// reported in Goldman & Hueter (1956) and subsequent reviews.
pub const SOUND_SPEED_BRAIN: f64 = 1546.0;

/// Small-signal sound speed in human adipose tissue at body temperature
/// (m/s).
///
/// Reference: Duck (1990) Table 4.6; Goss et al. (1978) Table V.
pub const SOUND_SPEED_FAT: f64 = 1450.0;

/// Nonlinearity parameter B/A for human liver parenchyma (dimensionless).
///
/// Drives the second-order pressure term in the Westervelt / KZK equation
/// through `β = 1 + B/(2A)`. Range across the published literature is
/// 6.5–7.6; the central value 6.75 reflects the Duck (1990) Table 4.16 mean.
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002) Table 1.
pub const B_OVER_A_LIVER: f64 = 6.75;

/// Nonlinearity parameter B/A for human renal cortex (dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002) Table 1.
pub const B_OVER_A_KIDNEY: f64 = 7.2;

/// Nonlinearity parameter B/A for human brain (mean grey + white matter,
/// dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Law et al. (1985) UMB 11(2), 307–318.
pub const B_OVER_A_BRAIN: f64 = 6.55;

/// Nonlinearity parameter B/A for human adipose tissue (dimensionless).
///
/// Fat is the most nonlinear soft tissue in the body; relevant for breast
/// imaging and subcutaneous-fat heating during HIFU.
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002) Table 1.
pub const B_OVER_A_FAT: f64 = 9.6;

/// Nonlinearity parameter B/A for skeletal muscle (dimensionless).
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002) Table 1.
pub const B_OVER_A_MUSCLE: f64 = 7.4;

/// Nonlinearity parameter B/A for water at 20°C (dimensionless).
///
/// Value: 5.2 — standard reference condition for acoustic calibration and water-path
/// simulations. Rises to ~5.4 at 60°C; drops slightly to ~5.0 at 37°C (body temp).
///
/// Reference: Duck, F.A. (1990) Table 4.16; Beyer, R.T. (1960) J. Acoust. Soc. Am. 32(6).
pub const B_OVER_A_WATER: f64 = 5.2;

/// Nonlinearity parameter B/A for water at 37°C / body temperature (dimensionless).
///
/// Value: 5.0 — B/A decreases slightly with temperature; the value at 37°C is
/// approximately 5.0, consistent with the Aanonsen et al. (1984) experimental
/// water-path validation conditions (degassed water, body temperature). This
/// constant should be used whenever a water medium at physiological temperature
/// is specified, e.g., literature-validation tests anchored to that experiment.
///
/// Reference:
/// - Aanonsen SI, Barkve T, Naze Tjøtta J, Tjøtta S (1984). "Distortion and
///   harmonic generation in the nearfield of a finite amplitude sound beam."
///   J. Acoust. Soc. Am. **75**(3): 749–768. DOI: 10.1121/1.390585.
/// - Duck FA (1990). *Physical Properties of Tissue*, Table 4.16.
pub const B_OVER_A_WATER_37C: f64 = 5.0;

/// Nonlinearity parameter B/A for whole blood at 37°C (dimensionless).
///
/// Reference: Duck (1990) Table 4.16.
pub const B_OVER_A_BLOOD: f64 = 6.1;

/// Nonlinearity parameter B/A for cerebrospinal fluid at body temperature (dimensionless).
///
/// CSF composition is near-water at physiological temperature; B/A ≈ 5.0.
/// Reference: Duck (1990) Table 4.16.
pub const B_OVER_A_CSF: f64 = 5.0;

/// Small-signal sound speed in whole blood at body temperature (m/s).
///
/// Value: 1584 m/s — measured for normal adult whole blood at 37°C.
///
/// Reference: Duck (1990) *Physical Properties of Tissue*, Table 4.6, p. 100.
/// Consistent with TISSUE const catalog (BLOOD.sound_speed = 1584.0).
pub const SOUND_SPEED_BLOOD: f64 = 1584.0;

/// Density of whole blood at 37°C (kg/m³)
///
/// Value: 1060.0 kg/m³ — measured value for normal adult whole blood.
///
/// Reference: ICRP Publication 23 (1975), *Report of the Task Group on Reference Man*,
/// Pergamon Press, Table 22 (p. 346). Also confirmed in:
/// Duck, F. A. (1990). *Physical Properties of Tissue*. Academic Press, London, p. 119.
pub const DENSITY_BLOOD: f64 = 1060.0;

/// Density of breast adipose tissue at body temperature (kg/m³).
///
/// Value: 911.0 kg/m³ — IT'IS Foundation database v4.0 value for breast fat
/// (slightly lower than generic adipose due to higher water content in breast).
///
/// Reference: IT'IS Foundation (2018) "Tissue Properties Database";
/// Hasgall et al. (2022) itis.swiss/tissue-properties.
pub const DENSITY_BREAST_FAT: f64 = 911.0;

/// Density of skeletal muscle at body temperature (kg/m³).
///
/// Value: 1090.0 kg/m³ — upper end of Duck (1990) Table 4.1 range (1041–1090),
/// consistent with IT'IS Foundation database v4.0 and the thermal bioheat
/// literature.
///
/// Reference: Duck (1990) Table 4.1; IT'IS Foundation (2018)
/// "Tissue Properties Database"; ICRP-89 (2002) Table 4.4.
pub const DENSITY_MUSCLE: f64 = 1090.0;

/// Small-signal sound speed in skeletal muscle at body temperature (m/s).
///
/// Value: 1580.0 m/s — mean of Duck (1990) Table 4.6 range (1547–1626 m/s).
///
/// Reference: Duck (1990) Table 4.6; Goss et al. (1978) Table V.
pub const SOUND_SPEED_MUSCLE: f64 = 1580.0;

/// Density of air at 20°C (kg/m³)
/// Value: 1.204 kg/m³ (at 20°C, 1 atm)
/// Reference: NIST Standard Reference Database
pub const DENSITY_AIR: f64 = 1.204;

/// Standard atmospheric pressure (Pa)
pub const ATMOSPHERIC_PRESSURE: f64 = 101325.0;

/// Vapor pressure of water at 20°C (Pa)
pub const VAPOR_PRESSURE_WATER_20C: f64 = 2339.0;

/// Gravitational acceleration (m/s²)
pub const GRAVITY: f64 = 9.80665;

/// Universal gas constant (J/(mol·K))
pub const GAS_CONSTANT: f64 = 8.314462618;

/// Avogadro's number (1/mol)
pub const AVOGADRO: f64 = 6.02214076e23;

/// Boltzmann constant (J/K)
pub const BOLTZMANN: f64 = 1.380649e-23;

/// Planck constant (J·s)
///
/// Exact defined value since the 2019 SI redefinition.
pub const PLANCK: f64 = 6.62607015e-34;

/// Reduced Planck constant ℏ = h / (2π) (J·s).
///
/// Reference: 2018 CODATA derived value `1.054_571_817e-34 J·s`.
pub const REDUCED_PLANCK: f64 = 1.054_571_817e-34;

/// Speed of light in vacuum (m/s)
pub const SPEED_OF_LIGHT: f64 = 299792458.0;

/// Stefan-Boltzmann constant (W/(m²·K⁴))
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;

/// Elementary charge (C)
pub const ELEMENTARY_CHARGE: f64 = 1.602176634e-19;

/// Vacuum permittivity (F/m)
/// Value: 8.8541878128e-12 F/m
/// Reference: 2018 CODATA recommended values
pub const VACUUM_PERMITTIVITY: f64 = 8.8541878128e-12;

/// Vacuum permeability (H/m)
///
/// Value: 1.25663706212×10⁻⁶ H/m (2018 CODATA recommended value).
/// Numerically equal to `4π × 10⁻⁷` to nine significant figures — the pre-2019
/// defined value. Both `4π·1e-7` and `1.25663706212e-6` are accepted as μ₀
/// in the SI literature, but downstream code must use this single canonical
/// value to avoid drift in dimensional checks against ε₀ and c via
/// `c² · ε₀ · μ₀ = 1`.
///
/// Reference: 2018 CODATA recommended values.
pub const VACUUM_PERMEABILITY: f64 = 1.25663706212e-6;

/// Vacuum impedance Z₀ = √(μ₀ / ε₀) (Ω).
///
/// Value: 376.730_313_668 Ω (CODATA 2018).
/// Numerically `μ₀ · c` to high precision.
pub const VACUUM_IMPEDANCE: f64 = 376.730_313_668;

/// Electron invariant mass (kg)
/// Value: 9.1093837015e-31 kg
/// Reference: 2018 CODATA recommended values
pub const ELECTRON_MASS: f64 = 9.1093837015e-31;

// Pi is already available through std::f64::consts::PI

// ============================================================================
// Elastic Constants
// ============================================================================

/// Bond transformation factor for anisotropic media
pub const BOND_TRANSFORM_FACTOR: f64 = 2.0;

/// Lamé to stiffness conversion factor
pub const LAME_TO_STIFFNESS_FACTOR: f64 = 2.0;

/// Symmetry tolerance for elastic tensors
pub const SYMMETRY_TOLERANCE: f64 = 1e-6;

// ============================================================================
// Optical Tissue Constants (near-infrared, 750–900 nm)
// ============================================================================

/// Optical absorption coefficient of soft tissue in the near-infrared (750–900 nm) [m⁻¹]
///
/// Representative value at ~785 nm in bulk breast tissue.
/// Typical range: 0.5–15 m⁻¹ depending on chromophore content and wavelength.
///
/// Reference: Jacques, S.L. (2013). "Optical properties of biological tissues: a review."
/// Phys. Med. Biol. 58(11), R37–R61. DOI: 10.1088/0031-9155/58/11/R37
pub const OPTICAL_ABSORPTION_TISSUE_NIR: f64 = 10.0; // m⁻¹

/// Reduced scattering coefficient of soft tissue in the near-infrared (750–900 nm) [m⁻¹]
///
/// Corresponds to anisotropy factor g ≈ 0.9 with μs ≈ 10 000 m⁻¹.
/// Typical range: 500–1500 m⁻¹.
///
/// Reference: Jacques, S.L. (2013). Phys. Med. Biol. 58(11), R37–R61.
/// DOI: 10.1088/0031-9155/58/11/R37
pub const REDUCED_SCATTERING_TISSUE_NIR: f64 = 1000.0; // m⁻¹

/// Default acoustic absorption coefficient of soft tissue [dB/(cm·MHz)].
///
/// Value: 0.5 dB/(cm·MHz) — mid-range for generic soft tissue at diagnostic
/// frequencies. To convert to Np/m at frequency f_MHz:
///   α [Np/m] = 0.5 [dB/(cm·MHz)] × f_MHz [MHz] × 100 [cm/m] × DB_TO_NP [Np/dB]
///            = 5.756 × f_MHz  [Np/m]
///
/// Typical range: 0.3–1.0 dB/(cm·MHz) for most soft tissues.
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue.*
/// Academic Press, London, Chapter 4, Table 4.1.
pub const ACOUSTIC_ABSORPTION_TISSUE: f64 = 0.5; // dB/(cm·MHz)

/// Density of human skin at body temperature (kg/m³).
///
/// Reference: Duck (1990) Table 4.1; IT'IS Foundation (2022).
pub const DENSITY_SKIN: f64 = 1100.0;

/// Small-signal sound speed in human skin at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_SKIN: f64 = 1600.0;

/// Density of human lung parenchyma at total lung capacity (kg/m³).
///
/// Much lower than other soft tissues due to high air fraction (~70% air by volume).
///
/// Reference: Duck (1990) Table 4.1; ICRP-89 (2002).
pub const DENSITY_LUNG: f64 = 400.0;

/// Small-signal sound speed in lung parenchyma at body temperature (m/s).
///
/// Dramatically reduced compared to soft tissue due to gas-liquid mixture.
///
/// Reference: Duck (1990) Table 4.6.
pub const SOUND_SPEED_LUNG: f64 = 650.0;

/// Density of human breast glandular tissue at body temperature (kg/m³).
///
/// Reference: IT'IS Foundation database v4.0 (2022); Hasgall et al. (2022).
pub const DENSITY_BREAST_GLAND: f64 = 1041.0;

/// Small-signal sound speed in breast glandular tissue at body temperature (m/s).
///
/// Reference: Duck (1990) Table 4.6; Mast (2000) Ultrasound Med. Biol.
pub const SOUND_SPEED_BREAST_GLAND: f64 = 1510.0;

/// Nonlinearity parameter B/A for cortical bone (dimensionless).
///
/// Bone is highly nonlinear; measured values range 6–9 depending on bone type
/// and mineralisation. The value 8.0 is representative for cortical bone.
///
/// Reference: Duck (1990) Table 4.16; Bjørnø (2002).
pub const B_OVER_A_BONE: f64 = 8.0;

/// Nonlinearity parameter B/A for human skin (dimensionless).
///
/// Reference: Duck (1990) Table 4.16.
pub const B_OVER_A_SKIN: f64 = 7.5;

/// Nonlinearity parameter B/A for lung parenchyma (dimensionless).
///
/// Lung has reduced effective B/A due to air content; value from Duck (1990).
pub const B_OVER_A_LUNG: f64 = 8.0;

/// Nonlinearity parameter B/A for breast glandular tissue (dimensionless).
///
/// Reference: IT'IS Foundation; Duck (1990) Table 4.16.
pub const B_OVER_A_BREAST_GLAND: f64 = 7.0;

/// Nonlinearity parameter B/A for generic soft tissue (dimensionless).
///
/// Value: 6.5 — representative mean for soft tissues excluding bone and fat.
/// Used when no tissue-specific B/A constant applies.
///
/// Physical basis: β = 1 + B/(2A) couples the second-order pressure term
/// in the Westervelt/KZK equation. Tissue-specific values span 6.1 (blood)
/// to 9.6 (fat); this constant represents the approximate median excluding
/// adipose tissue.
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.16.
pub const B_OVER_A_SOFT_TISSUE: f64 = 6.5;

/// Nonlinearity parameter B/A for air (ideal diatomic gas, dimensionless).
///
/// For a thermally perfect diatomic gas: B/A = γ − 1 = 1.4 − 1 = 0.4.
/// Follows from the isentropic equation of state p ∝ ρ^γ and the definition
/// B/A = ρ₀(∂²p/∂ρ²)_s / (∂p/∂ρ)_s evaluated at the ideal-gas limit.
///
/// Reference: Hamilton MF & Blackstock DT (1998). *Nonlinear Acoustics*.
/// Academic Press, Chapter 2, Eq. (2.28).
pub const B_OVER_A_AIR: f64 = 0.4;

/// Power-law absorption coefficient prefactor for water [dB/(cm·MHz²)].
///
/// Classical thermoviscous loss in water follows α(f) = α₀·f² where α₀ ≈ 0.002
/// at 20°C.  Pair with `WATER_ABSORPTION_POWER_Y = 2.0`.
///
/// Reference: Duck (1990) §2; Szabo (2004) §4.2; Kinsler et al. (2000)
/// *Fundamentals of Acoustics* (4th ed.), Table B.1.
pub const WATER_ABSORPTION_ALPHA_0_DB_CM_MHZ2: f64 = 0.002; // dB/(cm·MHz²)

/// Power-law frequency exponent y for water acoustic absorption (dimensionless).
///
/// Water exhibits classical thermoviscous absorption proportional to f²,
/// so y = 2.0 in α(f) = α₀·f^y.
///
/// Reference: Duck (1990) §2; Szabo (2004) §4.
pub const WATER_ABSORPTION_POWER_Y: f64 = 2.0;

/// Power-law frequency exponent y for soft tissue acoustic absorption (dimensionless).
///
/// Soft tissue absorption is approximately linear in frequency (y ≈ 1.0–1.2).
/// The value 1.1 is the consensus from kHz–MHz diagnostic ultrasound measurements.
///
/// Reference: Duck (1990) Table 4.1; Szabo (2004) §4; Goss et al. (1978).
pub const SOFT_TISSUE_ABSORPTION_POWER_Y: f64 = 1.1;

/// Acoustic absorption coefficient of human brain tissue [dB/(cm·MHz)].
///
/// Brain white and grey matter mean value at diagnostic and therapeutic
/// frequencies.  Used in transcranial ultrasound propagation models.
///
/// Reference: Duck, F.A. (1990). *Physical Properties of Tissue*, Table 4.1;
/// Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93.
pub const ACOUSTIC_ABSORPTION_BRAIN: f64 = 0.5; // dB/(cm·MHz)

/// Minimum skull bone acoustic absorption coefficient [dB/(cm·MHz)].
///
/// Value at bone fraction = 0 (brain–bone boundary, primarily trabecular bone).
/// Interpolated linearly to `ACOUSTIC_ABSORPTION_SKULL_MIN + ACOUSTIC_ABSORPTION_SKULL_RANGE`
/// at bone fraction = 1 (pure cortical bone, ≈ 20 dB/(cm·MHz)).
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93, Table I.
pub const ACOUSTIC_ABSORPTION_SKULL_MIN: f64 = 8.0; // dB/(cm·MHz)

/// Acoustic absorption range across skull bone fraction [dB/(cm·MHz)].
///
/// The full-bone absorption is
/// `ACOUSTIC_ABSORPTION_SKULL_MIN + ACOUSTIC_ABSORPTION_SKULL_RANGE` = 20 dB/(cm·MHz).
/// Linear interpolation over bone fraction ∈ [0, 1] follows Aubry 2003.
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93, Table I.
pub const ACOUSTIC_ABSORPTION_SKULL_RANGE: f64 = 12.0; // dB/(cm·MHz)

/// Hounsfield unit threshold below which tissue is classified as brain [HU].
///
/// Below this value the voxel is treated as brain/soft tissue in transcranial
/// ray-tracing models. Above it, bone fraction is interpolated linearly.
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93.
pub const HU_BONE_THRESHOLD: f64 = 300.0; // Hounsfield units

/// Hounsfield unit range for skull bone-fraction interpolation [HU].
///
/// `bone_fraction = (HU − HU_BONE_THRESHOLD) / HU_SKULL_RANGE`, clamped to [0, 1].
/// At HU = 300 + 1700 = 2000 the voxel is pure cortical bone.
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93.
pub const HU_SKULL_RANGE: f64 = 1700.0; // Hounsfield units

/// Minimum skull/trabecular-bone density used in HU-based interpolation [kg/m³].
///
/// Density at bone fraction = 0 (brain–bone boundary).  Increases linearly to
/// `DENSITY_SKULL_MIN + DENSITY_SKULL_CORTICAL_RANGE` = 1 900 kg/m³ at bone fraction = 1.
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93;
/// Duck (1990) Table 3.8.
pub const DENSITY_SKULL_MIN: f64 = 1200.0; // kg/m³

/// Density increment from the brain–bone boundary to pure cortical bone [kg/m³].
///
/// `density = DENSITY_SKULL_MIN + DENSITY_SKULL_CORTICAL_RANGE × bone_fraction`.
/// Maximum density (full cortical bone) = 1 900 kg/m³.
///
/// Reference: Aubry et al. (2003). J. Acoust. Soc. Am. 113(1):84–93;
/// Duck (1990) Table 3.8.
pub const DENSITY_SKULL_CORTICAL_RANGE: f64 = 700.0; // kg/m³
