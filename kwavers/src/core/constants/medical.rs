//! Medical and bioeffects constants

/// FDA derating factor for in situ intensity
pub const FDA_DERATING_FACTOR: f64 = 0.3;

/// Typical diagnostic ultrasound frequency (Hz)
pub const DIAGNOSTIC_FREQUENCY: f64 = 3.5e6;

/// Typical therapeutic ultrasound frequency (Hz)
pub const THERAPEUTIC_FREQUENCY: f64 = 1e6;

/// HIFU frequency range minimum (Hz)
pub const HIFU_FREQUENCY_MIN: f64 = 0.5e6;

/// HIFU frequency range maximum (Hz)
pub const HIFU_FREQUENCY_MAX: f64 = 5e6;

/// Spatial peak temporal average intensity limit (W/cm²)
pub const ISPTA_LIMIT: f64 = 720.0;

/// Spatial peak pulse average intensity limit (W/cm²)
pub const ISPPA_LIMIT: f64 = 190.0;

/// Thermal dose threshold for tissue damage (CEM43 equivalent minutes at 43°C).
///
/// Value: 240 CEM43 — corresponds to irreversible cell damage in soft tissue
/// (equivalent exposure to 43°C for 240 minutes).
///
/// Reference: Sapareto SA, Dewey WC (1984). "Thermal dose determination in
/// cancer therapy." Int. J. Radiat. Oncol. Biol. Phys. 10(6), 787–800.
pub const THERMAL_DOSE_THRESHOLD: f64 = 240.0;

/// Reference temperature for the CEM43 thermal dose formula (°C).
///
/// The Sapareto–Dewey formula CEM43 = ∫ R^(43−T) dt uses 43°C as the
/// reference temperature. Above 43°C: R = 0.5; below: R = 0.25.
///
/// Reference: Sapareto SA, Dewey WC (1984). Int. J. Radiat. Oncol.
/// Biol. Phys. 10(6), 787–800. DOI: 10.1016/0360-3016(84)90379-1.
pub const THERMAL_DOSE_REFERENCE_TEMP_C: f64 = 43.0;

/// CEM43 doubling constant R above the 43°C reference (dimensionless).
///
/// For T ≥ 43°C, each 1°C above the reference doubles the thermal dose rate.
/// R = 0.5 means the dose accumulates at e^(43−T)·ln(0.5) per minute,
/// which halves for each 1°C below 43°C in the above-threshold regime.
///
/// Reference: Sapareto SA, Dewey WC (1984). Int. J. Radiat. Oncol. Biol.
/// Phys. 10(6), 787–800.
pub const THERMAL_DOSE_R_ABOVE_43C: f64 = 0.5;

/// CEM43 doubling constant R below the 43°C reference (dimensionless).
///
/// For T < 43°C, thermal damage accumulates more slowly: R = 0.25 per 1°C
/// below 43°C (two doublings per degree, i.e., ~4× faster accumulation rate
/// per degree than the above-threshold regime).
///
/// Reference: Sapareto SA, Dewey WC (1984). Int. J. Radiat. Oncol. Biol.
/// Phys. 10(6), 787–800.
pub const THERMAL_DOSE_R_BELOW_43C: f64 = 0.25;

/// Thermal dose threshold for reversible tissue damage (CEM43 equivalent minutes at 43°C).
///
/// Value: 60 CEM43 — corresponds to onset of reversible thermal damage.
/// Below this threshold tissue typically recovers; above it damage may persist.
///
/// Reference: Dewhirst MW et al. (2003). "Basic principles of thermal dosimetry
/// and thermal thresholds for tissue damage from hyperthermia."
/// Int. J. Hyperthermia 19(3), 267–294.
pub const THERMAL_DOSE_DAMAGE_THRESHOLD_CEM43: f64 = 60.0;

/// Perfusion rate in generic soft tissue (1/s).
///
/// Generic default for non-brain soft tissue in Pennes bioheat equation.
pub const TISSUE_PERFUSION_RATE: f64 = 5e-4;

/// Brain-specific blood perfusion rate (1/s).
///
/// High-perfusion estimate for brain parenchyma: ~50 mL/min/100 g.
/// Reference: Yarnykh & Yuan (2010), *NeuroImage* 51(3).
/// Distinct from generic `TISSUE_PERFUSION_RATE` (5e-4 1/s).
pub const BLOOD_PERFUSION_RATE_BRAIN: f64 = 0.0064;

/// Blood specific heat capacity (J/(kg·K))
pub const BLOOD_SPECIFIC_HEAT: f64 = 3617.0;

/// Typical HIFU focal intensity (W/cm²)
pub const HIFU_FOCAL_INTENSITY: f64 = 1000.0;

/// IEC 62359:2017 reference acoustic power for TIS computation [W].
///
/// The standard defines TIS = W_stp [mW] / (210 · f_MHz), so
/// W_stp_reference = 40 mW (the lowest acoustic output considered in the standard).
///
/// Reference: IEC 62359:2017 §8.3; AIUM/NEMA UD-3:2012.
pub const IEC_REFERENCE_ACOUSTIC_POWER_W: f64 = 0.040; // W (= 40 mW)

/// IEC 62359 (2017) §8.3.2 — Thermal Index soft tissue (TIS) formula divisor
/// (mW · MHz)⁻¹.
///
/// TIS = W_stp [mW] / (IEC_TIS_DIVISOR · f_MHz)
///
/// Reference: IEC 62359 (2017). *Ultrasonics — Field characterization —
/// Test methods for the determination of thermal and mechanical indices
/// related to medical diagnostic ultrasonic fields.*
pub const IEC_TIS_DIVISOR: f64 = 210.0;

/// IEC 62359 (2017) §8.4 — Thermal Index bone (TIB) formula divisor (mW · MHz).
///
/// TIB = W [mW] · f_MHz / IEC_TIB_DIVISOR
///
/// Reference: IEC 62359 (2017). *Ultrasonics — Field characterization —
/// Test methods for the determination of thermal and mechanical indices
/// related to medical diagnostic ultrasonic fields.*
pub const IEC_TIB_DIVISOR: f64 = 40.0;

/// Default ultrasound frequency (Hz)
pub const DEFAULT_ULTRASOUND_FREQUENCY: f64 = 1e6;

/// Standard pressure amplitude (Pa)
pub const STANDARD_PRESSURE_AMPLITUDE: f64 = 1e6;

/// Standard beam width (m)
pub const STANDARD_BEAM_WIDTH: f64 = 0.01;

/// IEC 62127 tissue model specific heat capacity at 37°C (J/(kg·K)).
///
/// The IEC 62127-1 homogeneous tissue model (Table A.1) specifies
/// c_p = 3500 J/(kg·K), distinct from the Duck (1990) general soft-tissue
/// mean of 3600 J/(kg·K). Use this constant exclusively when computing
/// temperature rise for IEC 62127 compliance.
///
/// Reference: IEC 62127-1:2013, Annex A, Table A.1.
pub const IEC_TISSUE_SPECIFIC_HEAT: f64 = 3500.0;

/// IEC 62127-1 homogeneous-tissue acoustic absorption coefficient [dB/(cm·MHz)].
///
/// The IEC 62127-1 standardised tissue model (Table A.1) uses α = 0.3 dB/(cm·MHz)
/// for thermal-index compliance calculations.  This value underpins the TIS and
/// TIB formulae in the Output Display Standard.
///
/// Reference: IEC 62127-1:2013, Annex A, Table A.1.
pub const IEC_TISSUE_ABSORPTION_DB_CM_MHZ: f64 = 0.3;

// ── Mechanical Index safety limits ───────────────────────────────────────────
//
// FDA 510(k) guidance for diagnostic ultrasound output display standard (ODS)
// and WFUMB safety symposium consensus statements.
//
// References:
// - US FDA (2019). "Guidance for Industry and FDA Staff — Information for
//   Manufacturers Seeking Marketing Clearance of Diagnostic Ultrasound Systems
//   and Transducers." Table 1.
// - WFUMB (2015). Safety symposium on echographic contrast agents.
//   Ultrasound in Med. & Biol. 41(2), 311–333.

/// Mechanical index safety limit for general soft tissue (dimensionless).
///
/// FDA diagnostic output display standard: MI ≤ 1.9 for soft tissue.
///
/// Reference: FDA (2019) Table 1; AIUM/NEMA UD-2 Output Display Standard.
pub const MI_LIMIT_SOFT_TISSUE: f64 = 1.9;

/// Mechanical index safety limit for ophthalmic applications (dimensionless).
///
/// FDA diagnostic output display standard: MI ≤ 0.23 for ophthalmic use,
/// reflecting the higher sensitivity of ocular structures.
///
/// Reference: FDA (2019) Table 1.
pub const MI_LIMIT_OPHTHALMIC: f64 = 0.23;

/// Mechanical index safety limit for lung/bowel tissue with gas bodies
/// (dimensionless).
///
/// WFUMB guidance: MI ≤ 0.7 in the presence of gas-body-containing tissues
/// (lung, bowel) to reduce risk of lung haemorrhage and capillary rupture.
///
/// Reference: WFUMB (2015); FDA (2019) Table 1.
pub const MI_LIMIT_LUNG: f64 = 0.7;

/// Mechanical index safety limit for bowel tissue with gas bodies
/// (dimensionless). Numerically equal to `MI_LIMIT_LUNG`; distinct constant
/// for call-site readability in bowel-imaging contexts.
pub const MI_LIMIT_BOWEL: f64 = 0.7;

/// Conservative mechanical index safety limit for fetal applications
/// (dimensionless).
///
/// Value: 1.0 — no formal regulatory mandate below 1.9, but AIUM practice
/// guidelines recommend MI < 1.0 during pregnancy, especially in the first
/// trimester.
///
/// Reference: AIUM (2012) AIUM Practice Guideline for the Performance of Fetal
/// Echocardiography.
pub const MI_LIMIT_FETAL: f64 = 1.0;

/// Mechanical index safety limit for transcranial brain applications
/// (dimensionless).
///
/// Value: 1.5 — empirical limit below the FDA soft-tissue MI = 1.9 to reduce
/// risk of microhaemorrhage through the skull. No single FDA standard; value
/// follows transcranial FUS literature consensus.
///
/// Reference: O'Reilly MA, Hynynen K (2012). Ultrasound Med. Biol. 38(1), 1–12.
pub const MI_LIMIT_BRAIN: f64 = 1.5;

// ── MI-based cavitation onset thresholds ─────────────────────────────────────

/// Estimated MI at cavitation onset for general soft tissue (dimensionless).
///
/// Below this MI the probability of inertial cavitation in soft tissue is
/// considered clinically negligible for typical diagnostic frequencies.
///
/// Reference: Apfel RE, Holland CK (1991). Ultrasound Med. Biol. 17(2), 179–185.
pub const MI_CAVITATION_SOFT_TISSUE: f64 = 0.6;

/// Estimated MI at cavitation onset for ophthalmic tissue (dimensionless).
pub const MI_CAVITATION_OPHTHALMIC: f64 = 0.3;

/// Estimated MI at cavitation onset for lung tissue (dimensionless).
pub const MI_CAVITATION_LUNG: f64 = 0.4;

/// Estimated MI at cavitation onset for bowel tissue (dimensionless).
pub const MI_CAVITATION_BOWEL: f64 = 0.4;

/// Conservative estimated MI at cavitation onset for fetal tissue (dimensionless).
pub const MI_CAVITATION_FETAL: f64 = 0.5;

/// Estimated MI at cavitation onset for brain tissue (dimensionless).
pub const MI_CAVITATION_BRAIN: f64 = 0.55;

// ── Thermal Index safety limits ──────────────────────────────────────────────
//
// The Thermal Index (TI) is an estimate of the maximum temperature rise in
// degrees Celsius in tissue during sonication.  Safety limits are specified
// in the NEMA UD-3 / IEC 62359 Output Display Standard and endorsed by AIUM.
//
// References:
// - NEMA UD-3 (2004/2012). "Standard for Real-Time Display of Thermal and
//   Mechanical Acoustic Output Indices on Diagnostic Ultrasound Equipment."
// - IEC 62359:2010. "Ultrasonics — Field characterization — Test methods for
//   the determination of thermal and mechanical indices related to medical
//   diagnostic ultrasonic fields."
// - AIUM (2013). "AIUM Practice Parameter for the Performance of Diagnostic
//   and Screening Ultrasound of the Abdomen and/or Retroperitoneum."

/// Thermal index safety limit for soft tissue (dimensionless, °C equivalent).
///
/// TI ≤ 6.0 is the standard display limit for soft-tissue scanning in adults.
/// Below TI = 1 no restriction on scan duration; above TI = 6 the benefit–risk
/// ratio must be explicitly justified.
///
/// Reference: NEMA UD-3:2012, Table 1; IEC 62359:2010.
pub const TI_LIMIT_SOFT_TISSUE: f64 = 6.0;
