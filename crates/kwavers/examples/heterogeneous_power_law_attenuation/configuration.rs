use std::f64::consts::TAU;

pub(crate) const OUT_DIR: &str = "target/fullwave_attenuation";

/// 1 dB = 1/8.685889638 Np; cm⁻¹ → m⁻¹ multiplies by 100.
const NEPER_PER_DB: f64 = 1.0 / 8.685_889_638_065_035;

// One-dimensional water-like column. The spacing resolves 5 MHz at about ten
// points per wavelength, inside the pseudospectral solver's Nyquist limit.
pub(crate) const N: usize = 1536;
pub(crate) const DX: f64 = 3.0e-5;
pub(crate) const RHO: f64 = 1000.0;
pub(crate) const C0: f64 = 1540.0;
pub(crate) const F_REF: f64 = 1.0e6;
pub(crate) const F_MIN: f64 = 0.5e6;
pub(crate) const F_MAX: f64 = 5.0e6;
pub(crate) const N_ARMS: usize = 3;

pub(crate) const SOURCE_INDEX: usize = 220;
pub(crate) const SENSOR_NEAR: usize = 380;
pub(crate) const SENSOR_FAR: usize = 980;
pub(crate) const ABSORBER_CELLS: usize = 128;
/// Damps a pulse by many e-folds over the boundary layer at this CFL step.
pub(crate) const ABSORBER_GAMMA: f64 = 6.0e6;
pub(crate) const STEPS: usize = 9000;

/// Log-band-centred excitation frequency.
const PULSE_CENTRE_HZ: f64 = 1.6e6;
/// Gaussian envelope width; about two carrier cycles.
pub(crate) const PULSE_WIDTH_S: f64 = 0.35 / PULSE_CENTRE_HZ;
pub(crate) const GATE_HALF_STEPS: usize = 700;

pub(crate) const ANALYSIS_FREQUENCIES: [f64; 8] =
    [0.6e6, 0.9e6, 1.2e6, 1.6e6, 2.2e6, 3.0e6, 3.8e6, 4.6e6];
pub(crate) const ALPHA0_DB: [f64; 3] = [0.25, 0.5, 0.75];
pub(crate) const GAMMAS: [f64; 5] = [0.4, 0.7, 1.0, 1.3, 1.6];

pub(crate) fn alpha_np_m(alpha0_db: f64) -> f64 {
    alpha0_db * NEPER_PER_DB * 100.0
}

/// CFL-limited step for the pseudospectral Nyquist wavenumber.
pub(crate) fn time_step() -> f64 {
    0.4 * DX / (std::f64::consts::PI * C0)
}

/// Gaussian-modulated sine excitation, zero-padded by the solver.
pub(crate) fn excitation(dt: f64) -> Vec<f64> {
    let t0 = 3.0 * PULSE_WIDTH_S;
    let n_active = ((6.0 * PULSE_WIDTH_S) / dt).ceil() as usize;
    (0..n_active.min(STEPS))
        .map(|k| {
            let t = k as f64 * dt - t0;
            (-(t / PULSE_WIDTH_S).powi(2)).exp() * (TAU * PULSE_CENTRE_HZ * t).sin()
        })
        .collect()
}
