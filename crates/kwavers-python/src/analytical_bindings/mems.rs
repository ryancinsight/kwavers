//! PyO3 bindings for `kwavers_transducer::mems` (CMUT / PMUT / IVUS comparison).
//!
//! Physics in Rust; these thin wrappers expose the scalar models so the
//! `ch33_cmut_vs_pmut.py` figure script can plot without re-implementing physics.

use kwavers_transducer::mems::{cmut::CmutCell, comparison, plate, pmut::{PiezoFilm, PmutCell}};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn parse_film(name: &str) -> PyResult<PiezoFilm> {
    match name.to_ascii_lowercase().as_str() {
        "aln" => Ok(PiezoFilm::Aln),
        "pzt" => Ok(PiezoFilm::Pzt),
        other => Err(PyValueError::new_err(format!("unknown piezo film '{other}' (use 'aln' or 'pzt')"))),
    }
}

fn cmut(radius: f64, thickness: f64, gap: f64) -> PyResult<CmutCell> {
    CmutCell::silicon(radius, thickness, gap)
        .ok_or_else(|| PyValueError::new_err("invalid CMUT geometry (all dimensions must be > 0)"))
}

fn pmut(film: &str, radius: f64, t_p: f64, t_s: f64) -> PyResult<PmutCell> {
    PmutCell::new(radius, t_p, t_s, parse_film(film)?)
        .ok_or_else(|| PyValueError::new_err("invalid PMUT geometry (all dimensions must be > 0)"))
}

/// Clamped circular plate in-vacuo fundamental resonance [Hz].
#[pyfunction]
pub fn mems_clamped_plate_resonance(youngs: f64, thickness: f64, poisson: f64, density: f64, radius: f64) -> f64 {
    plate::vacuum_resonance(youngs, thickness, poisson, density, radius)
}

/// Lamb fluid-loaded (immersion) resonance [Hz].
#[pyfunction]
pub fn mems_immersion_resonance(vacuum_freq: f64, density_plate: f64, thickness: f64, density_fluid: f64, radius: f64) -> f64 {
    plate::immersion_resonance(vacuum_freq, density_plate, thickness, density_fluid, radius)
}

/// CMUT (Si) immersion resonance [Hz].
#[pyfunction]
pub fn cmut_resonance_immersion(radius: f64, thickness: f64, gap: f64, density_fluid: f64) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.immersion_resonance(density_fluid))
}

/// CMUT collapse (pull-in) voltage [V].
#[pyfunction]
pub fn cmut_collapse_voltage(radius: f64, thickness: f64, gap: f64) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.collapse_voltage())
}

/// CMUT bias-dependent electromechanical coupling k² [-].
#[pyfunction]
pub fn cmut_coupling_k2(radius: f64, thickness: f64, gap: f64, bias_voltage: f64) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.coupling_k2(bias_voltage))
}

/// CMUT dielectric self-heating power [W].
#[pyfunction]
pub fn cmut_self_heating(radius: f64, thickness: f64, gap: f64, v_ac: f64, freq: f64) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.self_heating_power(v_ac, freq))
}

/// CMUT fractional bandwidth from fluid loading [-].
#[pyfunction]
pub fn cmut_fractional_bandwidth(radius: f64, thickness: f64, density_fluid: f64) -> PyResult<f64> {
    // gap does not affect bandwidth; use a nominal value for construction
    Ok(cmut(radius, thickness, 0.1e-6)?.fractional_bandwidth(density_fluid))
}

/// PMUT immersion resonance [Hz] (film = "aln" | "pzt").
#[pyfunction]
pub fn pmut_resonance_immersion(film: &str, radius: f64, t_p: f64, t_s: f64, density_fluid: f64) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.immersion_resonance(density_fluid))
}

/// PMUT effective electromechanical coupling k² [-].
#[pyfunction]
pub fn pmut_coupling_k2(film: &str, radius: f64, t_p: f64, t_s: f64) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.coupling_k2())
}

/// PMUT dielectric self-heating power [W].
#[pyfunction]
pub fn pmut_self_heating(film: &str, radius: f64, t_p: f64, t_s: f64, v_ac: f64, freq: f64) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.self_heating_power(v_ac, freq))
}

/// PMUT fractional bandwidth from fluid loading [-].
#[pyfunction]
pub fn pmut_fractional_bandwidth(film: &str, radius: f64, t_p: f64, t_s: f64, density_fluid: f64) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.fractional_bandwidth(density_fluid))
}

/// CMUT gap-limited peak output pressure [Pa] (swing_fraction ≈ 1/3 conventional).
#[pyfunction]
pub fn cmut_max_output_pressure(radius: f64, thickness: f64, gap: f64, density_fluid: f64, sound_speed_fluid: f64, swing_fraction: f64) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.max_output_pressure(density_fluid, sound_speed_fluid, swing_fraction))
}

/// CMUT output derating when flexed to curvature `curvature` [1/m].
#[pyfunction]
pub fn cmut_flex_gap_derating(radius: f64, thickness: f64, gap: f64, curvature: f64) -> PyResult<f64> {
    Ok(cmut(radius, thickness, gap)?.flex_gap_derating(curvature))
}

/// PMUT drive-scaled peak output pressure [Pa] (film = "aln" | "pzt").
#[pyfunction]
pub fn pmut_max_output_pressure(film: &str, radius: f64, t_p: f64, t_s: f64, drive_voltage: f64, density_fluid: f64, sound_speed_fluid: f64) -> PyResult<f64> {
    Ok(pmut(film, radius, t_p, t_s)?.max_output_pressure(drive_voltage, density_fluid, sound_speed_fluid))
}

/// Therapy comparison. Returns
/// `[cmut_output_pa, pmut_output_pa, cmut_flex_derating, cmut_heating,
/// pmut_heating, recommended]` (recommended 0=CMUT, 1=PMUT).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn therapy_figure_of_merit(
    cmut_radius: f64,
    cmut_thickness: f64,
    cmut_gap: f64,
    pmut_film: &str,
    pmut_radius: f64,
    pmut_t_p: f64,
    pmut_t_s: f64,
    fluid_density: f64,
    fluid_sound_speed: f64,
    cmut_swing_fraction: f64,
    pmut_drive_voltage: f64,
    curvature: f64,
    substrate_output_factor: f64,
) -> PyResult<Vec<f64>> {
    let c = cmut(cmut_radius, cmut_thickness, cmut_gap)?;
    let p = pmut(pmut_film, pmut_radius, pmut_t_p, pmut_t_s)?;
    let v = comparison::evaluate_therapy(
        &c, &p, fluid_density, fluid_sound_speed, cmut_swing_fraction, pmut_drive_voltage, curvature, substrate_output_factor,
    );
    let recommended = if v.recommended == comparison::MutKind::Cmut { 0.0 } else { 1.0 };
    Ok(vec![
        v.cmut_output_pa,
        v.pmut_output_pa,
        v.cmut_flex_derating,
        v.cmut_heating,
        v.pmut_heating,
        recommended,
    ])
}

/// IVUS figure-of-merit comparison.
///
/// Returns `[cmut_fbw, pmut_fbw, cmut_heating, pmut_heating, cmut_drive_v,
/// pmut_drive_v, cmut_fom, pmut_fom, recommended]` where `recommended` is
/// `0.0` for CMUT and `1.0` for PMUT.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ivus_figure_of_merit(
    cmut_radius: f64,
    cmut_thickness: f64,
    cmut_gap: f64,
    pmut_film: &str,
    pmut_radius: f64,
    pmut_t_p: f64,
    pmut_t_s: f64,
    fluid_density: f64,
    pmut_drive_voltage: f64,
) -> PyResult<Vec<f64>> {
    let c = cmut(cmut_radius, cmut_thickness, cmut_gap)?;
    let p = pmut(pmut_film, pmut_radius, pmut_t_p, pmut_t_s)?;
    let v = comparison::evaluate_ivus(&c, &p, fluid_density, pmut_drive_voltage, comparison::IvusWeights::default());
    let recommended = if v.recommended == comparison::MutKind::Cmut { 0.0 } else { 1.0 };
    Ok(vec![
        v.cmut_fbw,
        v.pmut_fbw,
        v.cmut_heating,
        v.pmut_heating,
        v.cmut_drive_voltage,
        v.pmut_drive_voltage,
        v.cmut_fom,
        v.pmut_fom,
        recommended,
    ])
}
