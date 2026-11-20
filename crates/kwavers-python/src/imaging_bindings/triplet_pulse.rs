//! PyO3 wrappers for the triplet-pulse imaging capability (Rust port step 5,
//! KW-IMG-046).
//!
//! Thin bindings only: the sequence lives in `kwavers-imaging`, the
//! bubble-cloud coupling in `kwavers-simulation`, and the envelope DAS +
//! contrast in `kwavers-analysis`. Here we only marshal the Python call:
//! validate the arguments, delegate, and return NumPy arrays.
//!
//! Scope limits (kept explicit at the boundary, as in the Rust modules):
//! the cloud is forced to exist at every PNP (no nucleation physics), the
//! default thermal path is OFF to match the prototype's adiabatic KM solve,
//! and the contrast is the prototype's B-mode proxy metric — not Kuji
//! 2024's 3P/1P-distribution metric. The 2.45 MHz contrast peak is
//! proxy-only (D2: Kuji's own metric shows no peak); thermal-on ratios
//! are observed values, never parity targets.

use crate::array_utils::{leto2_to_pyarray2, pyarray2_to_leto2, vec_to_pyarray1};
use kwavers_analysis::signal_processing::triplet_pulse::{
    combine_methods, envelope_das, triplet_pulse_contrast, ImageRoi, TripletPulseGeometry,
};
use kwavers_imaging::ultrasound::triplet_pulse::TripletPulseConfig;
use kwavers_physics::acoustics::bubble_dynamics::acoustic_pressure::PressureAmplitude;
use kwavers_signal::FrequencyFilter;
use kwavers_simulation::imaging::triplet_pulse::{
    three_p_over_one_p_ratio, TripletPulseCloud, TripletPulseDrive,
};
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

#[cfg(test)]
mod tests;

fn kwavers_to_py(err: kwavers_core::error::KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("triplet-pulse failed: {err}"))
}

/// 3P/1P magnitude ratios for one bubble at the 1.5th fractional harmonic
/// and at the fundamental.
///
/// Parameters
/// ----------
/// r0_um : equilibrium bubble radius [µm]
/// pnp_mpa : imaging-pulse peak-negative pressure at the cloud [MPa]
/// frequency_hz : imaging fundamental [Hz] (default 1.74e6, Iwasaki 2018)
/// steps_per_cycle : KM solver steps per drive cycle (default 500)
/// thermal : native thermal/mass-transfer path (default False = thermal-off,
///     matching the prototype's adiabatic KM solve)
/// accommodation_coeff : Hertz–Knudsen accommodation coefficient
///     (default 0.04)
///
/// Returns
/// -------
/// (ratio_1p5f0, ratio_f0) : 3P/1P magnitude ratios at 1.5·f0 and f0.
/// Parity targets (thermal-off, 0.26 MPa, 1.74 MHz): 1.5f0 = 0.35 / 0.36 /
/// 0.22 for R0 = 2 / 3 / 5 µm (±0.05); f0 < 0.05. Thermal-on ratios are
/// observed values (1.5f0 = 0.248 / 0.302 / 0.143), not targets.
#[pyfunction]
#[pyo3(signature = (
    r0_um, pnp_mpa, frequency_hz = 1.74e6, steps_per_cycle = 500,
    thermal = false, accommodation_coeff = 0.04,
))]
pub fn triplet_pulse_ratio(
    py: Python<'_>,
    r0_um: f64,
    pnp_mpa: f64,
    frequency_hz: f64,
    steps_per_cycle: usize,
    thermal: bool,
    accommodation_coeff: f64,
) -> PyResult<(f64, f64)> {
    if !r0_um.is_finite() || r0_um <= 0.0 {
        return Err(PyValueError::new_err("r0_um must be finite and > 0"));
    }
    if !pnp_mpa.is_finite() || pnp_mpa < 0.0 {
        return Err(PyValueError::new_err("pnp_mpa must be finite and >= 0"));
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(PyValueError::new_err("frequency_hz must be finite and > 0"));
    }
    if steps_per_cycle == 0 {
        return Err(PyValueError::new_err("steps_per_cycle must be > 0"));
    }
    let amplitude = PressureAmplitude::from_pascals(pnp_mpa * 1.0e6);
    let (r15, rf0) = py
        .detach(|| {
            let r15 = three_p_over_one_p_ratio(
                r0_um * 1.0e-6,
                amplitude,
                frequency_hz,
                1.5 * frequency_hz,
                steps_per_cycle,
                thermal,
                accommodation_coeff,
            );
            let rf0 = three_p_over_one_p_ratio(
                r0_um * 1.0e-6,
                amplitude,
                frequency_hz,
                frequency_hz,
                steps_per_cycle,
                thermal,
                accommodation_coeff,
            );
            match (r15, rf0) {
                (Ok(a), Ok(b)) => Ok((a, b)),
                (Err(e), _) | (_, Err(e)) => Err(e),
            }
        })
        .map_err(kwavers_to_py)?;
    Ok((r15, rf0))
}

/// Build a triplet-pulse cloud for the given imaging frequency.
///
/// The Iwasaki (1.74 MHz) and Kuji (2.45 MHz) configurations use their
/// named constructors; any other frequency builds the same 12-cycle
/// flat-top drive (steady-state window cycles 6–10, segment 6 cycles)
/// with the default 0°/120°/240°/180° phases.
fn build_cloud(frequency_hz: f64, pnp_mpa: f64, thermal: bool) -> TripletPulseCloud {
    let amplitude = PressureAmplitude::from_pascals(pnp_mpa * 1.0e6);
    let mut cloud = if (frequency_hz - 1.74e6).abs() < 1.0 {
        TripletPulseCloud::iwasaki_2018(amplitude)
    } else {
        let config = if (frequency_hz - 2.45e6).abs() < 1.0 {
            TripletPulseConfig::kuji_2024()
        } else {
            TripletPulseConfig {
                frequency_hz,
                ..TripletPulseConfig::kuji_2024()
            }
        };
        TripletPulseCloud {
            config,
            drive: TripletPulseDrive {
                frequency_hz,
                amplitude_pa: amplitude,
                solve_cycles: 12,
                steady_start_cycle: 6,
                steady_end_cycle: 10,
                segment_cycles: 6,
            },
            density: 998.0,
            sound_speed: 1481.0,
            bubbles: Vec::new(),
            thermal: false,
        }
    };
    if thermal {
        cloud = cloud.with_thermal(true);
    }
    cloud
}

/// Four phase-transmission RF traces at one receiver from a forced bubble
/// cloud.
///
/// Parameters
/// ----------
/// bubble_positions_m : (n_bubbles, 3) bubble positions [m]
/// r0_um : equilibrium bubble radius [µm] (one radius for the whole cloud;
///     the prototype's clouds are single-radius)
/// pnp_mpa : imaging-pulse peak-negative pressure at the cloud [MPa]
/// receiver_m : receiver position [m] (length 3)
/// frequency_hz : imaging fundamental [Hz] (default 1.74e6)
/// steps_per_cycle : KM solver steps per drive cycle (default 500)
/// thermal : native thermal/mass-transfer path (default False)
///
/// Returns
/// -------
/// (ph0, ph120, ph240, ph180, dt_s) : the four phase traces [Pa] on the
/// receiver's clock (extended to cover the travel-time delay, windowed
/// afterwards — see the D1 fix) and the trace sampling interval [s].
/// The traces are on the KM grid (dt = 1 / (f0 · steps_per_cycle)), not
/// the prototype's 30 ns PSTD grid: compare peaks / band amplitudes,
/// not samples. Prototype reference: the a2b2 bubble-only RF peaks at
/// 1018.8 Pa (phase 0, R0 = 2 µm, 0.26 MPa); the Rust cloud peak is
/// within ±3 dB of it.
#[pyfunction]
#[pyo3(signature = (
    bubble_positions_m, r0_um, pnp_mpa, receiver_m,
    frequency_hz = 1.74e6, steps_per_cycle = 500, thermal = false,
))]
pub fn triplet_cloud_phase_traces(
    py: Python<'_>,
    bubble_positions_m: PyReadonlyArray2<f64>,
    r0_um: f64,
    pnp_mpa: f64,
    receiver_m: Vec<f64>,
    frequency_hz: f64,
    steps_per_cycle: usize,
    thermal: bool,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
)> {
    if !r0_um.is_finite() || r0_um <= 0.0 {
        return Err(PyValueError::new_err("r0_um must be finite and > 0"));
    }
    if !pnp_mpa.is_finite() || pnp_mpa < 0.0 {
        return Err(PyValueError::new_err("pnp_mpa must be finite and >= 0"));
    }
    if receiver_m.len() != 3 || !receiver_m.iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err(
            "receiver_m must be a length-3 list of finite values",
        ));
    }
    if !frequency_hz.is_finite() || frequency_hz <= 0.0 {
        return Err(PyValueError::new_err("frequency_hz must be finite and > 0"));
    }
    if steps_per_cycle == 0 {
        return Err(PyValueError::new_err("steps_per_cycle must be > 0"));
    }
    let positions = bubble_positions_m
        .as_array()
        .rows()
        .into_iter()
        .map(|row| {
            let v: Vec<f64> = row.iter().copied().collect();
            if v.len() != 3 || !v.iter().all(|x| x.is_finite()) {
                Err(PyValueError::new_err(
                    "bubble_positions_m must be (n, 3) finite",
                ))
            } else {
                Ok([v[0], v[1], v[2]])
            }
        })
        .collect::<PyResult<Vec<[f64; 3]>>>()?;
    if positions.is_empty() {
        return Err(PyValueError::new_err(
            "bubble_positions_m must contain at least one bubble",
        ));
    }
    let receiver = [receiver_m[0], receiver_m[1], receiver_m[2]];
    let dt_s = 1.0 / (frequency_hz * steps_per_cycle as f64);
    let (ph0, ph120, ph240, ph180) = py
        .detach(move || {
            let mut cloud = build_cloud(frequency_hz, pnp_mpa, thermal);
            for position in positions {
                cloud.add_bubble(position, r0_um * 1.0e-6);
            }
            let ph0 = cloud.simulate_phase(receiver, 0.0, steps_per_cycle);
            let ph120 = cloud.simulate_phase(receiver, 120.0, steps_per_cycle);
            let ph240 = cloud.simulate_phase(receiver, 240.0, steps_per_cycle);
            let ph180 = cloud.simulate_phase(receiver, 180.0, steps_per_cycle);
            match (ph0, ph120, ph240, ph180) {
                (Ok(a), Ok(b), Ok(c), Ok(d)) => Ok((a, b, c, d)),
                (Err(e), _, _, _) | (_, Err(e), _, _) | (_, _, Err(e), _) | (_, _, _, Err(e)) => {
                    Err(e)
                }
            }
        })
        .map_err(kwavers_to_py)?;
    Ok((
        vec_to_pyarray1(py, ph0),
        vec_to_pyarray1(py, ph120),
        vec_to_pyarray1(py, ph240),
        vec_to_pyarray1(py, ph180),
        dt_s,
    ))
}

/// Combine four phase-transmission RF channel records into 1P / PI / 3P.
///
/// Parameters
/// ----------
/// ph0, ph120, ph240, ph180 : (n_elem, n_samples) RF arrays [Pa]
///
/// Returns
/// -------
/// (one_pulse, pulse_inversion, triplet_pulse) : 1P = phase 0;
/// PI = mean(0, 180); 3P = mean(0, 120, 240).
#[pyfunction]
pub fn triplet_combine_methods(
    py: Python<'_>,
    ph0: PyReadonlyArray2<f64>,
    ph120: PyReadonlyArray2<f64>,
    ph240: PyReadonlyArray2<f64>,
    ph180: PyReadonlyArray2<f64>,
) -> PyResult<(Py<PyArray2<f64>>, Py<PyArray2<f64>>, Py<PyArray2<f64>>)> {
    let ph0 = pyarray2_to_leto2(&ph0)?;
    let ph120 = pyarray2_to_leto2(&ph120)?;
    let ph240 = pyarray2_to_leto2(&ph240)?;
    let ph180 = pyarray2_to_leto2(&ph180)?;
    let methods = combine_methods(ph0.view(), ph120.view(), ph240.view(), ph180.view())
        .map_err(kwavers_to_py)?;
    Ok((
        leto2_to_pyarray2(py, methods.one_pulse)?,
        leto2_to_pyarray2(py, methods.pulse_inversion)?,
        leto2_to_pyarray2(py, methods.triplet_pulse)?,
    ))
}

/// Zero-phase FFT brick-wall filter, applied per RF row.
///
/// Parameters
/// ----------
/// rf : (n_elem, n_samples) RF array
/// dt_s : sampling interval [s]
/// mode : "lowpass", "highpass", or "bandpass"
/// low_hz : cutoff [Hz] (lowpass/highpass) or band low edge [Hz]
/// high_hz : band high edge [Hz] (bandpass only)
///
/// Returns
/// -------
/// Filtered RF, same shape. The prototype's full-band filter is
/// lowpass at 4 MHz and its 1.5f0-band filter is bandpass 2.36–2.86 MHz.
#[pyfunction]
#[pyo3(signature = (rf, dt_s, mode, low_hz, high_hz = 0.0))]
pub fn triplet_filter_rf(
    py: Python<'_>,
    rf: PyReadonlyArray2<f64>,
    dt_s: f64,
    mode: &str,
    low_hz: f64,
    high_hz: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(PyValueError::new_err("dt_s must be finite and > 0"));
    }
    if !low_hz.is_finite() || low_hz < 0.0 {
        return Err(PyValueError::new_err("low_hz must be finite and >= 0"));
    }
    if mode == "bandpass" && (!high_hz.is_finite() || high_hz <= low_hz) {
        return Err(PyValueError::new_err(
            "bandpass needs 0 <= low_hz < high_hz",
        ));
    }
    if !matches!(mode, "lowpass" | "highpass" | "bandpass") {
        return Err(PyValueError::new_err(
            "mode must be 'lowpass', 'highpass', or 'bandpass'",
        ));
    }
    let rf = pyarray2_to_leto2(&rf)?;
    let filter = FrequencyFilter::new();
    let [n_rows, n_cols] = rf.shape();
    let mut out = leto::Array2::zeros([n_rows, n_cols]);
    for r in 0..n_rows {
        let row: Vec<f64> = (0..n_cols).map(|c| rf[[r, c]]).collect();
        let filtered = match mode {
            "lowpass" => filter.lowpass(&row, dt_s, low_hz),
            "highpass" => filter.highpass(&row, dt_s, low_hz),
            _ => filter.bandpass(&row, dt_s, low_hz, high_hz),
        }
        .map_err(kwavers_to_py)?;
        for (c, &v) in filtered.iter().enumerate() {
            out[[r, c]] = v;
        }
    }
    leto2_to_pyarray2(py, out)
}

/// Prototype-faithful envelope DAS image.
///
/// Parameters
/// ----------
/// rf : (n_elem, n_samples) RF array [Pa]
/// dt_s : sampling interval [s]
/// xs_m, ys_m : image grid, lateral / depth [m] (ascending)
/// aperture_m : array aperture [m] (default 10 mm, Stage 1)
/// domain_width_m : PSTD domain width [m] (default 38.4 mm, Stage 1)
/// element_depth_m : array-face depth [m] (default 2 mm, Stage 1)
/// sound_speed_m_s : DAS sound speed [m/s] (default 1540, tissue)
///
/// Returns
/// -------
/// (n_ys, n_xs) envelope image. Hilbert envelope first, nearest-sample
/// delays, point-source transmit from the array centre, divided by the
/// full element count — the prototype's algorithm, not the coherent RF
/// DAS in `kwavers-analysis`.
#[pyfunction]
#[pyo3(signature = (
    rf, dt_s, xs_m, ys_m, aperture_m = 10.0e-3, domain_width_m = 38.4e-3,
    element_depth_m = 2.0e-3, sound_speed_m_s = 1540.0,
))]
pub fn triplet_das_image(
    py: Python<'_>,
    rf: PyReadonlyArray2<f64>,
    dt_s: f64,
    xs_m: Vec<f64>,
    ys_m: Vec<f64>,
    aperture_m: f64,
    domain_width_m: f64,
    element_depth_m: f64,
    sound_speed_m_s: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    if !dt_s.is_finite() || dt_s <= 0.0 {
        return Err(PyValueError::new_err("dt_s must be finite and > 0"));
    }
    let rf = pyarray2_to_leto2(&rf)?;
    let geometry = TripletPulseGeometry::new(
        rf.shape()[0],
        aperture_m,
        domain_width_m,
        element_depth_m,
        sound_speed_m_s,
        dt_s,
    )
    .map_err(kwavers_to_py)?;
    let image = envelope_das(rf.view(), &geometry, &xs_m, &ys_m).map_err(kwavers_to_py)?;
    leto2_to_pyarray2(py, image)
}

/// Bubble-vs-tissue image contrast for one method's DAS image.
///
/// Parameters
/// ----------
/// image : (n_ys, n_xs) DAS image of the combined (bubble + tissue) run
/// baseline_image : (n_ys, n_xs) DAS image of the tissue-only run
/// xs_m, ys_m : image grid [m]
/// bubble_cx_m, bubble_cy_m, bubble_r_m : bubble-ROI disk [m]
///     (defaults: centre (19.2, 20) mm, r = 2 mm)
/// tissue_x_lo_m, tissue_x_hi_m, tissue_cy_m, tissue_half_depth_m :
///     tissue-ROI lateral band [m] (defaults: 28–33 mm, |y − 20| ≤ 2 mm)
///
/// Returns
/// -------
/// (mean_db, max_db, bubble_mean, bubble_max, tissue_mean) :
/// mean-bubble vs mean-tissue and max-bubble vs mean-tissue contrasts
/// [dB]. The mean statistic is the prototype's headline number; the max
/// statistic is Iwasaki 2018's own statistic — with matched statistics
/// the prototype does NOT reproduce the paper's ordering (see the Rust
/// module docs), so these numbers are prototype parity, not paper parity.
#[pyfunction]
#[pyo3(signature = (
    image, baseline_image, xs_m, ys_m,
    bubble_cx_m = 19.2e-3, bubble_cy_m = 20.0e-3, bubble_r_m = 2.0e-3,
    tissue_x_lo_m = 28.0e-3, tissue_x_hi_m = 33.0e-3,
    tissue_cy_m = 20.0e-3, tissue_half_depth_m = 2.0e-3,
))]
pub fn triplet_contrast(
    image: PyReadonlyArray2<f64>,
    baseline_image: PyReadonlyArray2<f64>,
    xs_m: Vec<f64>,
    ys_m: Vec<f64>,
    bubble_cx_m: f64,
    bubble_cy_m: f64,
    bubble_r_m: f64,
    tissue_x_lo_m: f64,
    tissue_x_hi_m: f64,
    tissue_cy_m: f64,
    tissue_half_depth_m: f64,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let image = pyarray2_to_leto2(&image)?;
    let baseline_image = pyarray2_to_leto2(&baseline_image)?;
    let bubble_roi = ImageRoi::Disk {
        center_x_m: bubble_cx_m,
        center_y_m: bubble_cy_m,
        radius_m: bubble_r_m,
    };
    let tissue_roi = ImageRoi::LateralBand {
        x_lo_m: tissue_x_lo_m,
        x_hi_m: tissue_x_hi_m,
        center_y_m: tissue_cy_m,
        half_depth_m: tissue_half_depth_m,
    };
    let contrast = triplet_pulse_contrast(
        image.view(),
        baseline_image.view(),
        &xs_m,
        &ys_m,
        &bubble_roi,
        &tissue_roi,
    )
    .map_err(kwavers_to_py)?;
    Ok((
        contrast.mean_db,
        contrast.max_db,
        contrast.bubble_mean,
        contrast.bubble_max,
        contrast.tissue_mean,
    ))
}

/// Register the triplet-pulse functions on the `pykwavers` module.
pub fn register_triplet(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(triplet_pulse_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(triplet_cloud_phase_traces, m)?)?;
    m.add_function(wrap_pyfunction!(triplet_combine_methods, m)?)?;
    m.add_function(wrap_pyfunction!(triplet_filter_rf, m)?)?;
    m.add_function(wrap_pyfunction!(triplet_das_image, m)?)?;
    m.add_function(wrap_pyfunction!(triplet_contrast, m)?)?;
    Ok(())
}
