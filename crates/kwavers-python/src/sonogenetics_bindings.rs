//! Python bindings for the kwavers sonogenetics neuromodulation module.
//!
//! All wave physics, membrane mechanics, channel gating, and neuron ODE stepping
//! execute in Rust.  Python receives pre-computed numpy arrays; no physics
//! computation appears in this file.
//!
//! # Exported functions
//!
//! | Python name | Rust kernel | Reference |
//! |---|---|---|
//! | `compute_acoustic_membrane_tension_py` | `pressure_to_membrane_tension_mn_m` | Timoshenko 1959; Sarvazyan 2010 |
//! | `boltzmann_open_probability_py` | `boltzmann_open_probability_from_tension_mn_m` | Sukharev 1997; Cox 2016 |
//! | `coupled_channel_drive_py` | `coupled_channel_drive` | Hille 2001 |
//! | `gaussian_beam_pressure_field_py` | `gaussian_beam_pressure_field` | Goodman 2005 §3.3 |
//! | `simulate_lif_neuron_py` | `simulate_lif_trace` | Koch 1999 |
//! | `lif_response_probability_py` | `lif_response_probability` | Koch 1999 |
//!
//! # Physics contract
//!
//! No physics — wave propagation, membrane mechanics, channel gating, or ODE
//! stepping — is implemented in this file.  Every computation is delegated to
//! `kwavers_physics::acoustics::therapy::sonogenetics`.
//!
//! # References
//!
//! - Timoshenko, S.P. & Woinowsky-Krieger, S. (1959). *Theory of Plates and Shells*. McGraw-Hill.
//! - Sarvazyan, A.P. et al. (2010). Acoustic radiation force — a review. *Curr. Med. Imaging Rev.*, 6(1), 15-25.
//! - Sukharev, S.I. et al. (1997). Mechanosensitive channel MscL in E. coli. *Biophys. J.*, 72, 193-203.
//! - Cox, C.D. et al. (2016). PIEZO1 gated by bilayer tension. *Nat. Commun.*, 7, 10366.
//! - Hille, B. (2001). *Ion Channels of Excitable Membranes*, 3rd ed. Sinauer.
//! - Goodman, J.W. (2005). *Introduction to Fourier Optics*, 3rd ed. §3.3.
//! - Koch, C. (1999). *Biophysics of Computation*. Oxford University Press.

use crate::array_utils::leto3_to_pyarray3;
use kwavers_physics::acoustics::therapy::sonogenetics::{
    boltzmann_open_probability_from_tension_mn_m, coupled_channel_drive,
    gaussian_beam_pressure_field, lif_response_probability, pressure_threshold_p_open,
    pressure_to_membrane_tension_mn_m, simulate_lif_trace, LifParams, PressureThresholdParams,
};
use leto::Array3;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn kwavers_to_py(err: kwavers_core::error::KwaversError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(err.to_string())
}

// ── Membrane tension ──────────────────────────────────────────────────────────

/// Compute per-sample acoustic membrane tension from peak pressure [mN/m].
///
/// # Derivation (Timoshenko 1959 §1.1; Sarvazyan 2010 Eq. 3)
///
/// For a progressive plane wave the acoustic radiation pressure is:
///
/// ```text
/// P_rad = I / c = p² / (2ρc²)   `Pa`
/// ```
///
/// Laplace thin-shell equilibrium for a spherical cell of radius R:
///
/// ```text
/// ΔT_membrane = P_rad · R / 2 = p² · R / (4ρc²)   [N/m]
/// ```
///
/// Output is scaled to mN/m (×10³) for direct comparison with published
/// literature values (Sukharev 1997; Cox 2016).
///
/// # Arguments
///
/// - `pressure_pa`: 1-D array of peak acoustic pressure `Pa`
/// - `density_kg_m3`: medium density ρ [kg/m³]
/// - `sound_speed_m_s`: medium sound speed c [m/s]
/// - `cell_radius_m`: cell soma radius R `m`
///
/// # Returns
///
/// 1-D numpy array of membrane tension [mN/m], same length as input.
#[pyfunction]
pub fn compute_acoustic_membrane_tension_py<'py>(
    py: Python<'py>,
    pressure_pa: PyReadonlyArray1<'py, f64>,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    cell_radius_m: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let pressure = pressure_pa.as_slice()?;
    let tension_mn_m = py
        .detach(|| {
            pressure_to_membrane_tension_mn_m(
                pressure,
                density_kg_m3,
                sound_speed_m_s,
                cell_radius_m,
            )
        })
        .map_err(kwavers_to_py)?;
    Ok(PyArray1::from_vec(py, tension_mn_m))
}

// ── Channel open probability ──────────────────────────────────────────────────

/// Compute Boltzmann open probability from membrane tension using slope parameterisation.
///
/// # Two-state Boltzmann model (Sukharev 1997; Cox 2016)
///
/// ```text
/// P_open = 1 / (1 + exp(-A · (ΔT − T_half) / (k_B · θ)))
/// ```
///
/// where `A` is the in-plane gating area `m²` and `θ` is the absolute
/// temperature `K`.  Phenomenological fits typically report a logistic slope
/// `σ = k_B θ / A` [N/m] rather than `A` directly.  This binding accepts
/// `slope_mn_m` [mN/m] and derives `A = k_B θ / (slope_mn_m × 10⁻³)`.
///
/// # Arguments
///
/// - `tension_mn_m`: 1-D membrane tension array [mN/m]
/// - `half_tension_mn_m`: tension at half-maximum activation T_half [mN/m]
/// - `slope_mn_m`: logistic slope in tension units σ [mN/m]
/// - `temperature_k`: absolute temperature θ `K`
///
/// # Returns
///
/// 1-D numpy array of open probabilities P_open ∈ [0, 1].
#[pyfunction]
pub fn boltzmann_open_probability_py<'py>(
    py: Python<'py>,
    tension_mn_m: PyReadonlyArray1<'py, f64>,
    half_tension_mn_m: f64,
    slope_mn_m: f64,
    temperature_k: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let tension = tension_mn_m.as_slice()?;
    let p_open = py
        .detach(|| {
            boltzmann_open_probability_from_tension_mn_m(
                tension,
                half_tension_mn_m,
                slope_mn_m,
                temperature_k,
            )
        })
        .map_err(kwavers_to_py)?;
    Ok(PyArray1::from_vec(py, p_open))
}

/// Compute pressure-threshold open probability from acoustic radiation pressure.
///
/// # Model
///
/// ```text
/// P_open = 1 / (1 + exp(-(P_rad - P_half) / s))
/// ```
///
/// This is the hsTRPA1 pressure-gating branch used by
/// `kwavers_physics::acoustics::therapy::sonogenetics::channels::gating`.
/// The binding performs only array shape conversion; the gate equation and
/// parameter validation execute in Rust core.
///
/// # Arguments
///
/// - `radiation_pressure_pa`: 1-D acoustic radiation pressure `Pa`
/// - `half_pressure_pa`: half-activation radiation pressure P_half `Pa`
/// - `steepness_pa`: positive sigmoid steepness s `Pa`
///
/// # Returns
///
/// 1-D numpy array of open probabilities P_open in [0, 1].
#[pyfunction]
pub fn pressure_threshold_open_probability_py<'py>(
    py: Python<'py>,
    radiation_pressure_pa: PyReadonlyArray1<'py, f64>,
    half_pressure_pa: f64,
    steepness_pa: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let pressure = radiation_pressure_pa.as_slice()?;
    let field = Array3::from_shape_vec((pressure.len(), 1, 1), pressure.to_vec())
        .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
    let params = PressureThresholdParams {
        half_pressure_pa,
        steepness_pa,
        single_channel_conductance_s: 0.0,
        reversal_potential_v: 0.0,
    };
    let p_open = py
        .detach(|| pressure_threshold_p_open(&field, &params))
        .map_err(kwavers_to_py)?;
    Ok(PyArray1::from_vec(py, p_open.iter().copied().collect()))
}

// ── Coupled channel drive ─────────────────────────────────────────────────────

/// Compute normalised mechanochemical channel drive from acoustic pressure.
///
/// Applies the full sonogenetics pipeline to each pressure sample:
///
/// ```text
/// 1. p → I = p² / (2ρc)                    [W/m²]
/// 2. I → ΔT = I · R / (2c)                 [N/m]  (Laplace thin-shell)
/// 3. ΔT → P_open,k = Boltzmann(ΔT; T_half,k, slope_k, θ)
/// 4. drive = clamp(Σ_k w_k · P_open,k / Σ_k |w_k|, −1, 1)
/// ```
///
/// The normalisation Σ_k |w_k| ensures drive ∈ [−1, 1] regardless of the number
/// of channels.  Negative weights encode inhibitory (K⁺ leak) channels.
///
/// # Arguments
///
/// - `pressure_pa`: 1-D pressure time series or parameter sweep `Pa`
/// - `half_tensions_mn_m`: per-channel half-activation tension T_half [mN/m]
/// - `slopes_mn_m`: per-channel logistic slope σ [mN/m]
/// - `conductance_weights`: signed conductance weight w_k (dimensionless)
/// - `density_kg_m3`: medium density ρ [kg/m³]
/// - `sound_speed_m_s`: medium sound speed c [m/s]
/// - `cell_radius_m`: cell soma radius R `m`
/// - `temperature_k`: absolute temperature θ `K`
///
/// # Returns
///
/// 1-D numpy array of normalised drive ∈ [−1, 1].
///
/// # Errors
///
/// Returns `ValueError` if channel parameter vectors have unequal length,
/// `temperature_k ≤ 0`, or any `slopes_mn_m`K` ≤ 0`.
#[pyfunction]
pub fn coupled_channel_drive_py<'py>(
    py: Python<'py>,
    pressure_pa: PyReadonlyArray1<'py, f64>,
    half_tensions_mn_m: Vec<f64>,
    slopes_mn_m: Vec<f64>,
    conductance_weights: Vec<f64>,
    density_kg_m3: f64,
    sound_speed_m_s: f64,
    cell_radius_m: f64,
    temperature_k: f64,
) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let pressure = pressure_pa.as_slice()?;
    let drive = py
        .detach(|| {
            coupled_channel_drive(
                pressure,
                &half_tensions_mn_m,
                &slopes_mn_m,
                &conductance_weights,
                density_kg_m3,
                sound_speed_m_s,
                cell_radius_m,
                temperature_k,
            )
        })
        .map_err(kwavers_to_py)?;
    Ok(PyArray1::from_vec(py, drive))
}

// ── Gaussian beam ─────────────────────────────────────────────────────────────

/// Generate an analytical 3-D paraxial Gaussian beam pressure field.
///
/// # Formula (Goodman 2005 §3.3)
///
/// ```text
/// P(x, y, z) = P_peak · exp(−(r² / (2σ_lat²) + z² / (2σ_ax²)))
/// ```
///
/// where `r² = x² + y²` and `σ = FWHM / (2√(2 ln 2))`.  Axes are centred at
/// the array midpoint with uniform spacing `(dx_m, dy_m, dz_m)`.
///
/// # Arguments
///
/// - `nx`, `ny`, `nz`: grid dimensions [voxels]
/// - `dx_m`, `dy_m`, `dz_m`: voxel spacing `m`
/// - `peak_pressure_pa`: peak positive pressure at focus P_peak `Pa`
/// - `lateral_fwhm_m`: lateral FWHM of the focal zone `m`
/// - `axial_fwhm_m`: axial FWHM of the focal zone `m`
///
/// # Returns
///
/// dict with keys:
/// - `x`, `y`, `z`: 3-D coordinate arrays `m`, shape (nx, ny, nz)
/// - `pressure`: 3-D pressure amplitude field `Pa`, shape (nx, ny, nz)
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn gaussian_beam_pressure_field_py<'py>(
    py: Python<'py>,
    nx: usize,
    ny: usize,
    nz: usize,
    dx_m: f64,
    dy_m: f64,
    dz_m: f64,
    peak_pressure_pa: f64,
    lateral_fwhm_m: f64,
    axial_fwhm_m: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let field = py
        .detach(|| {
            gaussian_beam_pressure_field(
                nx,
                ny,
                nz,
                dx_m,
                dy_m,
                dz_m,
                peak_pressure_pa,
                lateral_fwhm_m,
                axial_fwhm_m,
            )
        })
        .map_err(kwavers_to_py)?;
    let dict = PyDict::new(py);
    dict.set_item("x", leto3_to_pyarray3(py, field.x_m)?)?;
    dict.set_item("y", leto3_to_pyarray3(py, field.y_m)?)?;
    dict.set_item("z", leto3_to_pyarray3(py, field.z_m)?)?;
    dict.set_item("pressure", leto3_to_pyarray3(py, field.pressure_pa)?)?;
    Ok(dict)
}

// ── LIF neuron ────────────────────────────────────────────────────────────────

/// Simulate a leaky integrate-and-fire (LIF) neuron driven by an ion current trace.
///
/// # Model (Koch 1999 Table 1.1)
///
/// ```text
/// C_m · dV/dt = −G_leak · (V − E_leak) + I_ion(t)
/// ```
///
/// Forward-Euler discretisation with spike-and-reset:
///
/// ```text
/// V[n+1] = V`N` + (dt/C_m) · (−G_leak·(V`N`−E_leak) + I_ion`N`)
/// if V[n+1] ≥ V_thresh → spike, V ← V_reset, clamp for τ_ref
/// ```
///
/// # Arguments
///
/// - `i_ion_a`: ion current time series [A]; length N defines the simulation duration
/// - `dt_s`: uniform time step `s`; must be strictly positive
/// - `capacitance_f`: membrane capacitance C_m [F]  (default 100 pF, Koch 1999)
/// - `leak_conductance_s`: leak conductance G_leak `s`  (default 10 nS)
/// - `leak_reversal_v`: leak reversal potential E_leak `V`  (default −65 mV)
/// - `threshold_v`: spike threshold V_thresh `V`  (default −55 mV)
/// - `reset_v`: post-spike reset voltage V_reset `V`  (default −65 mV)
/// - `refractory_s`: absolute refractory period τ_ref `s`  (default 2 ms)
///
/// # Returns
///
/// dict with keys:
/// - `voltage_v`: membrane voltage trace `V`, length N
/// - `spike_times_s`: chronological spike times `s`
/// - `spike_count`: total number of spikes (int)
#[pyfunction]
#[pyo3(signature = (
    i_ion_a,
    dt_s,
    capacitance_f = 100.0e-12,
    leak_conductance_s = 10.0e-9,
    leak_reversal_v = -65.0e-3,
    threshold_v = -55.0e-3,
    reset_v = -65.0e-3,
    refractory_s = 2.0e-3,
))]
pub fn simulate_lif_neuron_py<'py>(
    py: Python<'py>,
    i_ion_a: PyReadonlyArray1<'py, f64>,
    dt_s: f64,
    capacitance_f: f64,
    leak_conductance_s: f64,
    leak_reversal_v: f64,
    threshold_v: f64,
    reset_v: f64,
    refractory_s: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let current = i_ion_a.as_slice()?;
    let trace = py
        .detach(|| {
            let params = LifParams {
                capacitance_f,
                leak_conductance_s,
                leak_reversal_v,
                threshold_v,
                reset_v,
                refractory_s,
            };
            simulate_lif_trace(current, dt_s, params)
        })
        .map_err(kwavers_to_py)?;
    let spike_count = trace.spike_times_s.len();
    let dict = PyDict::new(py);
    dict.set_item("voltage_v", PyArray1::from_vec(py, trace.voltage_v))?;
    dict.set_item("spike_times_s", PyArray1::from_vec(py, trace.spike_times_s))?;
    dict.set_item("spike_count", spike_count)?;
    Ok(dict)
}

/// Convert LIF spike times into a Gaussian-smoothed response probability.
///
/// The Rust core samples the spike train, applies the normalized Gaussian
/// smoothing kernel, and clamps the response to `[0, 1]` after normalizing by
/// the theoretical maximum firing rate.
#[pyfunction]
#[pyo3(signature = (spike_times_s, n_samples, dt_s, smoothing_sigma_s, f_max_hz))]
pub fn lif_response_probability_py<'py>(
    py: Python<'py>,
    spike_times_s: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    dt_s: f64,
    smoothing_sigma_s: f64,
    f_max_hz: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let spikes = spike_times_s.as_slice()?;
    let response = py
        .detach(|| lif_response_probability(spikes, n_samples, dt_s, smoothing_sigma_s, f_max_hz))
        .map_err(kwavers_to_py)?;
    let dict = PyDict::new(py);
    dict.set_item("spike_train", PyArray1::from_vec(py, response.spike_train))?;
    dict.set_item(
        "response_probability",
        PyArray1::from_vec(py, response.response_probability),
    )?;
    Ok(dict)
}

// ── Module registration ───────────────────────────────────────────────────────

pub fn register_sonogenetics(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(
        compute_acoustic_membrane_tension_py,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(boltzmann_open_probability_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        pressure_threshold_open_probability_py,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(coupled_channel_drive_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(gaussian_beam_pressure_field_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(simulate_lif_neuron_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(lif_response_probability_py, m)?)?;
    Ok(())
}
