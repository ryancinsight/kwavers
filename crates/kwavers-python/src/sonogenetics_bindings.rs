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
//! | `compute_acoustic_membrane_tension_py` | `compute_membrane_tension` | Timoshenko 1959; Sarvazyan 2010 |
//! | `boltzmann_open_probability_py` | `boltzmann_p_open` | Sukharev 1997; Cox 2016 |
//! | `coupled_channel_drive_py` | `boltzmann_p_open` (multi-channel) | Hille 2001 |
//! | `gaussian_beam_pressure_field_py` | Analytical Gaussian envelope | Goodman 2005 §3.3 |
//! | `simulate_lif_neuron_py` | `LifNeuron::step` | Koch 1999 |
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

use kwavers_core::constants::fundamental::BOLTZMANN as K_B;
use kwavers_physics::acoustics::therapy::sonogenetics::{
    boltzmann_p_open, compute_membrane_tension, BoltzmannGatingParams, CellMembraneParams,
    LifNeuron, LifParams,
};
use ndarray::{Array1, Array3};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

// ── Membrane tension ──────────────────────────────────────────────────────────

/// Compute per-sample acoustic membrane tension from peak pressure [mN/m].
///
/// # Derivation (Timoshenko 1959 §1.1; Sarvazyan 2010 Eq. 3)
///
/// For a progressive plane wave the acoustic radiation pressure is:
///
/// ```text
/// P_rad = I / c = p² / (2ρc²)   [Pa]
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
/// - `pressure_pa`: 1-D array of peak acoustic pressure [Pa]
/// - `density_kg_m3`: medium density ρ [kg/m³]
/// - `sound_speed_m_s`: medium sound speed c [m/s]
/// - `cell_radius_m`: cell soma radius R [m]
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
    let p = pressure_pa.as_array().to_owned();
    let n = p.len();
    let tension_mn_m = py.detach(|| {
        // Intensity: I = p² / (2·ρ·c)
        let intensity_1d: Array1<f64> =
            p.mapv(|pi| pi * pi / (2.0 * density_kg_m3 * sound_speed_m_s));
        // Reshape to (N, 1, 1) for Rust kernel
        let intensity_3d = intensity_1d
            .into_shape_with_order((n, 1, 1))
            .expect("1-D-to-(N,1,1) reshape is infallible");
        let c_3d = Array3::from_elem((n, 1, 1), sound_speed_m_s);
        let params = CellMembraneParams {
            radius_m: cell_radius_m,
            // Canonical lipid bilayer thickness h = 5 nm (Engelman 2005).
            thickness_m: 5.0e-9,
        };
        // ΔT in N/m → scale to mN/m
        let tension_3d = compute_membrane_tension(&intensity_3d, &c_3d, &params);
        tension_3d
            .into_shape_with_order(n)
            .expect("(N,1,1)-to-1-D reshape is infallible")
            .mapv(|t| t * 1.0e3)
    });
    Ok(tension_mn_m.into_pyarray(py))
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
/// where `A` is the in-plane gating area [m²] and `θ` is the absolute
/// temperature [K].  Phenomenological fits typically report a logistic slope
/// `σ = k_B θ / A` [N/m] rather than `A` directly.  This binding accepts
/// `slope_mn_m` [mN/m] and derives `A = k_B θ / (slope_mn_m × 10⁻³)`.
///
/// # Arguments
///
/// - `tension_mn_m`: 1-D membrane tension array [mN/m]
/// - `half_tension_mn_m`: tension at half-maximum activation T_half [mN/m]
/// - `slope_mn_m`: logistic slope in tension units σ [mN/m]
/// - `temperature_k`: absolute temperature θ [K]
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
    if temperature_k <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "temperature_k must be strictly positive",
        ));
    }
    if slope_mn_m <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "slope_mn_m must be strictly positive",
        ));
    }
    let t = tension_mn_m.as_array().to_owned();
    let n = t.len();
    // Derive gating area from slope parameterisation: A = k_B·θ / slope [m²]
    let gating_area_m2 = K_B * temperature_k / (slope_mn_m * 1.0e-3);
    let params = BoltzmannGatingParams {
        gating_area_m2,
        half_tension_n_per_m: half_tension_mn_m * 1.0e-3,
        // single_channel_conductance_s and reversal_potential_v are unused by
        // boltzmann_p_open; set to zero to satisfy struct completeness.
        single_channel_conductance_s: 0.0,
        reversal_potential_v: 0.0,
    };
    let p_open = py.detach(|| {
        // Convert mN/m → N/m for Rust kernel
        let t_si: Array1<f64> = t.mapv(|v| v * 1.0e-3);
        let t_3d = t_si
            .into_shape_with_order((n, 1, 1))
            .expect("1-D-to-(N,1,1) reshape is infallible");
        let out = boltzmann_p_open(&t_3d, &params, temperature_k)
            .expect("temperature_k > 0 is guaranteed by pre-call validation");
        out.into_shape_with_order(n)
            .expect("(N,1,1)-to-1-D reshape is infallible")
    });
    Ok(p_open.into_pyarray(py))
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
/// - `pressure_pa`: 1-D pressure time series or parameter sweep [Pa]
/// - `half_tensions_mn_m`: per-channel half-activation tension T_half [mN/m]
/// - `slopes_mn_m`: per-channel logistic slope σ [mN/m]
/// - `conductance_weights`: signed conductance weight w_k (dimensionless)
/// - `density_kg_m3`: medium density ρ [kg/m³]
/// - `sound_speed_m_s`: medium sound speed c [m/s]
/// - `cell_radius_m`: cell soma radius R [m]
/// - `temperature_k`: absolute temperature θ [K]
///
/// # Returns
///
/// 1-D numpy array of normalised drive ∈ [−1, 1].
///
/// # Errors
///
/// Returns `ValueError` if channel parameter vectors have unequal length,
/// `temperature_k ≤ 0`, or any `slopes_mn_m[k] ≤ 0`.
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
    let n_ch = half_tensions_mn_m.len();
    if slopes_mn_m.len() != n_ch || conductance_weights.len() != n_ch {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "half_tensions_mn_m, slopes_mn_m, and conductance_weights must have equal length",
        ));
    }
    if temperature_k <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "temperature_k must be strictly positive",
        ));
    }
    for (k, &slope) in slopes_mn_m.iter().enumerate() {
        if slope <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "slopes_mn_m[{k}] = {slope} must be strictly positive"
            )));
        }
    }
    let p = pressure_pa.as_array().to_owned();
    let n = p.len();
    let drive = py.detach(|| {
        // Step 1: intensity
        let intensity_1d: Array1<f64> =
            p.mapv(|pi| pi * pi / (2.0 * density_kg_m3 * sound_speed_m_s));
        let intensity_3d = intensity_1d
            .into_shape_with_order((n, 1, 1))
            .expect("1-D-to-(N,1,1) reshape is infallible");
        let c_3d = Array3::from_elem((n, 1, 1), sound_speed_m_s);
        let membrane_params = CellMembraneParams {
            radius_m: cell_radius_m,
            thickness_m: 5.0e-9,
        };
        // Step 2: membrane tension (N/m)
        let tension_3d = compute_membrane_tension(&intensity_3d, &c_3d, &membrane_params);
        // Step 3 + 4: accumulate weighted P_open over all channels
        let norm: f64 = conductance_weights.iter().map(|w| w.abs()).sum();
        let mut drive = Array1::<f64>::zeros(n);
        for k in 0..n_ch {
            let gating_area_m2 = K_B * temperature_k / (slopes_mn_m[k] * 1.0e-3);
            let gating_params = BoltzmannGatingParams {
                gating_area_m2,
                half_tension_n_per_m: half_tensions_mn_m[k] * 1.0e-3,
                single_channel_conductance_s: 0.0,
                reversal_potential_v: 0.0,
            };
            let p_open_3d = boltzmann_p_open(&tension_3d, &gating_params, temperature_k)
                .expect("temperature_k > 0 guaranteed by pre-call validation");
            let p_open_1d = p_open_3d
                .into_shape_with_order(n)
                .expect("(N,1,1)-to-1-D reshape is infallible");
            let w = conductance_weights[k];
            drive.zip_mut_with(&p_open_1d, |d, &p| *d += w * p);
        }
        if norm > 0.0 {
            drive.mapv_inplace(|d| (d / norm).clamp(-1.0, 1.0));
        }
        drive
    });
    Ok(drive.into_pyarray(py))
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
/// - `dx_m`, `dy_m`, `dz_m`: voxel spacing [m]
/// - `peak_pressure_pa`: peak positive pressure at focus P_peak [Pa]
/// - `lateral_fwhm_m`: lateral FWHM of the focal zone [m]
/// - `axial_fwhm_m`: axial FWHM of the focal zone [m]
///
/// # Returns
///
/// dict with keys:
/// - `x`, `y`, `z`: 3-D coordinate arrays [m], shape (nx, ny, nz)
/// - `pressure`: 3-D pressure amplitude field [Pa], shape (nx, ny, nz)
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
    if lateral_fwhm_m <= 0.0 || axial_fwhm_m <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "lateral_fwhm_m and axial_fwhm_m must be strictly positive",
        ));
    }
    // σ = FWHM / (2·√(2·ln2));  pre-compute reciprocals.
    let fwhm_to_sigma = (2.0 * (2.0_f64 * std::f64::consts::LN_2).sqrt()).recip();
    let sigma_lat = lateral_fwhm_m * fwhm_to_sigma;
    let sigma_ax = axial_fwhm_m * fwhm_to_sigma;
    let inv2sl2 = 0.5 / (sigma_lat * sigma_lat);
    let inv2sa2 = 0.5 / (sigma_ax * sigma_ax);
    let cx = (nx as f64 - 1.0) / 2.0;
    let cy = (ny as f64 - 1.0) / 2.0;
    let cz = (nz as f64 - 1.0) / 2.0;

    let (x_arr, y_arr, z_arr, pressure) = py.detach(|| {
        let mut x_arr = Array3::<f64>::zeros((nx, ny, nz));
        let mut y_arr = Array3::<f64>::zeros((nx, ny, nz));
        let mut z_arr = Array3::<f64>::zeros((nx, ny, nz));
        let mut pressure = Array3::<f64>::zeros((nx, ny, nz));
        for i in 0..nx {
            let xi = (i as f64 - cx) * dx_m;
            for j in 0..ny {
                let yj = (j as f64 - cy) * dy_m;
                let r2 = xi * xi + yj * yj;
                let lat_term = r2 * inv2sl2;
                for k in 0..nz {
                    let zk = (k as f64 - cz) * dz_m;
                    x_arr[[i, j, k]] = xi;
                    y_arr[[i, j, k]] = yj;
                    z_arr[[i, j, k]] = zk;
                    pressure[[i, j, k]] =
                        peak_pressure_pa * (-(lat_term + zk * zk * inv2sa2)).exp();
                }
            }
        }
        (x_arr, y_arr, z_arr, pressure)
    });
    let dict = PyDict::new(py);
    dict.set_item("x", x_arr.into_pyarray(py))?;
    dict.set_item("y", y_arr.into_pyarray(py))?;
    dict.set_item("z", z_arr.into_pyarray(py))?;
    dict.set_item("pressure", pressure.into_pyarray(py))?;
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
/// V[n+1] = V[n] + (dt/C_m) · (−G_leak·(V[n]−E_leak) + I_ion[n])
/// if V[n+1] ≥ V_thresh → spike, V ← V_reset, clamp for τ_ref
/// ```
///
/// # Arguments
///
/// - `i_ion_a`: ion current time series [A]; length N defines the simulation duration
/// - `dt_s`: uniform time step [s]; must be strictly positive
/// - `capacitance_f`: membrane capacitance C_m [F]  (default 100 pF, Koch 1999)
/// - `leak_conductance_s`: leak conductance G_leak [S]  (default 10 nS)
/// - `leak_reversal_v`: leak reversal potential E_leak [V]  (default −65 mV)
/// - `threshold_v`: spike threshold V_thresh [V]  (default −55 mV)
/// - `reset_v`: post-spike reset voltage V_reset [V]  (default −65 mV)
/// - `refractory_s`: absolute refractory period τ_ref [s]  (default 2 ms)
///
/// # Returns
///
/// dict with keys:
/// - `voltage_v`: membrane voltage trace [V], length N
/// - `spike_times_s`: chronological spike times [s]
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
    if dt_s <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "dt_s must be strictly positive",
        ));
    }
    let current = i_ion_a.as_array().to_owned();
    let n = current.len();
    let (voltage, spike_times) = py.detach(|| {
        let params = LifParams {
            capacitance_f,
            leak_conductance_s,
            leak_reversal_v,
            threshold_v,
            reset_v,
            refractory_s,
        };
        let mut neuron = LifNeuron::new(params);
        let mut voltage = Vec::with_capacity(n);
        for (idx, &i) in current.iter().enumerate() {
            let t_now = idx as f64 * dt_s;
            // step() only errors for dt ≤ 0; guaranteed safe by pre-call validation.
            let _ = neuron.step(i, dt_s, t_now);
            voltage.push(neuron.membrane_voltage());
        }
        let spike_times: Vec<f64> = neuron.spike_times().to_vec();
        (voltage, spike_times)
    });
    let spike_count = spike_times.len();
    let dict = PyDict::new(py);
    dict.set_item(
        "voltage_v",
        Array1::from_vec(voltage).into_pyarray(py),
    )?;
    dict.set_item(
        "spike_times_s",
        Array1::from_vec(spike_times).into_pyarray(py),
    )?;
    dict.set_item("spike_count", spike_count)?;
    Ok(dict)
}

// ── Module registration ───────────────────────────────────────────────────────

pub fn register_sonogenetics(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(
        compute_acoustic_membrane_tension_py,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(boltzmann_open_probability_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(coupled_channel_drive_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(gaussian_beam_pressure_field_py, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(simulate_lif_neuron_py, m)?)?;
    Ok(())
}
