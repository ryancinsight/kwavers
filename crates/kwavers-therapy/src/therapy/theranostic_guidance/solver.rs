//! Same-device exposure synthesis, finite-frequency inverse, RTM, and harmonic reconstruction.
//!
//! # Fusion weight derivation
//!
//! The fused lesion score is a convex combination of four independent lesion estimates
//! modulated by a passive-channel confidence gate:
//!
//! ```text
//! F = (w_a·A + w_s·S + w_h·H + w_u·U) · (0.25 + 0.75·max(S, U))
//! ```
//!
//! Weights are chosen so that the two highest-SNR channels (active Born inverse and
//! subharmonic passive) dominate, while harmonic and ultraharmonic provide contrast
//! refinement. The gate suppresses fused output in regions with no subharmonic or
//! ultraharmonic cavitation signature.
//!
//! | Channel | Symbol | Weight | Rationale |
//! |---------|--------|--------|-----------|
//! | Active Born inverse | A | 0.40 | Highest SNR; directly images sound-speed contrast |
//! | Subharmonic passive | S | 0.25 | Second strongest cavitation marker |
//! | Harmonic inverse    | H | 0.20 | Tissue-nonlinearity contrast; correlated with A |
//! | Ultraharmonic passive| U| 0.15 | Lowest SNR; broadband cavitation marker |

use kwavers_core::error::{KwaversError, KwaversResult};

/// Fusion weight for the active Born inverse channel (highest SNR).
const FUSED_WEIGHT_ACTIVE: f64 = 0.40;
/// Fusion weight for the subharmonic passive channel.
const FUSED_WEIGHT_SUBHARMONIC: f64 = 0.25;
/// Fusion weight for the harmonic inverse channel.
const FUSED_WEIGHT_HARMONIC: f64 = 0.20;
/// Fusion weight for the ultraharmonic passive channel (lowest SNR).
const FUSED_WEIGHT_ULTRAHARMONIC: f64 = 0.15;
/// Minimum confidence gate floor applied to the passive cavitation gate.
const FUSED_GATE_FLOOR: f64 = 0.25;
use kwavers_solver::inverse::same_aperture::{
    active_grid, fundamental_operator, harmonic_operator, image_from_vector, passive_operator,
    solve_tikhonov_h1, ultraharmonic_operator, vector_from_image, EncodedOperator, LinearOperator,
    PcgSettings, SameApertureMedium, SameApertureSettings, C_REF_M_S, SAME_APERTURE_OPERATOR_MODEL,
};
use leto::Array2;

use super::config::{PassiveReconstructionMode, TheranosticInverseConfig};
use super::elastic_shear::{
    reconstruct_elastic_shear, ElasticShearReconstructionResult, THERANOSTIC_ELASTIC_SHEAR_MODEL,
};
use super::exposure::normalize_positive;
use super::geometry::{angle_span, build_device_layout, DeviceLayout, Point2};
use super::medium::{target_contrast, PreparedTheranosticSlice};
use super::metrics::{metrics_for, ReconstructionMetrics};
use super::transmit_schedule::{select_transmit_schedule, TransmitScheduleResult};
use super::waveform::{
    passive_acoustic_maps, simulate_peak_pressure_exposure, simulate_waveform_adjoint_rtm,
    WaveformSimulationResult,
};
use kwavers_solver::inverse::same_aperture::ActiveGrid;

/// Canonical model identifier exported through PyO3 and figure metadata.
///
/// # Contract
///
/// The identifier changes only when the mathematical inverse operator changes.
/// The current model uses finite-frequency same-aperture rows and solves the
/// graph-Laplacian-regularized normal equations by preconditioned conjugate
/// gradients.
pub const THERANOSTIC_OPERATOR_MODEL: &str = SAME_APERTURE_OPERATOR_MODEL;
pub const THERANOSTIC_OPERATOR_BACKEND: &str = "matrix_free_finite_frequency_same_aperture";
pub const THERANOSTIC_INVERSE_MODEL_FAMILY: &str =
    "reduced_born_normal_equation_plus_linear_acoustic_rtm_plus_iterative_nonlinear_elastic_fwi";

/// Full-waveform-inversion flag for the **acoustic** reconstruction channels.
///
/// `false` because the acoustic anatomy / lesion / harmonic / ultraharmonic
/// channels are reduced-Born / Tikhonov inversions of the same-aperture
/// finite-frequency operator (linearised, one-shot Gauss–Newton-class
/// normal-equation solve), not full-waveform inversion. The 2-D RTM channel
/// is a single-pass adjoint imaging condition on time-domain traces, also
/// not FWI. See `waveform/mod.rs:10` which states "It is not nonlinear wave
/// propagation and it is not an iterative acoustic full-waveform inversion."
///
/// The elastic-shear channel (`elastic_shear`) IS an iterative
/// gradient-descent FWI with line search; that fact is exposed via the
/// dedicated [`THERANOSTIC_ELASTIC_SHEAR_MODEL`] identifier rather than
/// being conflated with the acoustic inverse here.
pub const THERANOSTIC_FULL_WAVE_INVERSION: bool = false;

/// Nonlinear-wave-propagation flag.
///
/// `false`: the forward acoustic exposure and RTM forward both solve the
/// linear scalar wave equation `p_tt = c² Δp + s` with linear power-law
/// attenuation. Westervelt / KZK nonlinearity is **not** integrated here;
/// for second-harmonic / shock-formation modelling use
/// `kwavers_physics::acoustics::nonlinear` instead.
pub const THERANOSTIC_NONLINEAR_WAVE_PROPAGATION: bool = false;

/// Iterative-elastic-FWI flag.
///
/// `true`: the elastic-shear channel runs `config.elastic_fwi_iterations`
/// iterations of `migrate-residual → line-search → accept/reject` against
/// the elastic PSTD forward operator, with a backtracking step-size policy.
/// See `elastic_shear::inversion::run_iterative_elastic_fwi`.
pub const THERANOSTIC_ITERATIVE_ELASTIC_FWI: bool = true;

#[derive(Clone, Debug)]
pub struct TheranosticInverseResult {
    pub prepared: PreparedTheranosticSlice,
    pub layout: DeviceLayout,
    pub exposure: Array2<f64>,
    pub exposure_raw_peak_pressure: Array2<f64>,
    pub exposure_model: String,
    pub exposure_backend: String,
    pub exposure_uses_hybrid_pstd_fdtd: bool,
    pub exposure_source_count: usize,
    pub exposure_time_steps: usize,
    pub exposure_dt_s: f64,
    pub exposure_workspace_values: usize,
    pub lesion_target: Array2<f64>,
    pub anatomy_reconstruction: Array2<f64>,
    pub active_lesion_reconstruction: Array2<f64>,
    pub waveform_rtm_reconstruction: Array2<f64>,
    pub elastic_shear_reconstruction: Array2<f64>,
    pub subharmonic_reconstruction: Array2<f64>,
    pub harmonic_reconstruction: Array2<f64>,
    pub ultraharmonic_reconstruction: Array2<f64>,
    pub fused_reconstruction: Array2<f64>,
    pub anatomy_metrics: ReconstructionMetrics,
    pub active_metrics: ReconstructionMetrics,
    pub waveform_metrics: ReconstructionMetrics,
    pub elastic_shear_metrics: ReconstructionMetrics,
    pub subharmonic_metrics: ReconstructionMetrics,
    pub harmonic_metrics: ReconstructionMetrics,
    pub ultraharmonic_metrics: ReconstructionMetrics,
    pub fused_metrics: ReconstructionMetrics,
    pub objective_history: Vec<f64>,
    pub transmit_schedule: TransmitScheduleResult,
    pub measurements: usize,
    pub encoded_measurements: usize,
    pub unencoded_measurements: usize,
    pub inverse_encoding_rows_per_code: usize,
    pub active_voxels: usize,
    pub operator_backend: String,
    pub operator_storage_values: usize,
    pub dense_operator_values: usize,
    pub inverse_model_family: String,
    pub elastic_shear_model: String,
    pub elastic_shear: ElasticShearReconstructionResult,
    pub is_full_wave_inversion: bool,
    pub uses_nonlinear_wave_propagation: bool,
    pub waveform: WaveformSimulationResult,
}

pub fn run_theranostic_inverse(
    prepared: PreparedTheranosticSlice,
    config: &TheranosticInverseConfig,
) -> KwaversResult<TheranosticInverseResult> {
    config.validate()?;
    let layout = build_device_layout(
        config,
        &prepared.body_mask,
        &prepared.target_mask,
        prepared.spacing_m,
    )?;
    let transmit_schedule = select_transmit_schedule(&layout, &prepared, config.transmit_schedule)?;
    let acquisition_layout = acquisition_layout_for_schedule(&layout, &transmit_schedule);
    let active_mask = &prepared.body_mask;
    let active = active_grid(active_mask, prepared.spacing_m);
    if active.len() < 16 {
        return Err(KwaversError::InvalidInput(
            "theranostic active support has fewer than 16 voxels".to_owned(),
        ));
    }
    let medium = SameApertureMedium {
        attenuation_np_per_m_mhz: &prepared.attenuation_np_per_m_mhz,
        spacing_m: prepared.spacing_m,
    };
    let settings = SameApertureSettings {
        frequencies_hz: &config.frequencies_hz,
        receiver_offsets: &config.receiver_offsets,
        phase_speed_m_s: C_REF_M_S,
    };
    let fundamental = EncodedOperator::deterministic_signs(
        fundamental_operator(
            medium,
            &transmit_schedule.source_elements,
            &active,
            settings,
        ),
        config.inverse_encoding_rows_per_code,
    );
    let harmonic = EncodedOperator::deterministic_signs(
        harmonic_operator(
            medium,
            &transmit_schedule.source_elements,
            &active,
            settings,
        ),
        config.inverse_encoding_rows_per_code,
    );
    let ultraharmonic = EncodedOperator::deterministic_signs(
        ultraharmonic_operator(
            medium,
            &transmit_schedule.source_elements,
            &active,
            settings,
        ),
        config.inverse_encoding_rows_per_code,
    );
    let passive = EncodedOperator::deterministic_signs(
        passive_operator(
            medium,
            &layout.therapy_elements,
            &layout.imaging_receivers,
            &active,
            &config.frequencies_hz,
        ),
        config.inverse_encoding_rows_per_code,
    );
    let measurements = fundamental.rows() + passive.rows() + harmonic.rows() + ultraharmonic.rows();
    let unencoded_measurements = fundamental.encoding_spec().original_rows()
        + passive.encoding_spec().original_rows()
        + harmonic.encoding_spec().original_rows()
        + ultraharmonic.encoding_spec().original_rows();
    let operator_storage_values = fundamental.storage_values()
        + passive.storage_values()
        + harmonic.storage_values()
        + ultraharmonic.storage_values();
    let dense_operator_values = fundamental.dense_values()
        + passive.dense_values()
        + harmonic.dense_values()
        + ultraharmonic.dense_values();
    let inverse_settings = inverse_settings(config);
    let exposure_result = simulate_peak_pressure_exposure(&prepared, &layout, config);
    let exposure = exposure_result.exposure.clone();
    let lesion_target = lesion_source(&prepared, &exposure);
    let waveform =
        simulate_waveform_adjoint_rtm(&prepared, &acquisition_layout, config, &lesion_target);

    let anatomy_target = target_contrast(&prepared);
    let anatomy_vec = vector_from_image(&anatomy_target, &active);
    let anatomy_result = solve_tikhonov_h1(&fundamental, &anatomy_vec, &active, inverse_settings);
    let mut history = anatomy_result.objective_history;
    let anatomy_reconstruction = image_from_vector(
        &anatomy_result.model,
        &active,
        (active_mask.shape()[0], active_mask.shape()[1]),
    );

    let mut lesion_speed = lesion_target.clone();
    for v in lesion_speed.iter_mut() {
        *v = *v * config.lesion_delta_c_m_s / C_REF_M_S;
    }
    let lesion_speed_vec = vector_from_image(&lesion_speed, &active);
    let active_result =
        solve_tikhonov_h1(&fundamental, &lesion_speed_vec, &active, inverse_settings);
    history.extend(active_result.objective_history);
    let active_lesion_reconstruction = normalize_positive(
        &image_from_vector(
            &negated(&active_result.model),
            &active,
            (active_mask.shape()[0], active_mask.shape()[1]),
        ),
        active_mask,
    );
    let waveform_rtm_reconstruction = waveform.reconstruction.clone();
    let elastic_shear = reconstruct_elastic_shear(&prepared, &layout, config, &lesion_target)?;
    let elastic_shear_reconstruction = elastic_shear.reconstruction.clone();

    // Harmonic (2·f₀) is an active-transmit tissue-nonlinearity inverse and is
    // unaffected by the passive-reconstruction mode.
    let harmonic_target = harmonic_target(&prepared, &lesion_target);
    let harmonic_vec = vector_from_image(&harmonic_target, &active);
    let harmonic_result = solve_tikhonov_h1(&harmonic, &harmonic_vec, &active, inverse_settings);
    history.extend(harmonic_result.objective_history);
    let harmonic_reconstruction = normalize_positive(
        &image_from_vector(
            &harmonic_result.model,
            &active,
            (active_mask.shape()[0], active_mask.shape()[1]),
        ),
        active_mask,
    );

    // Metric reference for the ultraharmonic channel (lesion-derived); also the
    // inversion target in the finite-frequency-operator mode.
    let ultraharmonic_target = ultraharmonic_target(&prepared, &lesion_target);

    // Subharmonic (f₀/2) and ultraharmonic (3·f₀/2) are passive cavitation
    // channels. They are reconstructed either by the finite-frequency operator
    // inverse against a synthetic target (default) or by genuine passive
    // acoustic mapping of a simulated cavitation emission (config flag).
    let (subharmonic_reconstruction, ultraharmonic_reconstruction) =
        match config.passive_reconstruction {
            PassiveReconstructionMode::FiniteFrequencyOperator => {
                let sub_target_vec = vector_from_image(&lesion_target, &active);
                let sub_result =
                    solve_tikhonov_h1(&passive, &sub_target_vec, &active, inverse_settings);
                history.extend(sub_result.objective_history);
                let subharmonic = normalize_positive(
                    &image_from_vector(
                        &sub_result.model,
                        &active,
                        (active_mask.shape()[0], active_mask.shape()[1]),
                    ),
                    active_mask,
                );

                let ultraharmonic_vec = vector_from_image(&ultraharmonic_target, &active);
                let ultra_result = solve_tikhonov_h1(
                    &ultraharmonic,
                    &ultraharmonic_vec,
                    &active,
                    inverse_settings,
                );
                history.extend(ultra_result.objective_history);
                let ultraharmonic = normalize_positive(
                    &image_from_vector(
                        &ultra_result.model,
                        &active,
                        (active_mask.shape()[0], active_mask.shape()[1]),
                    ),
                    active_mask,
                );
                (subharmonic, ultraharmonic)
            }
            PassiveReconstructionMode::PassiveAcousticMapping => {
                passive_pam_channels(&prepared, &acquisition_layout, config, &active, active_mask)?
            }
        };

    let fused_reconstruction = fuse_maps(
        &active_lesion_reconstruction,
        &subharmonic_reconstruction,
        &harmonic_reconstruction,
        &ultraharmonic_reconstruction,
        active_mask,
    );
    let anatomy_metrics = metrics_for(&anatomy_target, &anatomy_reconstruction, active_mask);
    let active_metrics = metrics_for(&lesion_target, &active_lesion_reconstruction, active_mask);
    let waveform_metrics = metrics_for(&lesion_target, &waveform_rtm_reconstruction, active_mask);
    let elastic_shear_metrics =
        metrics_for(&lesion_target, &elastic_shear_reconstruction, active_mask);
    let subharmonic_metrics = metrics_for(&lesion_target, &subharmonic_reconstruction, active_mask);
    let harmonic_metrics = metrics_for(&harmonic_target, &harmonic_reconstruction, active_mask);
    let ultraharmonic_metrics = metrics_for(
        &ultraharmonic_target,
        &ultraharmonic_reconstruction,
        active_mask,
    );
    let fused_metrics = metrics_for(&lesion_target, &fused_reconstruction, active_mask);

    let _span = angle_span(&layout);
    Ok(TheranosticInverseResult {
        prepared,
        layout,
        exposure,
        exposure_raw_peak_pressure: exposure_result.raw_peak_pressure,
        exposure_model: exposure_result.model_name.to_owned(),
        exposure_backend: exposure_result.backend_name.to_owned(),
        exposure_uses_hybrid_pstd_fdtd: exposure_result.uses_hybrid_pstd_fdtd,
        exposure_source_count: exposure_result.source_count,
        exposure_time_steps: exposure_result.time_steps,
        exposure_dt_s: exposure_result.dt_s,
        exposure_workspace_values: exposure_result.workspace_values,
        lesion_target,
        anatomy_reconstruction,
        active_lesion_reconstruction,
        waveform_rtm_reconstruction,
        elastic_shear_reconstruction,
        subharmonic_reconstruction,
        harmonic_reconstruction,
        ultraharmonic_reconstruction,
        fused_reconstruction,
        anatomy_metrics,
        active_metrics,
        waveform_metrics,
        elastic_shear_metrics,
        subharmonic_metrics,
        harmonic_metrics,
        ultraharmonic_metrics,
        fused_metrics,
        objective_history: history,
        transmit_schedule,
        measurements,
        encoded_measurements: measurements,
        unencoded_measurements,
        inverse_encoding_rows_per_code: config.inverse_encoding_rows_per_code,
        active_voxels: active.len(),
        operator_backend: THERANOSTIC_OPERATOR_BACKEND.to_owned(),
        operator_storage_values,
        dense_operator_values,
        inverse_model_family: THERANOSTIC_INVERSE_MODEL_FAMILY.to_owned(),
        elastic_shear_model: THERANOSTIC_ELASTIC_SHEAR_MODEL.to_owned(),
        elastic_shear,
        is_full_wave_inversion: THERANOSTIC_FULL_WAVE_INVERSION,
        uses_nonlinear_wave_propagation: THERANOSTIC_NONLINEAR_WAVE_PROPAGATION,
        waveform,
    })
}

fn acquisition_layout_for_schedule(
    layout: &DeviceLayout,
    schedule: &TransmitScheduleResult,
) -> DeviceLayout {
    let mut scheduled = vec![false; schedule.total_element_count];
    for &idx in &schedule.active_indices {
        if idx < scheduled.len() {
            scheduled[idx] = true;
        }
    }
    let mut imaging_receivers = layout
        .therapy_elements
        .iter()
        .enumerate()
        .filter_map(|(idx, &point)| (!scheduled[idx]).then_some(point))
        .collect::<Vec<_>>();
    imaging_receivers.extend(layout.imaging_receivers.iter().copied());
    DeviceLayout {
        therapy_elements: schedule.source_elements.clone(),
        imaging_receivers,
        focus_m: layout.focus_m,
        skin_contact_m: layout.skin_contact_m,
        model_name: layout.model_name.clone(),
    }
}

fn lesion_source(prepared: &PreparedTheranosticSlice, exposure: &Array2<f64>) -> Array2<f64> {
    let target_peak = exposure
        .iter()
        .zip(prepared.target_mask.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .fold(0.0, f64::max)
        .max(1.0e-12);
    Array2::from_shape_fn(exposure.shape(), |idx| {
        if prepared.target_mask[idx] {
            (exposure[idx] / target_peak).clamp(0.0, 1.0)
        } else {
            0.0
        }
    })
}

fn harmonic_target(prepared: &PreparedTheranosticSlice, lesion: &Array2<f64>) -> Array2<f64> {
    let median = prepared
        .sound_speed_m_s
        .iter()
        .zip(prepared.body_mask.iter())
        .filter_map(|(value, active)| active.then_some(*value))
        .sum::<f64>()
        / prepared.body_mask.iter().filter(|v| **v).count().max(1) as f64;
    Array2::from_shape_fn(lesion.shape(), |idx| {
        let contrast = ((prepared.sound_speed_m_s[idx] - median).abs() / 120.0).clamp(0.0, 1.0);
        lesion[idx] * (0.8 + 0.2 * contrast)
    })
}

fn ultraharmonic_target(prepared: &PreparedTheranosticSlice, lesion: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(lesion.shape(), |idx| {
        let attenuation = prepared.attenuation_np_per_m_mhz[idx];
        lesion[idx] * (0.7 + 0.3 * (attenuation / 18.0).clamp(0.0, 1.0))
    })
}

/// Genuine passive-acoustic-mapping reconstruction of the subharmonic (f₀/2) and
/// ultraharmonic (3f₀/2) cavitation channels.
///
/// Simulates a single broadband bubble-cloud emission at the target region
/// through the prepared heterogeneous medium, records the receiver traces on the
/// same-device receive aperture, band-passes them to each cavitation band, and
/// DMAS-beamforms each into an intensity over the active grid. One forward solve
/// serves both bands. Each map is body-mask-normalized to match the other
/// reconstruction channels. Returns `(subharmonic_map, ultraharmonic_map)`.
///
/// ## Same-array send/receive
///
/// The receive aperture is the therapy array itself (the elements switch from
/// transmit to receive between therapy bursts), together with any dedicated
/// imaging receivers. This mirrors clinical practice — e.g. transcranial
/// histotripsy ACE mapping uses the 256-element hemispherical histotripsy
/// transducer's own elements as passive receivers (Sukovich et al. 2020) — and
/// is required for the transcranial helmet, which has no separate imaging array.
fn passive_pam_channels(
    prepared: &PreparedTheranosticSlice,
    layout: &DeviceLayout,
    config: &TheranosticInverseConfig,
    active: &ActiveGrid,
    active_mask: &Array2<bool>,
) -> KwaversResult<(Array2<f64>, Array2<f64>)> {
    let grid_points: Vec<Point2> = active
        .points_m
        .iter()
        .map(|p| Point2 {
            x_m: p.x_m,
            y_m: p.y_m,
        })
        .collect();
    let emission_points = target_emission_points(prepared);
    let pam_speed = mean_body_sound_speed(prepared);
    let f0 = config.frequencies_hz[0];
    let band_centers = [0.5 * f0, 1.5 * f0]; // subharmonic, ultraharmonic

    // Same-array receive aperture: therapy elements (in receive mode) plus any
    // dedicated imaging receivers. `passive_acoustic_maps` reads the receive
    // aperture from `imaging_receivers`, so route the combined set there.
    //
    // Aperture subsampling was analysed and is intentionally NOT applied here.
    // To remain alias-free across all reconstructed bands the limiting
    // wavelength is the ultraharmonic 3f₀/2 (λ_min/2 ≈ 0.8 mm at f₀ = 650 kHz);
    // the clinical apertures are already at or below this spacing (the brain
    // helmet is in fact undersampled at the ultraharmonic), so spatial-Nyquist
    // decimation would either be a no-op or alias the high-frequency band the
    // aberration correction just recovered. The eikonal redundancy is instead
    // removed losslessly by `eikonal_delay_matrix`, which solves once per unique
    // refined source cell (coincident dense-array elements share a solve).
    let mut receive_aperture = layout.imaging_receivers.clone();
    receive_aperture.extend(layout.therapy_elements.iter().copied());
    let receive_layout = DeviceLayout {
        imaging_receivers: receive_aperture,
        ..layout.clone()
    };

    let maps = passive_acoustic_maps(
        prepared,
        &receive_layout,
        config,
        &grid_points,
        &emission_points,
        f0,
        &band_centers,
        pam_speed,
    )?;
    let to_image = |intensity: &[f64]| -> Array2<f64> {
        let model: Vec<f32> = intensity.iter().map(|&v| v as f32).collect();
        normalize_positive(
            &image_from_vector(
                &model,
                active,
                (active_mask.shape()[0], active_mask.shape()[1]),
            ),
            active_mask,
        )
    };
    Ok((to_image(&maps[0]), to_image(&maps[1])))
}

/// Body-centred physical coordinates of the cavitation emission cells (the
/// target/lesion region), using the same origin convention as `active_grid`.
fn target_emission_points(prepared: &PreparedTheranosticSlice) -> Vec<Point2> {
    let [nx, ny] = prepared.target_mask.shape();
    let cx = (nx.saturating_sub(1)) as f64 * 0.5;
    let cy = (ny.saturating_sub(1)) as f64 * 0.5;
    prepared
        .target_mask
        .indexed_iter()
        .filter_map(|([ix, iy], &active)| {
            active.then_some(Point2 {
                x_m: (ix as f64 - cx) * prepared.spacing_m,
                y_m: (iy as f64 - cy) * prepared.spacing_m,
            })
        })
        .collect()
}

/// Mean sound speed over the body mask — the homogeneous steering speed for the
/// PAM delay model. Falls back to the reference speed when the body is empty.
fn mean_body_sound_speed(prepared: &PreparedTheranosticSlice) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (speed, active) in prepared
        .sound_speed_m_s
        .iter()
        .zip(prepared.body_mask.iter())
    {
        if *active {
            sum += *speed;
            count += 1;
        }
    }
    if count == 0 {
        C_REF_M_S
    } else {
        sum / count as f64
    }
}

fn fuse_maps(
    a: &Array2<f64>,
    s: &Array2<f64>,
    h: &Array2<f64>,
    u: &Array2<f64>,
    mask: &Array2<bool>,
) -> Array2<f64> {
    let fused = Array2::from_shape_fn(a.shape(), |idx| {
        if mask[idx] {
            (FUSED_WEIGHT_ACTIVE * a[idx]
                + FUSED_WEIGHT_SUBHARMONIC * s[idx]
                + FUSED_WEIGHT_HARMONIC * h[idx]
                + FUSED_WEIGHT_ULTRAHARMONIC * u[idx])
                * (FUSED_GATE_FLOOR + (1.0 - FUSED_GATE_FLOOR) * s[idx].max(u[idx]))
        } else {
            0.0
        }
    });
    normalize_positive(&fused, mask)
}

fn negated(values: &[f32]) -> Vec<f32> {
    values.iter().map(|v| -*v).collect()
}

fn inverse_settings(config: &TheranosticInverseConfig) -> PcgSettings {
    PcgSettings {
        iterations: config.iterations,
        regularization: config.regularization,
        smoothness_weight: config.smoothness_weight,
        noise_fraction: config.noise_fraction,
    }
}
