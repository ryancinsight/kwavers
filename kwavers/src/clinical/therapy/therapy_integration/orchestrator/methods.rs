use crate::clinical::therapy::therapy_integration::acoustic::AcousticWaveSolver;
use crate::clinical::therapy::therapy_integration::config::TherapyIntegrationModality;
use crate::clinical::therapy::therapy_integration::intensity_tracker::IntensityTracker;
use crate::clinical::therapy::therapy_integration::safety_controller::{
    SafetyController, TherapyAction,
};
use crate::clinical::therapy::therapy_integration::state::{
    SafetyMetrics, TherapyIntegrationSafetyStatus, TherapySessionState,
};
use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use ndarray::Array3;

use super::super::config::TherapySessionConfig;
use super::{
    cavitation, chemical, execution, initialization, lithotripsy, microbubble, safety,
    TherapyIntegrationOrchestrator,
};

impl TherapyIntegrationOrchestrator {
    /// Create new therapy integration orchestrator
    ///
    /// Initializes the orchestrator with configuration, grid, and medium properties.
    /// Automatically initializes modality-specific subsystems based on the configuration.
    ///
    /// # Errors
    ///
    /// Returns error if any subsystem initialization fails
    pub fn new(
        config: TherapySessionConfig,
        grid: Grid,
        medium: Box<dyn Medium>,
    ) -> KwaversResult<Self> {
        let acoustic_solver = AcousticWaveSolver::new(&grid, &*medium)?;

        let ceus_system = if config.primary_modality == TherapyIntegrationModality::Microbubble
            || config
                .secondary_modalities
                .contains(&TherapyIntegrationModality::Microbubble)
        {
            Some(initialization::init_ceus_system(&grid, &*medium)?)
        } else {
            None
        };

        let transcranial_system = if config.primary_modality
            == TherapyIntegrationModality::Transcranial
            || config
                .secondary_modalities
                .contains(&TherapyIntegrationModality::Transcranial)
        {
            Some(initialization::init_transcranial_system(&config, &grid)?)
        } else {
            None
        };

        let chemical_model = if config.primary_modality == TherapyIntegrationModality::Sonodynamic
            || config
                .secondary_modalities
                .contains(&TherapyIntegrationModality::Sonodynamic)
        {
            Some(initialization::init_chemical_model(&grid)?)
        } else {
            None
        };

        let cavitation_controller = if config.primary_modality
            == TherapyIntegrationModality::Histotripsy
            || config.primary_modality == TherapyIntegrationModality::Oncotripsy
            || config
                .secondary_modalities
                .contains(&TherapyIntegrationModality::Histotripsy)
            || config
                .secondary_modalities
                .contains(&TherapyIntegrationModality::Oncotripsy)
        {
            Some(initialization::init_cavitation_controller(&config)?)
        } else {
            None
        };

        let lithotripsy_simulator = if config.primary_modality
            == TherapyIntegrationModality::Lithotripsy
            || config
                .secondary_modalities
                .contains(&TherapyIntegrationModality::Lithotripsy)
        {
            Some(initialization::init_lithotripsy_simulator(&config, &grid)?)
        } else {
            None
        };

        let session_state = TherapySessionState {
            current_time: 0.0,
            progress: 0.0,
            acoustic_field: None,
            microbubble_concentration: None,
            cavitation_activity: None,
            chemical_concentrations: None,
            safety_metrics: SafetyMetrics {
                thermal_index: 0.0,
                mechanical_index: 0.0,
                cavitation_dose: 0.0,
                temperature_rise: Array3::zeros(grid.dimensions()),
            },
        };

        let mut safety_controller = SafetyController::new(config.safety_limits.clone(), None);
        safety_controller.start_monitoring(0.0);

        // 0.1 s averaging window; 10 µs acoustic time step.
        let intensity_tracker = IntensityTracker::new(0.1, 1e-5)?;

        Ok(Self {
            config,
            grid,
            medium,
            _acoustic_solver: acoustic_solver,
            ceus_system,
            _transcranial_system: transcranial_system,
            chemical_model,
            cavitation_controller,
            lithotripsy_simulator,
            safety_controller,
            intensity_tracker,
            session_state,
        })
    }

    /// Execute therapy session step
    ///
    /// Advances the therapy session by one time step, including acoustic field generation,
    /// real-time intensity monitoring, safety evaluation, and modality-specific updates.
    ///
    /// ## Acoustic-field fidelity (applies to ALL modalities, incl. HIFU)
    ///
    /// The acoustic field is currently produced by the **linear Gaussian-beam
    /// estimator** [`execution::generate_acoustic_field`], valid only at
    /// low/diagnostic intensity. For high-intensity HIFU this does **not** model
    /// shock formation, nonlinear harmonic heating, or cavitation — the KZK
    /// nonlinear solver (`solver::forward::nonlinear::kzk_solver_plugin`) exists
    /// but is **not yet wired into this orchestration path**. Treat HIFU
    /// intensity/dose from this loop as a planning estimate, not a nonlinear
    /// prediction. Tracked: gap_audit.md CLD-2 → backlog Sprint C.
    ///
    /// # Errors
    ///
    /// Returns error if any physics subsystem fails
    pub fn execute_therapy_step(&mut self, dt: f64) -> KwaversResult<()> {
        self.session_state.current_time += dt;
        self.session_state.progress = self.session_state.current_time / self.config.duration;

        let mut acoustic_field =
            execution::generate_acoustic_field(&self.grid, &self.config.acoustic_params)?;

        let power_factor = self.safety_controller.power_reduction_factor();
        if power_factor < 1.0 {
            acoustic_field.pressure *= power_factor;
            for vel in &mut [
                &mut acoustic_field.velocity_x,
                &mut acoustic_field.velocity_y,
                &mut acoustic_field.velocity_z,
            ] {
                **vel *= power_factor;
            }
        }

        let corrected_field = acoustic_field;

        // Acoustic impedance field Z = ρ₀ · c₀ (Rayl = kg/(m²·s)).
        // Derived from the medium's per-voxel density and sound-speed arrays.
        let impedance = {
            let rho = self.medium.density_array();
            let c = self.medium.sound_speed_array();
            &rho * &c
        };

        // Record intensity for thermal-dose accumulation tracking; return value not used
        // directly (TI is derived from the temperature field, not from SPTA).
        self.intensity_tracker.record_intensity(
            &corrected_field.pressure,
            &impedance,
            self.session_state.current_time,
        )?;

        let temperature_field = execution::calculate_acoustic_heating(
            &corrected_field,
            &self.grid,
            dt,
            self.config.acoustic_params.focal_depth,
        );

        self.intensity_tracker
            .update_thermal_dose(&temperature_field, dt)?;

        // Thermal Index (IEC 62359): TI ≈ ΔT_max (°C) — ratio of acoustic power to power
        // required for a 1 °C rise. The computed temperature field (in °C, baseline 37 °C)
        // gives the best available proxy in the Gaussian heating model.
        // TI = T_max - T_ambient; T_ambient = 37 °C per CEM43 convention.
        let t_max = temperature_field
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        self.session_state.safety_metrics.thermal_index = (t_max - BODY_TEMPERATURE_C).max(0.0);

        // Mechanical Index (FDA 510(k) guidance, IEC 62359):
        // MI = p_neg_peak_derated (MPa) / sqrt(f_center (MHz))
        //    = (pnp_Pa / 1e6) / sqrt(f_Hz / 1e6)
        //    = pnp_Pa / (1e3 × sqrt(f_Hz))
        self.session_state.safety_metrics.mechanical_index = self.config.acoustic_params.pnp
            / (1e3 * (self.config.acoustic_params.frequency).sqrt());
        self.session_state.safety_metrics.temperature_rise = temperature_field.clone();

        let safety_action = self.safety_controller.evaluate_safety(
            SafetyMetrics {
                thermal_index: self.session_state.safety_metrics.thermal_index,
                mechanical_index: self.session_state.safety_metrics.mechanical_index,
                cavitation_dose: self.session_state.safety_metrics.cavitation_dose,
                temperature_rise: temperature_field.clone(),
            },
            self.session_state.current_time,
        )?;

        match safety_action {
            TherapyAction::Continue => {}
            TherapyAction::Warning => {}
            TherapyAction::ReducePower => {}
            TherapyAction::Stop => {}
        }

        if let Some(ref mut ceus) = self.ceus_system {
            let concentration =
                microbubble::update_microbubble_dynamics(ceus, &corrected_field, dt)?;
            self.session_state.microbubble_concentration = concentration;
        }

        if let Some(ref mut controller) = self.cavitation_controller {
            let cavitation_activity = cavitation::update_cavitation_control(
                controller,
                &corrected_field,
                &self.config.acoustic_params,
                dt,
            )?;
            self.session_state.cavitation_activity = Some(cavitation_activity.clone());

            let total_cavitation_activity: f64 =
                cavitation_activity.iter().sum::<f64>() / cavitation_activity.len() as f64;
            self.session_state.safety_metrics.cavitation_dose += total_cavitation_activity * dt;
        }

        if let Some(ref mut chemistry) = self.chemical_model {
            let chemical_concentrations = chemical::update_chemical_reactions(
                chemistry,
                &corrected_field,
                self.session_state.cavitation_activity.as_ref(),
                &self.config.acoustic_params,
                &self.grid,
                &*self.medium,
                dt,
            )?;
            self.session_state.chemical_concentrations = Some(chemical_concentrations);
        }

        if let Some(ref mut simulator) = self.lithotripsy_simulator {
            let progress = lithotripsy::execute_lithotripsy_step(simulator, &corrected_field, dt)?;
            self.session_state.progress = progress;
        }

        // NOTE: safety_metrics.{thermal_index, mechanical_index, temperature_rise} are set
        // above from SPTA and PNP before the safety-controller evaluation.
        // safety_metrics.cavitation_dose is accumulated in the cavitation block above.
        // update_safety_metrics() is NOT called here to avoid overwriting the SPTA-derived
        // TI with the P_rms formula (wrong for therapy bandwidths) and to avoid
        // double-counting cavitation dose when cavitation_activity is Some.

        self.session_state.acoustic_field = Some(corrected_field);

        Ok(())
    }

    /// Evaluates current safety metrics against configured limits.
    pub fn check_safety_limits(&self) -> TherapyIntegrationSafetyStatus {
        safety::check_safety_limits(
            &self.session_state.safety_metrics,
            &self.config.safety_limits,
            self.session_state.current_time,
        )
    }

    /// Returns true if the safety controller has detected a critical limit violation.
    pub fn should_stop(&self) -> bool {
        self.safety_controller.should_stop()
    }

    /// Returns a multiplier [0.0, 1.0] indicating current acoustic power level.
    pub fn power_reduction_factor(&self) -> f64 {
        self.safety_controller.power_reduction_factor()
    }

    /// Returns reference to current therapy session state.
    pub fn session_state(&self) -> &TherapySessionState {
        &self.session_state
    }

    /// Returns reference to therapy session configuration.
    pub fn config(&self) -> &TherapySessionConfig {
        &self.config
    }
}
