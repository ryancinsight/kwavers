//! Sensor data extraction methods for `ElasticWaveSolver`.

use super::super::definition::ElasticWaveSolver;
use leto::{Array2, ArrayView2};

/// Per-component recorded sensor traces `(x, y, z)`, each `Some` when the
/// corresponding component buffer was allocated; shape `(n_sensors, steps)`.
type VelocityComponentTraces = (
    Option<Array2<f64>>,
    Option<Array2<f64>>,
    Option<Array2<f64>>,
);

impl ElasticWaveSolver {
    pub fn extract_recorded_data(&self) -> Option<Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    /// Returns the per-component **particle-velocity** time series at
    /// sensor positions: `(vx, vy, vz)` traces in m/s.
    ///
    /// Returns `(ux_data, uy_data, uz_data)` where each is
    /// `Some(Array2<f64>)` of shape `(n_sensors, recorded_steps)` when
    /// the corresponding component buffer was allocated by the spec
    /// passed to the underlying `SensorRecorder`. The current
    /// `propagate` path allocates all three; this accessor exposes them
    /// through the public API for callers (e.g. PyO3 bridge) that cannot
    /// reach the `pub(crate)` `sensor_recorder` field directly.
    ///
    /// Phase A.2.5 of ADR 007.
    ///
    /// # Theorem (recorded quantity)
    ///
    /// For sensor index `s` at grid position `(i, j, k)` and recorded
    /// step `n ∈ [0, save_every · n_records − 1]` with stride
    /// `save_every`, the recorder satisfies
    ///
    /// ```text
    /// ux_data[s, n] = vx(i, j, k, t_n),     t_n = n · save_every · dt.
    /// ```
    ///
    /// # Proof
    ///
    /// In `propagation/mod.rs` the per-step branch invokes
    /// `sensor_recorder.record_velocity_step(&current_field.vx,
    /// &current_field.vy, &current_field.vz)`. The recorder copies the
    /// argument samples into `ux_data[s, col]` directly without scaling
    /// or differentiation. The integrator stores particle velocity in
    /// `field.vx` per the velocity-Verlet half-step
    /// `vx ← vx + (dt/2) · ax`, where `ax = (∇·σ)_x / ρ`.
    /// Therefore the recorded trace equals particle velocity at the
    /// sample times. ∎
    ///
    /// Cross-engine note: KWave.jl's `KWaveSensor(record=[:ux])` returns
    /// the same quantity (velocity), so direct comparison requires no
    /// unit conversion. Conversely, applying `np.gradient(_, dt)` to
    /// this output produces acceleration, not velocity, and inflates
    /// peak-amplitude ratios by a factor of order `1/dt` (~10⁷ at
    /// dt = 4 × 10⁻⁸ s).
    pub fn extract_recorded_velocity_components(&self) -> VelocityComponentTraces {
        (
            self.sensor_recorder.extract_ux_data(),
            self.sensor_recorder.extract_uy_data(),
            self.sensor_recorder.extract_uz_data(),
        )
    }

    /// Back-compat alias for [`Self::extract_recorded_velocity_components`].
    ///
    /// This method does **not** return displacement, despite the legacy
    /// name; it has always returned particle velocity (m/s). The shim
    /// is kept until callers migrate.
    #[doc(alias = "extract_recorded_velocity_components")]
    #[deprecated(
        since = "0.1.1",
        note = "Use extract_recorded_velocity_components — this method returns \
                particle velocity (m/s), not displacement."
    )]
    pub fn extract_recorded_displacement_components(&self) -> VelocityComponentTraces {
        self.extract_recorded_velocity_components()
    }

    /// Borrow the full allocated sensor displacement buffer without cloning.
    #[must_use]
    pub fn recorded_data_view(&self) -> Option<ArrayView2<'_, f64>> {
        self.sensor_recorder.pressure_data_view()
    }

    /// Borrow only populated sensor displacement samples without cloning.
    #[must_use]
    pub fn recorded_data_prefix_view(&self) -> Option<ArrayView2<'_, f64>> {
        self.sensor_recorder.recorded_pressure_view()
    }
}
