use super::PSTDSolver;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array2;

impl PSTDSolver {
    /// Run `checkpoint_steps` steps then persist full solver state to `path`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn run_to_checkpoint(
        &mut self,
        checkpoint_steps: usize,
        path: &std::path::Path,
    ) -> KwaversResult<()> {
        use crate::solver::forward::pstd::checkpoint::PSTDCheckpoint;

        // Mirror run_orchestrated convention: only record initial state for IVP (p0) sources.
        if self.time_step_index == 0 && self.source_handler.has_initial_pressure() {
            self.sensor_recorder.record_step(&self.fields.p)?;
        }
        for _ in 0..checkpoint_steps {
            self.step_forward()?;
        }

        let (sensor_data, sensor_next_step, sensor_expected_steps) = self
            .sensor_recorder
            .checkpoint_state_view()
            .map_or((None, 0, 0), |(view, ns, es)| {
                (Some(view.to_owned()), ns, es)
            });

        // Zero-clone: serialize directly from borrowed solver field references.
        // Avoids 6 × Array3<f64>::clone() — for a 256³ grid this saves ~768 MiB
        // of intermediate allocations per checkpoint.
        PSTDCheckpoint::save_borrowed(
            path,
            self.grid.nx,
            self.grid.ny,
            self.grid.nz,
            self.time_step_index,
            self.config.nt,
            self.config.dt,
            &self.fields.p,
            &self.fields.ux,
            &self.fields.uy,
            &self.fields.uz,
            &self.rhox,
            &self.rhoy,
            &self.rhoz,
            sensor_data.as_ref(),
            sensor_next_step,
            sensor_expected_steps,
        )
    }

    /// Restore state from `path` and run `remaining_steps` steps to completion.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn run_from_checkpoint(
        &mut self,
        path: &std::path::Path,
        remaining_steps: usize,
    ) -> KwaversResult<Option<Array2<f64>>> {
        use crate::solver::forward::pstd::checkpoint::PSTDCheckpoint;

        let ckpt = PSTDCheckpoint::load(path)?;
        self.run_from_checkpoint_loaded(ckpt, path, remaining_steps)
    }

    /// Restore state from an already loaded checkpoint and continue the run.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn run_from_checkpoint_loaded(
        &mut self,
        ckpt: crate::solver::forward::pstd::checkpoint::PSTDCheckpoint,
        path: &std::path::Path,
        remaining_steps: usize,
    ) -> KwaversResult<Option<Array2<f64>>> {
        ckpt.validate_restore_contract(
            self.grid.nx,
            self.grid.ny,
            self.grid.nz,
            self.config.nt,
            self.config.dt,
        )?;

        let expected_remaining = self
            .config
            .nt
            .checked_sub(ckpt.time_step_index)
            .ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "checkpoint time_step_index {} exceeds solver total_steps {}",
                    ckpt.time_step_index, self.config.nt
                ))
            })?;
        if remaining_steps != expected_remaining {
            return Err(KwaversError::InvalidInput(format!(
                "checkpoint remaining_steps {} ≠ expected {}",
                remaining_steps, expected_remaining
            )));
        }

        self.fields.p.assign(&ckpt.p);
        self.fields.ux.assign(&ckpt.ux);
        self.fields.uy.assign(&ckpt.uy);
        self.fields.uz.assign(&ckpt.uz);
        self.rhox.assign(&ckpt.rhox);
        self.rhoy.assign(&ckpt.rhoy);
        self.rhoz.assign(&ckpt.rhoz);
        self.time_step_index = ckpt.time_step_index;

        if let Some(sensor_data) = ckpt.sensor_data {
            self.sensor_recorder
                .restore_from_checkpoint(sensor_data, ckpt.sensor_next_step)?;
        }

        let _ = std::fs::remove_file(path);

        for _ in 0..remaining_steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }
}
