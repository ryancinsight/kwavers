// Medium property re-upload between consecutive scan-line runs.
//
// SRP boundary: this module changes when the set of medium-dependent GPU
// buffers changes. Time-stepping and bind-group layout changes belong to their
// own modules.

use super::{
    state::{PstdMediumUpdateState, PstdStateProvider, WgpuPstdState},
    time_loop::commands::{PstdCommandProvider, WgpuPstdCommandProvider},
    GpuPstdSolver,
};

impl PstdMediumUpdateState for WgpuPstdState {
    fn update_medium_variable<T>(&mut self, c0_flat: &[T], rho0_flat: &[T])
    where
        T: Into<f64> + Copy,
    {
        self.scratch_c0_sq
            .iter_mut()
            .zip(c0_flat)
            .for_each(|(o, &c)| {
                let c = c.into() as f32;
                *o = c * c;
            });
        self.scratch_rho0_inv
            .iter_mut()
            .zip(rho0_flat)
            .for_each(|(o, &r)| *o = 1.0 / (r.into() as f32));
        self.scratch_rho0_flat
            .iter_mut()
            .zip(rho0_flat)
            .for_each(|(o, &r)| *o = r.into() as f32);

        let commands = WgpuPstdCommandProvider::new(self.device(), self.queue());
        commands.write_buffer(&self.medium_buffers.c0_sq, 0, &self.scratch_c0_sq);
        commands.write_buffer(&self.medium_buffers.rho0, 0, &self.scratch_rho0_flat);
        commands.write_buffer(&self.medium_buffers.rho0_inv, 0, &self.scratch_rho0_inv);
    }

    fn update_medium<T>(
        &mut self,
        c0_flat: &[T],
        rho0_flat: &[T],
        bon_a_flat: &[f32],
        absorb_tau_flat: &[f32],
        absorb_eta_flat: &[f32],
    ) where
        T: Into<f64> + Copy,
    {
        self.update_medium_variable(c0_flat, rho0_flat);
        let commands = WgpuPstdCommandProvider::new(self.device(), self.queue());
        commands.write_buffer(&self.medium_buffers.bon_a, 0, bon_a_flat);
        commands.write_buffer(&self.absorption_buffers.tau, 0, absorb_tau_flat);
        commands.write_buffer(&self.absorption_buffers.eta, 0, absorb_eta_flat);
    }

    fn disable_source_correction(&self) {
        let commands = WgpuPstdCommandProvider::new(self.device(), self.queue());
        commands.write_buffer(
            &self.medium_buffers.source_kappa,
            0,
            &self.scratch_source_kappa_ones,
        );
    }
}

impl<P> GpuPstdSolver<P>
where
    P: PstdStateProvider,
    P::State: PstdMediumUpdateState,
{
    /// Update only the variable medium-dependent GPU buffers for a new scan-line simulation.
    ///
    /// Reuses the existing device, pipelines, kappa, PML, nonlinearity, and
    /// absorption state. Only the sound speed and density tensors that change
    /// between scan lines are re-uploaded. This avoids redundant writes of the
    /// static `bon_a`, `absorb_tau`, and `absorb_eta` buffers.
    ///
    /// Call this before `run()` when only the phantom slice changes between runs.
    pub fn update_medium_variable<T>(&mut self, c0_flat: &[T], rho0_flat: &[T])
    where
        T: Into<f64> + Copy,
    {
        self.state.update_medium_variable(c0_flat, rho0_flat);
    }

    /// Update all medium-dependent GPU buffers for a new scan-line simulation.
    ///
    /// This is the full refresh path for callers that change nonlinearity or
    /// attenuation terms between runs. Scan-line B-mode paths should prefer
    /// `update_medium_variable()` so the static buffers remain resident.
    pub fn update_medium<T>(
        &mut self,
        c0_flat: &[T],
        rho0_flat: &[T],
        bon_a_flat: &[f32],
        absorb_tau_flat: &[f32],
        absorb_eta_flat: &[f32],
    ) where
        T: Into<f64> + Copy,
    {
        self.state.update_medium(
            c0_flat,
            rho0_flat,
            bon_a_flat,
            absorb_tau_flat,
            absorb_eta_flat,
        );
    }

    /// Overwrite the source_kappa buffer with `1.0` everywhere, effectively
    /// disabling the k-space source correction `sinc(c * dt * |k| / 2)` that
    /// `apply_source_kappa` multiplies onto the injected velocity source.
    ///
    /// This matches k-wave-python's `u_mode = "additive-no-correction"`, which
    /// is the default set by `NotATransducer` (see kwave/ksource.py:186 and
    /// kwave/ktransducer.py:244). Call before `run()` on any scan line where
    /// the caller wants raw additive injection; effects persist across runs
    /// until `enable_source_correction()` is called.
    pub fn disable_source_correction(&self) {
        self.state.disable_source_correction();
    }
}
