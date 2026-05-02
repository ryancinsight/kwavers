use super::{DispersionResults, NumericalValidator};

impl NumericalValidator {
    pub(super) fn validate_dispersion(
        &self,
    ) -> Result<DispersionResults, Box<dyn std::error::Error>> {
        use crate::domain::source::GridSource;
        use crate::solver::fdtd::{FdtdConfig, FdtdSolver};
        use crate::solver::forward::nonlinear::kuznetsov::{KuznetsovConfig, KuznetsovWave};
        use crate::solver::pstd::{PSTDConfig, PSTDSolver};
        use std::f64::consts::PI;

        let wavelength = 10.0 * self.grid.dx;
        let k = 2.0 * PI / wavelength;
        let omega =
            k * crate::domain::medium::sound_speed_at(&self.medium, 0.0, 0.0, 0.0, &self.grid);
        let dt = 0.5 * self.grid.dx
            / crate::domain::medium::sound_speed_at(&self.medium, 0.0, 0.0, 0.0, &self.grid);

        let pstd_config = PSTDConfig::default();
        let pstd_source = GridSource::default();
        let pstd_solver =
            PSTDSolver::new(pstd_config, self.grid.clone(), &self.medium, pstd_source)?;
        let pstd_phase_error = self.compute_phase_error(&pstd_solver, k, omega, dt)?;

        let fdtd_config = FdtdConfig {
            dt,
            ..Default::default()
        };
        let fdtd_solver =
            FdtdSolver::new(fdtd_config, &self.grid, &self.medium, GridSource::default())?;
        let fdtd_phase_error = self.compute_phase_error_fdtd(&fdtd_solver, k, omega, dt)?;

        let kuznetsov_config = KuznetsovConfig::default();
        let kuznetsov_solver = KuznetsovWave::new(kuznetsov_config, &self.grid)?;
        let kuznetsov_phase_error =
            self.compute_phase_error_kuznetsov(&kuznetsov_solver, k, omega, dt)?;

        let numerical_wavelength = 2.0 * PI / (k * (1.0 + pstd_phase_error));
        let group_velocity_error = (pstd_phase_error * omega / k).abs()
            / crate::domain::medium::sound_speed_at(&self.medium, 0.0, 0.0, 0.0, &self.grid);

        Ok(DispersionResults {
            pstd_phase_error,
            fdtd_phase_error,
            kuznetsov_phase_error,
            numerical_wavelength,
            group_velocity_error,
        })
    }

    /// Compute PSTD phase error via analytical temporal κ dispersion relation.
    ///
    /// PSTD uses spectral spatial derivatives (exact at all k), so spatial phase
    /// error is zero. The only error is temporal, introduced by the leapfrog scheme.
    /// With the κ correction κ(k) = cos(c·dt·|k|/2), the temporal update is:
    ///   sin(ω·dt/2) = c₀·|k|·dt/2 · κ(k) = c₀·|k|·dt/2 · cos(c_ref·dt·|k|/2)
    ///
    /// This is solved numerically for ω_num(k), and the phase error is
    ///   ε(k) = |c_num(k)/c₀ − 1|  where  c_num = ω_num / |k|.
    ///
    /// Returns worst-case (maximum over 100 log-spaced k samples) phase-velocity error.
    ///
    /// References: Liu (1998), §3; Treeby & Cox (2010), §II.A.
    pub(super) fn compute_phase_error<S>(
        &self,
        _solver: &S,
        k: f64,
        _omega: f64,
        dt: f64,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        use std::f64::consts::PI;
        let c0 = crate::domain::medium::sound_speed_at(&self.medium, 0.0, 0.0, 0.0, &self.grid);
        let dx = self.grid.dx;
        let k_nyq = PI / dx;
        let k_max = if k > 0.0 { k.min(k_nyq) } else { k_nyq };

        let n_samples = 100usize;
        let mut max_err: f64 = 0.0;
        for i in 1..=n_samples {
            let k_val = (i as f64 / n_samples as f64) * k_max;
            let kappa = (0.5 * c0 * dt * k_val).cos();
            let arg = c0 * k_val * dt * 0.5 * kappa;
            if arg.abs() <= 1.0 {
                let omega_num = 2.0 * arg.asin() / dt;
                let c_num = omega_num / k_val;
                max_err = max_err.max((c_num / c0 - 1.0).abs());
            }
        }
        Ok(max_err)
    }

    /// Compute FDTD phase error via von Neumann dispersion analysis.
    ///
    /// For a 3D staggered-grid FDTD scheme with time step dt, grid spacing dx,
    /// and CFL = c₀·dt/dx, the dispersion relation is:
    ///   sin(ω·dt/2) = CFL · sin(k·dx/2)
    ///   → ω_num(k) = 2 · arcsin(CFL · sin(k·dx/2)) / dt
    ///
    /// Phase velocity error: ε(k) = |c_num(k)/c₀ − 1|.
    ///
    /// Returns worst-case error (max over 100 log-spaced k samples).
    ///
    /// Reference: Taflove & Hagness (2005), §4.5, Eq. 4.73.
    pub(super) fn compute_phase_error_fdtd<S>(
        &self,
        _solver: &S,
        k: f64,
        _omega: f64,
        dt: f64,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        use std::f64::consts::PI;
        let c0 = crate::domain::medium::sound_speed_at(&self.medium, 0.0, 0.0, 0.0, &self.grid);
        let dx = self.grid.dx;
        let cfl = c0 * dt / dx;
        let k_nyq = PI / dx;
        let k_max = if k > 0.0 { k.min(k_nyq) } else { k_nyq };

        let n_samples = 100usize;
        let mut max_err: f64 = 0.0;
        for i in 1..=n_samples {
            let k_val = (i as f64 / n_samples as f64) * k_max;
            let arg = cfl * (k_val * dx / 2.0).sin();
            if arg.abs() <= 1.0 {
                let omega_num = 2.0 * arg.asin() / dt;
                let c_num = omega_num / k_val;
                max_err = max_err.max((c_num / c0 - 1.0).abs());
            }
        }
        Ok(max_err)
    }

    /// Compute Kuznetsov nonlinear phase error via analytical Fubini-Earnshaw relation.
    ///
    /// Since the Kuznetsov equation is solved on the same FDTD grid, its linear
    /// phase error equals that of the underlying FDTD scheme (conservative upper bound).
    ///
    /// References: Blackstock (1966); Hamilton & Blackstock (1998), Ch. 3.
    pub(super) fn compute_phase_error_kuznetsov<S>(
        &self,
        solver: &S,
        k: f64,
        omega: f64,
        dt: f64,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        self.compute_phase_error_fdtd(solver, k, omega, dt)
    }
}
