//! 1-D pseudospectral memory-variable viscoacoustic solver.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{Complex64, Fft1d, Shape1D, FFT_CACHE_1D};
use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;
use ndarray::Array1;
use std::sync::Arc;

/// One relaxation arm with its precomputed exponential-integrator coefficients.
#[derive(Debug, Clone, Copy)]
struct Arm {
    /// Relaxation strength `ΔMₗ` \[Pa].
    delta_m: f64,
    /// `1/τₗ` \[s⁻¹].
    inv_tau: f64,
    /// `e^{-Δt/τₗ}` (decay over one step).
    decay: f64,
    /// `τₗ (1 − e^{-Δt/τₗ})` — the forcing weight in the exponential integrator.
    forcing: f64,
}

/// Time-domain memory-variable viscoacoustic solver (1-D, pseudospectral).
///
/// Construct from a [`GeneralizedMaxwellModel`] (the relaxation spectrum) or from
/// raw moduli, then call [`Self::step`] to advance the velocity–pressure state.
#[derive(Clone)]
pub struct ViscoacousticMemorySolver {
    n: usize,
    dx: f64,
    dt: f64,
    rho: f64,
    /// Unrelaxed (instantaneous) modulus `M_U = M_∞ + Σ ΔMₗ` \[Pa].
    m_u: f64,
    /// Equilibrium (relaxed) modulus `M_∞` \[Pa] — sets the potential energy norm.
    m_inf: f64,
    arms: Vec<Arm>,

    // State (all length n).
    p: Array1<f64>,
    v: Array1<f64>,
    sigma: Vec<Array1<f64>>, // one memory field per arm

    // Preallocated work buffers (no per-step allocation).
    dvdx: Array1<f64>,
    dpdx: Array1<f64>,
    fft: Arc<Fft1d>,
    ikd: Vec<Complex64>,         // i·k spectral first-derivative multiplier
    scratch: Array1<Complex64>,  // FFT line buffer
}

impl std::fmt::Debug for ViscoacousticMemorySolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViscoacousticMemorySolver")
            .field("n", &self.n)
            .field("dx", &self.dx)
            .field("dt", &self.dt)
            .field("rho", &self.rho)
            .field("m_u", &self.m_u)
            .field("arms", &self.arms.len())
            .finish()
    }
}

impl ViscoacousticMemorySolver {
    /// Build from raw parameters: grid `n`/`dx`, time step `dt`, density `ρ`,
    /// equilibrium modulus `M_∞`, and relaxation arms `(ΔMₗ, τₗ)`.
    ///
    /// An empty arm list yields the lossless wave equation (modulus `M_∞`).
    /// # Errors
    /// - `n == 0`, or non-positive `dx`/`dt`/`ρ`/`M_∞`, or any non-positive arm.
    pub fn new(
        n: usize,
        dx: f64,
        dt: f64,
        rho: f64,
        m_inf: f64,
        arms: &[(f64, f64)],
    ) -> KwaversResult<Self> {
        if n == 0 || dx <= 0.0 || dt <= 0.0 || rho <= 0.0 || m_inf <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "viscoacoustic solver requires n>0 and positive dx, dt, ρ, M_∞".to_owned(),
            ));
        }
        if arms.iter().any(|&(dm, tau)| dm <= 0.0 || tau <= 0.0) {
            return Err(KwaversError::InvalidInput(
                "relaxation arms require ΔM>0 and τ>0".to_owned(),
            ));
        }

        let m_u = m_inf + arms.iter().map(|&(dm, _)| dm).sum::<f64>();
        let built: Vec<Arm> = arms
            .iter()
            .map(|&(delta_m, tau)| {
                let decay = (-dt / tau).exp();
                Arm {
                    delta_m,
                    inv_tau: 1.0 / tau,
                    decay,
                    forcing: tau * (1.0 - decay),
                }
            })
            .collect();

        // Spectral i·k multiplier (FFT wavenumber order).
        let norm = TWO_PI / (n as f64 * dx);
        let ikd: Vec<Complex64> = (0..n)
            .map(|i| {
                let m = if i < n / 2 { i as f64 } else { i as f64 - n as f64 };
                Complex64::new(0.0, m * norm)
            })
            .collect();

        let sigma = vec![Array1::zeros(n); built.len()];
        Ok(Self {
            n,
            dx,
            dt,
            rho,
            m_u,
            m_inf,
            arms: built,
            p: Array1::zeros(n),
            v: Array1::zeros(n),
            sigma,
            dvdx: Array1::zeros(n),
            dpdx: Array1::zeros(n),
            fft: FFT_CACHE_1D.get_or_create(Shape1D { n }),
            ikd,
            scratch: Array1::from_elem(n, Complex64::default()),
        })
    }

    /// Build from a [`GeneralizedMaxwellModel`] (its `M_∞`, arms `(ΔMₗ, τₗ)`, and
    /// density define the medium) plus the grid and time step.
    /// # Errors
    /// - Propagates [`Self::new`] validation failures.
    pub fn from_generalized_maxwell(
        model: &GeneralizedMaxwellModel,
        n: usize,
        dx: f64,
        dt: f64,
    ) -> KwaversResult<Self> {
        Self::new(
            n,
            dx,
            dt,
            model.density(),
            model.equilibrium_modulus(),
            model.arms(),
        )
    }

    /// Unrelaxed (high-frequency) sound speed `√(M_U/ρ)` \[m·s⁻¹] — the CFL speed.
    #[must_use]
    pub fn unrelaxed_speed(&self) -> f64 {
        (self.m_u / self.rho).sqrt()
    }

    /// Pressure field \[Pa].
    #[must_use]
    pub fn pressure(&self) -> &Array1<f64> {
        &self.p
    }

    /// Overwrite the pressure field (length must equal `n`); resets velocity and
    /// memory to zero. Used to set an initial condition.
    /// # Errors
    /// - Length mismatch.
    pub fn set_pressure(&mut self, p: &Array1<f64>) -> KwaversResult<()> {
        if p.len() != self.n {
            return Err(KwaversError::InvalidInput(
                "pressure length must equal grid size".to_owned(),
            ));
        }
        self.p.assign(p);
        self.v.fill(0.0);
        for s in &mut self.sigma {
            s.fill(0.0);
        }
        Ok(())
    }

    /// Acoustic energy `Σₓ [p²/(2M_∞) + ρv²/2] Δx` \[J·m⁻²]. Conserved (to
    /// leapfrog round-off) for the lossless medium; decays monotonically with
    /// relaxation.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let pe: f64 = self.p.iter().map(|&p| p * p).sum::<f64>() / (2.0 * self.m_inf);
        let ke: f64 = self.v.iter().map(|&v| v * v).sum::<f64>() * (self.rho / 2.0);
        (pe + ke) * self.dx
    }

    /// In-place spectral first derivative `∂field/∂x → out` (periodic).
    fn derivative(
        fft: &Fft1d,
        ikd: &[Complex64],
        scratch: &mut Array1<Complex64>,
        field: &Array1<f64>,
        out: &mut Array1<f64>,
    ) {
        for (s, &f) in scratch.iter_mut().zip(field.iter()) {
            *s = Complex64::new(f, 0.0);
        }
        fft.forward_complex_inplace(scratch);
        for (s, &ik) in scratch.iter_mut().zip(ikd.iter()) {
            *s *= ik;
        }
        fft.inverse_complex_inplace(scratch);
        for (o, s) in out.iter_mut().zip(scratch.iter()) {
            *o = s.re;
        }
    }

    /// Advance the state by one time step `Δt`.
    pub fn step(&mut self) {
        // 1. Velocity half-step: v += -(Δt/ρ) ∂p/∂x.
        Self::derivative(&self.fft, &self.ikd, &mut self.scratch, &self.p, &mut self.dpdx);
        let cv = -self.dt / self.rho;
        for (v, &dpdx) in self.v.iter_mut().zip(self.dpdx.iter()) {
            *v += cv * dpdx;
        }

        // 2. Dilatation rate D = ∂v/∂x at the half step.
        Self::derivative(&self.fft, &self.ikd, &mut self.scratch, &self.v, &mut self.dvdx);

        // 3. Pressure update with the trapezoidal relaxation contribution:
        //    p += -Δt (M_U D + Σ_l ½(σ_l + σ_l^new)/τ_l), advancing each σ_l with
        //    the exact exponential integrator σ_l^new = decay σ_l − ΔM_l D τ_l(1−decay).
        let dt = self.dt;
        let m_u = self.m_u;
        for idx in 0..self.n {
            let d = self.dvdx[idx];
            let mut relax = 0.0;
            for (arm, sigma) in self.arms.iter().zip(self.sigma.iter_mut()) {
                let old = sigma[idx];
                let new = arm.decay.mul_add(old, -arm.delta_m * d * arm.forcing);
                relax += 0.5 * (old + new) * arm.inv_tau;
                sigma[idx] = new;
            }
            self.p[idx] -= dt * (m_u * d + relax);
        }
    }
}
