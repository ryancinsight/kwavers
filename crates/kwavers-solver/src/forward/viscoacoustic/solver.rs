//! N-dimensional pseudospectral memory-variable viscoacoustic solver.
//!
//! One canonical implementation covers 1-D, 2-D, and 3-D: a `(n,1,1)` grid is
//! 1-D, `(nx,ny,1)` is 2-D, and `(nx,ny,nz)` is full 3-D — the spectral
//! derivative along a singleton axis is identically zero, so the lower-D cases
//! reduce exactly with no special-casing.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d};
use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;
use ndarray::{Array3, Axis, Zip};
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

/// Time-domain memory-variable viscoacoustic solver (1-D/2-D/3-D, pseudospectral).
///
/// Construct from a [`GeneralizedMaxwellModel`] (the relaxation spectrum) or from
/// raw moduli, then call [`Self::step`] to advance the velocity–pressure state.
#[derive(Clone)]
pub struct ViscoacousticMemorySolver {
    nx: usize,
    ny: usize,
    nz: usize,
    cell_volume: f64, // dx·dy·dz
    dt: f64,
    rho: f64,
    /// Unrelaxed (instantaneous) modulus `M_U = M_∞ + Σ ΔMₗ` \[Pa].
    m_u: f64,
    /// Equilibrium (relaxed) modulus `M_∞` \[Pa] — sets the potential-energy norm.
    m_inf: f64,
    arms: Vec<Arm>,

    // Spectral derivative: apollo's batched, cache-tiled, parallel per-axis 3-D
    // FFT (forward_axis → ·ik → inverse_axis) reusing one complex scratch.
    fft: Arc<Fft3d>,
    kx: Vec<f64>,
    ky: Vec<f64>,
    kz: Vec<f64>,
    cbuf: Array3<Complex64>,

    // State.
    p: Array3<f64>,
    vx: Array3<f64>,
    vy: Array3<f64>,
    vz: Array3<f64>,
    sigma: Vec<Array3<f64>>, // one memory field per arm

    // Preallocated derivative buffers (gx is reused as the divergence ∇·v and gy
    // as the relaxation accumulator after the velocity-divergence pass).
    gx: Array3<f64>,
    gy: Array3<f64>,
    gz: Array3<f64>,

    // Optional absorbing boundary: per-cell multiplicative decay `exp(-γ Δt)`
    // applied to `p` and `v` each step, with `γ` ramping up inside the boundary
    // layer and zero in the interior (`None` ⇒ periodic, non-absorbing).
    damping_decay: Option<Array3<f64>>,
}

impl std::fmt::Debug for ViscoacousticMemorySolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ViscoacousticMemorySolver")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .field("dt", &self.dt)
            .field("rho", &self.rho)
            .field("m_u", &self.m_u)
            .field("arms", &self.arms.len())
            .finish()
    }
}

impl ViscoacousticMemorySolver {
    /// Build from raw parameters: grid `(nx,ny,nz)` with spacings `(dx,dy,dz)`,
    /// time step `dt`, density `ρ`, equilibrium modulus `M_∞`, and relaxation
    /// arms `(ΔMₗ, τₗ)`. An empty arm list yields the lossless wave equation.
    /// # Errors
    /// - Any zero dimension, non-positive `dx`/`dy`/`dz`/`dt`/`ρ`/`M_∞`, or a
    ///   non-positive arm parameter.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
        rho: f64,
        m_inf: f64,
        arms: &[(f64, f64)],
    ) -> KwaversResult<Self> {
        if nx == 0
            || ny == 0
            || nz == 0
            || dx <= 0.0
            || dy <= 0.0
            || dz <= 0.0
            || dt <= 0.0
            || rho <= 0.0
            || m_inf <= 0.0
        {
            return Err(KwaversError::InvalidInput(
                "viscoacoustic solver requires positive grid, spacings, dt, ρ, M_∞".to_owned(),
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

        let shape = (nx, ny, nz);
        Ok(Self {
            nx,
            ny,
            nz,
            cell_volume: dx * dy * dz,
            dt,
            rho,
            m_u,
            m_inf,
            arms: built.clone(),
            fft: get_fft_for_grid(nx, ny, nz),
            kx: fft_wavenumbers(nx, dx),
            ky: fft_wavenumbers(ny, dy),
            kz: fft_wavenumbers(nz, dz),
            cbuf: Array3::from_elem(shape, Complex64::new(0.0, 0.0)),
            p: Array3::zeros(shape),
            vx: Array3::zeros(shape),
            vy: Array3::zeros(shape),
            vz: Array3::zeros(shape),
            sigma: vec![Array3::zeros(shape); built.len()],
            gx: Array3::zeros(shape),
            gy: Array3::zeros(shape),
            gz: Array3::zeros(shape),
            damping_decay: None,
        })
    }

    /// Enable an **absorbing boundary layer** (sponge) of `thickness` cells on
    /// every face whose axis is long enough to hold it. Outgoing waves entering
    /// the layer are damped before they reach (and wrap around) the periodic
    /// boundary, suppressing artificial reflections.
    ///
    /// `gamma_max` \[s⁻¹] is the peak damping rate at the outermost cell; the
    /// rate ramps in as a quadratic profile `γ(d) = γ_max ((L-d)/L)²` over the
    /// layer depth `d ∈ [0, L)` (zero in the interior), summed across axes so
    /// corners damp in every direction. A smooth ramp keeps layer reflection low.
    /// Calling again rebuilds the profile; `thickness = 0` disables it.
    pub fn enable_absorbing_layer(&mut self, thickness: usize, gamma_max: f64) {
        if thickness == 0 || gamma_max <= 0.0 {
            self.damping_decay = None;
            return;
        }
        // Per-axis ramp: γ contribution at index `i` along an axis of extent `n`.
        let ramp = |i: usize, n: usize| -> f64 {
            if n <= 2 * thickness {
                return 0.0; // axis too short to host the layer (e.g. singleton)
            }
            let l = thickness as f64;
            let depth = if i < thickness {
                (thickness - i) as f64 // distance from the low boundary
            } else if i >= n - thickness {
                (i - (n - thickness) + 1) as f64 // distance from the high boundary
            } else {
                0.0
            };
            let frac = depth / l;
            gamma_max * frac * frac
        };

        let dt = self.dt;
        let decay = Array3::from_shape_fn((self.nx, self.ny, self.nz), |(i, j, k)| {
            let gamma = ramp(i, self.nx) + ramp(j, self.ny) + ramp(k, self.nz);
            (-gamma * dt).exp()
        });
        self.damping_decay = Some(decay);
    }

    /// 1-D convenience constructor (`ny = nz = 1`).
    /// # Errors
    /// - Propagates [`Self::new`] validation failures.
    pub fn new_1d(
        n: usize,
        dx: f64,
        dt: f64,
        rho: f64,
        m_inf: f64,
        arms: &[(f64, f64)],
    ) -> KwaversResult<Self> {
        Self::new(n, 1, 1, dx, 1.0, 1.0, dt, rho, m_inf, arms)
    }

    /// Build from a [`GeneralizedMaxwellModel`] (its `M_∞`, arms, and density)
    /// plus the grid and time step.
    /// # Errors
    /// - Propagates [`Self::new`] validation failures.
    #[allow(clippy::too_many_arguments)]
    pub fn from_generalized_maxwell(
        model: &GeneralizedMaxwellModel,
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        dt: f64,
    ) -> KwaversResult<Self> {
        Self::new(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
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
    pub fn pressure(&self) -> &Array3<f64> {
        &self.p
    }

    /// Overwrite the pressure field (shape must match the grid); resets velocity
    /// and memory variables to zero. Used to set an initial condition.
    /// # Errors
    /// - Shape mismatch.
    pub fn set_pressure(&mut self, p: &Array3<f64>) -> KwaversResult<()> {
        if p.shape() != [self.nx, self.ny, self.nz] {
            return Err(KwaversError::InvalidInput(
                "pressure shape must equal the grid".to_owned(),
            ));
        }
        self.p.assign(p);
        self.vx.fill(0.0);
        self.vy.fill(0.0);
        self.vz.fill(0.0);
        for s in &mut self.sigma {
            s.fill(0.0);
        }
        Ok(())
    }

    /// Acoustic energy `Σ [p²/(2M_∞) + ρ|v|²/2] ΔV` \[J]. Conserved (to leapfrog
    /// round-off) for the lossless medium; decays monotonically with relaxation.
    #[must_use]
    pub fn energy(&self) -> f64 {
        let pe: f64 = self.p.iter().map(|&p| p * p).sum::<f64>() / (2.0 * self.m_inf);
        let ke: f64 = self
            .vx
            .iter()
            .chain(self.vy.iter())
            .chain(self.vz.iter())
            .map(|&v| v * v)
            .sum::<f64>()
            * (self.rho / 2.0);
        (pe + ke) * self.cell_volume
    }

    /// Spectral derivative `∂field/∂xₐ → out` via apollo's batched tiled per-axis
    /// 3-D FFT: forward along `axis`, multiply by `i·k`, inverse along `axis`.
    /// Reuses the owned complex scratch `cbuf` (no allocation).
    fn axis_derivative(
        fft: &Fft3d,
        k: &[f64],
        axis: usize,
        field: &Array3<f64>,
        cbuf: &mut Array3<Complex64>,
        out: &mut Array3<f64>,
    ) {
        Zip::from(&mut *cbuf)
            .and(field)
            .for_each(|c, &f| *c = Complex64::new(f, 0.0));
        fft.forward_axis_complex_inplace(cbuf, axis);
        for (m, mut lane) in cbuf.axis_iter_mut(Axis(axis)).enumerate() {
            let factor = Complex64::new(0.0, k[m]);
            lane.mapv_inplace(|v| v * factor);
        }
        fft.inverse_axis_complex_inplace(cbuf, axis);
        Zip::from(&mut *out).and(&*cbuf).for_each(|o, c| *o = c.re);
    }

    /// Advance the state by one time step `Δt`.
    pub fn step(&mut self) {
        // 1. Velocity half-step: v += -(Δt/ρ) ∇p (per component).
        let cv = -self.dt / self.rho;
        Self::axis_derivative(&self.fft, &self.kx, 0, &self.p, &mut self.cbuf, &mut self.gx);
        Self::axis_derivative(&self.fft, &self.ky, 1, &self.p, &mut self.cbuf, &mut self.gy);
        Self::axis_derivative(&self.fft, &self.kz, 2, &self.p, &mut self.cbuf, &mut self.gz);
        Zip::from(&mut self.vx).and(&self.gx).for_each(|v, &g| *v += cv * g);
        Zip::from(&mut self.vy).and(&self.gy).for_each(|v, &g| *v += cv * g);
        Zip::from(&mut self.vz).and(&self.gz).for_each(|v, &g| *v += cv * g);

        // 2. Dilatation rate D = ∇·v (accumulated into gx).
        Self::axis_derivative(&self.fft, &self.kx, 0, &self.vx, &mut self.cbuf, &mut self.gx);
        Self::axis_derivative(&self.fft, &self.ky, 1, &self.vy, &mut self.cbuf, &mut self.gy);
        Self::axis_derivative(&self.fft, &self.kz, 2, &self.vz, &mut self.cbuf, &mut self.gz);
        Zip::from(&mut self.gx)
            .and(&self.gy)
            .and(&self.gz)
            .for_each(|d, &y, &z| *d += y + z); // gx = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z

        // 3. Advance each σ_l with the exact exponential integrator and accumulate
        //    its trapezoidal pressure contribution into gy (reused as the relax sum).
        self.gy.fill(0.0);
        for (arm, sigma) in self.arms.iter().zip(self.sigma.iter_mut()) {
            let arm = *arm;
            Zip::from(sigma)
                .and(&self.gx)
                .and(&mut self.gy)
                .for_each(|s, &d, r| {
                    let old = *s;
                    let new = arm.decay.mul_add(old, -arm.delta_m * d * arm.forcing);
                    *r += 0.5 * (old + new) * arm.inv_tau;
                    *s = new;
                });
        }

        // 4. Pressure update: p += -Δt (M_U D + Σ_l ½(σ_l+σ_l^new)/τ_l).
        let (m_u, dt) = (self.m_u, self.dt);
        Zip::from(&mut self.p)
            .and(&self.gx)
            .and(&self.gy)
            .for_each(|p, &d, &relax| *p -= dt * (m_u * d + relax));

        // 5. Absorbing boundary: damp p and v inside the sponge layer.
        if let Some(decay) = &self.damping_decay {
            Zip::from(&mut self.p).and(decay).for_each(|p, &d| *p *= d);
            Zip::from(&mut self.vx).and(decay).for_each(|v, &d| *v *= d);
            Zip::from(&mut self.vy).and(decay).for_each(|v, &d| *v *= d);
            Zip::from(&mut self.vz).and(decay).for_each(|v, &d| *v *= d);
        }
    }
}

/// FFT-order signed wavenumbers `k[m] = 2π m'/(n·Δx)` with `m' = m` for `m < n/2`
/// and `m' = m − n` otherwise. For `n = 1` this is `[0]` (derivative along a
/// singleton axis is zero).
fn fft_wavenumbers(n: usize, dx: f64) -> Vec<f64> {
    let norm = TWO_PI / (n as f64 * dx);
    (0..n)
        .map(|m| {
            let signed = if m < n / 2 { m as f64 } else { m as f64 - n as f64 };
            signed * norm
        })
        .collect()
}
