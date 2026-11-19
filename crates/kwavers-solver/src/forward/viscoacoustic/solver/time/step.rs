use crate::forward::viscoacoustic::solver::Coeff;
use crate::forward::viscoacoustic::solver::ViscoacousticMemorySolver;
use kwavers_math::fft::Complex64;
use kwavers_math::fft::Fft3d;
use leto::Array3;

impl ViscoacousticMemorySolver {
    #[inline(always)]
    fn update_velocity_half_step<const AXIS: usize>(
        fft: &Fft3d,
        k: &[f64],
        pressure: &Array3<f64>,
        cbuf: &mut Array3<Complex64>,
        grad: &mut Array3<f64>,
        velocity: &mut Array3<f64>,
        inv_rho: &Coeff,
        dt: f64,
    ) {
        Self::axis_derivative(fft, k, AXIS, pressure, cbuf, grad);
        match inv_rho {
            Coeff::Uniform(ir) => {
                for (v, &g) in velocity.iter_mut().zip(grad.iter()) {
                    *v += -dt * ir * g;
                }
            }
            Coeff::Field(inv_rho) => leto_ops::zip_mut_with(
                velocity.view_mut(),
                (&grad.view(), &inv_rho.view()),
                |v, (&g, &ir)| *v += -dt * ir * g,
            )
            .expect("invariant: velocity update fields share grid shape"),
        }
    }

    /// Advance the state by one time step `Δt`.
    ///
    /// # Panics
    ///
    /// Panics if a caller-supplied shape or an internal solver state violates
    /// the precondition required by this operation.
    pub fn step(&mut self) {
        // 1. Velocity half-step: v += -(Δt/ρ(x)) ∇p (per component, per voxel).
        // Staging slots are assigned in canonical axis order — first active
        // axis → `gx`, second → `gy`, third → `gz` — so every layout stages
        // into grid-shaped buffers only (`gz` exists exactly in the layout
        // whose step needs a third slot) and each slot is consumed by its
        // velocity update before the divergence pass restages it. An inactive
        // axis stages no derivative and updates no velocity: its component is
        // the exact positive zero for all time (the value the full-storage
        // layout wrote and then added zero to).
        let dt = self.dt;
        let axes = self.axes;
        let slot_x: Option<usize> = axes.x.then_some(0);
        let slot_y: Option<usize> = axes.y.then_some(usize::from(axes.x));
        let slot_z: Option<usize> = axes.z.then_some(usize::from(axes.x) + usize::from(axes.y));
        {
            let buffers: [&mut Array3<f64>; 3] = [&mut self.gx, &mut self.gy, &mut self.gz];
            // The velocity update is `v += -(Δt·inv_rho)·∂p/∂α`: the branch
            // on the coefficient representation happens here, once, with the
            // scalar hoisted out of the flat pass (identical operations and
            // order to the grid fold over a constant field).
            if let Some(s) = slot_x {
                Self::update_velocity_half_step::<0>(
                    &self.fft,
                    &self.kx,
                    &self.p,
                    &mut self.cbuf,
                    &mut *buffers[s],
                    &mut self.vx,
                    &self.inv_rho,
                    dt,
                );
            }
            if let Some(s) = slot_y {
                Self::update_velocity_half_step::<1>(
                    &self.fft,
                    &self.ky,
                    &self.p,
                    &mut self.cbuf,
                    &mut *buffers[s],
                    &mut self.vy,
                    &self.inv_rho,
                    dt,
                );
            }
            if let Some(s) = slot_z {
                Self::update_velocity_half_step::<2>(
                    &self.fft,
                    &self.kz,
                    &self.p,
                    &mut self.cbuf,
                    &mut *buffers[s],
                    &mut self.vz,
                    &self.inv_rho,
                    dt,
                );
            }

            // 2. Dilatation rate D = ∇·v, staged into the same slots: the
            // first active axis's derivative lands in `gx` — the divergence
            // output — and the rest in `gy`/`gz`. A missing axis contributes
            // its exact positive-zero derivative (the identity the
            // full-storage layout derives from the zero velocity and the
            // singleton wavenumber).
            if let Some(s) = slot_x {
                Self::axis_derivative(
                    &self.fft,
                    &self.kx,
                    0,
                    &self.vx,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
            }
            if let Some(s) = slot_y {
                Self::axis_derivative(
                    &self.fft,
                    &self.ky,
                    1,
                    &self.vy,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
            }
            if let Some(s) = slot_z {
                Self::axis_derivative(
                    &self.fft,
                    &self.kz,
                    2,
                    &self.vz,
                    &mut self.cbuf,
                    &mut *buffers[s],
                );
            }
        }

        // gx = ∂vx/∂x + (∂vy/∂y + ∂vz/∂z): the reference association, in one
        // flat combine pass per layout. With a single active axis the staged
        // derivative already is the divergence; with none it is the zero fill.
        match axes.count() {
            0 => self.gx.fill(0.0),
            1 => {}
            2 => leto_ops::zip_mut_with(self.gx.view_mut(), &self.gy.view(), |d_, &y| {
                *d_ += y;
            })
            .expect("invariant: divergence components share grid shape"),
            _ => leto_ops::zip_mut_with(
                self.gx.view_mut(),
                (&self.gy.view(), &self.gz.view()),
                |d_, (&y, &z)| *d_ += y + z,
            )
            .expect("invariant: divergence components share grid shape"),
        }

        // 3. Advance each σ_l with the exact exponential integrator and
        //    accumulate its trapezoidal pressure contribution into gy (reused
        //    as the relax sum): σ_new = decay·σ + gain·D. The branch on the
        //    arm's coefficient representation happens once per arm; the
        //    scalar path hoists all three coefficients and runs the same
        //    operations in the same order as the grid path.
        self.gy.fill(0.0);
        for (arm, sigma) in self.arms.iter().zip(self.sigma.iter_mut()) {
            let s_slice = sigma
                .as_slice_mut()
                .expect("invariant: viscoacoustic σ arm is contiguous");
            let gx_slice = self
                .gx
                .as_slice()
                .expect("invariant: viscoacoustic gx is contiguous");
            let gy_slice = self
                .gy
                .as_slice_mut()
                .expect("invariant: viscoacoustic gy is contiguous");
            if let (Coeff::Uniform(decay), Coeff::Uniform(gain), Coeff::Uniform(inv_tau)) =
                (&arm.decay, &arm.gain, &arm.inv_tau)
            {
                let (decay, gain, inv_tau) = (*decay, *gain, *inv_tau);
                for idx in 0..s_slice.len() {
                    let old = s_slice[idx];
                    let new = decay.mul_add(old, gain * gx_slice[idx]);
                    gy_slice[idx] += 0.5 * (old + new) * inv_tau;
                    s_slice[idx] = new;
                }
                continue;
            }
            let decay_slice = arm.decay.slice();
            let gain_slice = arm.gain.slice();
            let inv_tau_slice = arm.inv_tau.slice();
            for idx in 0..s_slice.len() {
                let old = s_slice[idx];
                let new = decay_slice[idx].mul_add(old, gain_slice[idx] * gx_slice[idx]);
                gy_slice[idx] += 0.5 * (old + new) * inv_tau_slice[idx];
                s_slice[idx] = new;
            }
        }

        // 4. Pressure update: p += -Δt (M_U(x) D + Σ_l ½(σ_l+σ_l^new)/τ_l(x)).
        if let Coeff::Uniform(mu) = &self.m_u {
            let mu = *mu;
            let p_slice = self
                .p
                .as_slice_mut()
                .expect("invariant: viscoacoustic p is contiguous");
            let gx_slice = self
                .gx
                .as_slice()
                .expect("invariant: viscoacoustic gx is contiguous");
            let gy_slice = self
                .gy
                .as_slice_mut()
                .expect("invariant: viscoacoustic gy is contiguous");
            for idx in 0..p_slice.len() {
                p_slice[idx] -= dt * (mu * gx_slice[idx] + gy_slice[idx]);
            }
        } else {
            let Coeff::Field(m_u) = &self.m_u else {
                unreachable!("invariant: grid path reads field coefficients")
            };
            leto_ops::zip_mut_with(
                self.p.view_mut(),
                (&self.gx.view(), &self.gy.view(), &m_u.view()),
                |p, (&d, &relax, &mu)| *p -= dt * (mu * d + relax),
            )
            .expect("invariant: pressure update fields share grid shape");
        }

        // 5. Soft pressure sources: p[index] += signal[step].
        for (index, signal) in &self.pressure_sources {
            if let Some(&s) = signal.get(self.step_count) {
                self.p[[index.0, index.1, index.2]] += s;
            }
        }

        // 6. Absorbing boundary: damp p and v inside the sponge layer. Each
        // active axis is damped exactly once here (the removed inactive-axis
        // pass added only a multiply by one over a zero field).
        if let Some(decay) = &self.damping_decay {
            leto_ops::zip_mut_with(self.p.view_mut(), &decay.view(), |p, &d| *p *= d)
                .expect("invariant: pressure and damping fields share grid shape");
            if axes.x {
                leto_ops::zip_mut_with(self.vx.view_mut(), &decay.view(), |v, &d| *v *= d)
                    .expect("invariant: velocity-x and damping fields share grid shape");
            }
            if axes.y {
                leto_ops::zip_mut_with(self.vy.view_mut(), &decay.view(), |v, &d| *v *= d)
                    .expect("invariant: velocity-y and damping fields share grid shape");
            }
            if axes.z {
                leto_ops::zip_mut_with(self.vz.view_mut(), &decay.view(), |v, &d| *v *= d)
                    .expect("invariant: velocity-z and damping fields share grid shape");
            }
        }

        // 7. Record sensor traces, then advance the simulation clock.
        for (trace, &index) in self
            .sensor_record
            .iter_mut()
            .zip(self.pressure_sensors.iter())
        {
            trace.push(self.p[[index.0, index.1, index.2]]);
        }
        self.step_count += 1;
    }
}
