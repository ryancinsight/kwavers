use crate::forward::viscoacoustic::solver::fft_wavenumbers;
use crate::forward::viscoacoustic::solver::ActiveAxes;
use crate::forward::viscoacoustic::solver::Arm;
use crate::forward::viscoacoustic::solver::Coeff;
use crate::forward::viscoacoustic::solver::ViscoacousticGrid;
use crate::forward::viscoacoustic::solver::ViscoacousticMemorySolver;
use kwavers_math::fft::get_fft_for_grid;
use kwavers_math::fft::Complex64;
use leto::Array3;

impl ViscoacousticMemorySolver {
    /// Allocate state/scratch and assemble the solver from prepared per-voxel
    /// `inv_rho`, `m_inf`, and arm coefficient fields. `M_U = M_∞ + Σ ΔMₗ` is
    /// recovered from the arm gains; the CFL speed is the grid max of `√(M_U/ρ)`.
    ///
    /// `axes` is the storage mask. Production constructors derive it from the
    /// grid; the differential tests inject the all-active mask at a singleton
    /// grid to build the six-array reference the acceptance oracle compares
    /// against. The step code is identical under both layouts — an inactive
    /// axis is the exact positive-zero identity — so the two are bitwise
    /// comparable, and the injected reference is test-local, not a public
    /// constructor surface.
    pub(in crate::forward::viscoacoustic::solver) fn assemble(
        grid: ViscoacousticGrid,
        inv_rho: Coeff,
        m_inf: Coeff,
        arms: Vec<Arm>,
        axes: ActiveAxes,
    ) -> Self {
        let ViscoacousticGrid {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
        } = grid;

        let shape = (nx, ny, nz);
        // M_U(x) = M_∞(x) + Σ ΔMₗ(x); recover ΔMₗ = −gain / (τ(1−decay)) = −gain·inv_tau/(1−decay).
        // A uniform base with uniform arms stays scalar; anything per-voxel
        // produces the grid. The scalar sum applies the same per-arm terms in
        // the same order as the grid pass, so the two agree bitwise.
        let m_u = if m_inf.is_uniform() && arms.iter().all(|arm| arm.gain.is_uniform()) {
            let mut u = m_inf.value();
            for arm in &arms {
                u += -arm.gain.value() * arm.inv_tau.value() / (1.0 - arm.decay.value());
            }
            Coeff::Uniform(u)
        } else {
            let mut m_u = match &m_inf {
                Coeff::Uniform(v) => Array3::from_elem(shape, *v),
                Coeff::Field(a) => a.clone(),
            };
            for arm in &arms {
                match (&arm.gain, &arm.decay, &arm.inv_tau) {
                    (Coeff::Uniform(gain), Coeff::Uniform(decay), Coeff::Uniform(inv_tau)) => {
                        let (gain, decay, inv_tau) = (*gain, *decay, *inv_tau);
                        for mu in m_u.iter_mut() {
                            *mu += -gain * inv_tau / (1.0 - decay);
                        }
                    }
                    (Coeff::Field(gain), Coeff::Field(decay), Coeff::Field(inv_tau)) => {
                        leto_ops::zip_mut_with(
                            m_u.view_mut(),
                            (&gain.view(), &decay.view(), &inv_tau.view()),
                            |mu, (&gain, &decay, &inv_tau)| {
                                *mu += -gain * inv_tau / (1.0 - decay);
                            },
                        )
                        .expect("invariant: viscoacoustic modulus fields share grid shape");
                    }
                    _ => unreachable!("invariant: an arm's coefficients share one representation"),
                }
            }
            Coeff::Field(m_u)
        };
        // The CFL reference is the grid max of √(M_U/ρ); a fully uniform pair
        // collapses to the same single expression.
        let max_unrelaxed_speed = match (&m_u, &inv_rho) {
            (Coeff::Uniform(mu), Coeff::Uniform(ir)) => (mu * ir).sqrt(),
            (Coeff::Field(mu), Coeff::Field(ir)) => {
                leto_ops::zip_fold(&mu.view(), &ir.view(), 0.0_f64, |acc, &mu, &ir| {
                    acc.max((mu * ir).sqrt())
                })
                .expect("invariant: modulus and inv_rho fields share grid shape")
            }
            (Coeff::Uniform(mu), Coeff::Field(ir)) => ir
                .view()
                .iter()
                .fold(0.0_f64, |acc, &ir| acc.max((mu * ir).sqrt())),
            (Coeff::Field(mu), Coeff::Uniform(ir)) => mu
                .view()
                .iter()
                .fold(0.0_f64, |acc, &mu| acc.max((mu * ir).sqrt())),
        };

        // Storage layout: velocity/wavenumber state only for active axes.
        // `gx` (divergence output) and `gy` (second staging slot, then the
        // relaxation accumulator) stay grid-shaped; `gz` is the third staging
        // slot and exists only in the all-active layout — the only step that
        // needs three simultaneous slots. An inactive axis owns no storage:
        // its velocity stays exactly zero for all time and its derivative is
        // the exact positive zero fill.
        let stage = |active: bool| {
            if active {
                Array3::zeros(shape)
            } else {
                Array3::zeros((0, 0, 0))
            }
        };
        let all_active = axes.x && axes.y && axes.z;

        Self {
            nx,
            ny,
            nz,
            cell_volume: dx * dy * dz,
            dt,
            axes,
            inv_rho,
            m_u,
            m_inf,
            max_unrelaxed_speed,
            sigma: vec![Array3::zeros(shape); arms.len()],
            arms,
            fft: get_fft_for_grid(nx, ny, nz),
            kx: if axes.x {
                fft_wavenumbers(nx, dx)
            } else {
                Vec::new()
            },
            ky: if axes.y {
                fft_wavenumbers(ny, dy)
            } else {
                Vec::new()
            },
            kz: if axes.z {
                fft_wavenumbers(nz, dz)
            } else {
                Vec::new()
            },
            cbuf: Array3::from_elem([nx, ny, nz], Complex64::new(0.0, 0.0)),
            p: Array3::zeros(shape),
            vx: stage(axes.x),
            vy: stage(axes.y),
            vz: stage(axes.z),
            gx: Array3::zeros(shape),
            gy: Array3::zeros(shape),
            gz: stage(all_active),
            damping_decay: None,
            step_count: 0,
            pressure_sources: Vec::new(),
            pressure_sensors: Vec::new(),
            sensor_record: Vec::new(),
        }
    }
}
