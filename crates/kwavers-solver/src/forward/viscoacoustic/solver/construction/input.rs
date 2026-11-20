use crate::forward::viscoacoustic::solver::build_arm;
use crate::forward::viscoacoustic::solver::build_uniform_arm;
use crate::forward::viscoacoustic::solver::is_positive_finite;
use crate::forward::viscoacoustic::solver::ActiveAxes;
use crate::forward::viscoacoustic::solver::Arm;
use crate::forward::viscoacoustic::solver::Coeff;
use crate::forward::viscoacoustic::solver::ViscoacousticGrid;
use crate::forward::viscoacoustic::solver::ViscoacousticMemorySolver;
use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl ViscoacousticMemorySolver {
    /// Build from raw parameters: grid `(nx,ny,nz)` with spacings `(dx,dy,dz)`,
    /// time step `dt`, density `ρ`, equilibrium modulus `M_∞`, and relaxation
    /// arms `(ΔMₗ, τₗ)`. An empty arm list yields the lossless wave equation.
    /// # Errors
    /// - Any zero dimension, non-finite or non-positive
    ///   `dx`/`dy`/`dz`/`dt`/`ρ`/`M_∞`, or a non-finite or non-positive arm
    ///   parameter (`NaN` and infinity are rejected before any allocation).
    pub fn new(
        grid: ViscoacousticGrid,
        rho: f64,
        m_inf: f64,
        arms: &[(f64, f64)],
    ) -> KwaversResult<Self> {
        let ViscoacousticGrid {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
        } = grid;

        if nx == 0
            || ny == 0
            || nz == 0
            || !is_positive_finite(dx)
            || !is_positive_finite(dy)
            || !is_positive_finite(dz)
            || !is_positive_finite(dt)
            || !is_positive_finite(rho)
            || !is_positive_finite(m_inf)
        {
            return Err(KwaversError::InvalidInput(
                "viscoacoustic solver requires positive finite grid, spacings, dt, ρ, M_∞"
                    .to_owned(),
            ));
        }
        if arms
            .iter()
            .any(|&(dm, tau)| !is_positive_finite(dm) || !is_positive_finite(tau))
        {
            return Err(KwaversError::InvalidInput(
                "relaxation arms require positive finite ΔM and τ".to_owned(),
            ));
        }

        // Homogeneous medium: every coefficient is a broadcast scalar by
        // construction (the parameters cannot vary across voxels), so the
        // solver retains one f64 per coefficient instead of a full grid —
        // at every grid shape. The scalar coefficients carry the same values
        // the broadcast fields would, and the step evaluates the same float
        // operations in the same order, so the uniform layout is bitwise
        // comparable with the grid layout.
        let arm_fields: Vec<Arm> = arms
            .iter()
            .map(|&(dm, tau)| build_uniform_arm(dm, tau, dt))
            .collect();
        Ok(Self::assemble(
            ViscoacousticGrid::new(nx, ny, nz, dx, dy, dz, dt),
            Coeff::Uniform(1.0 / rho),
            Coeff::Uniform(m_inf),
            arm_fields,
            ActiveAxes::of(nx, ny, nz),
        ))
    }

    /// Build a **heterogeneous** medium from per-voxel fields: density `ρ(x)`,
    /// equilibrium modulus `M_∞(x)`, and relaxation arms `(ΔMₗ(x), τₗ(x))` (all of
    /// grid shape). This lets a CT-derived tissue model (§4.5) drive the broadband
    /// solver with spatially-varying viscoacoustic properties.
    ///
    /// A zero-strength arm is inert (an exactly lossless voxel), so `ΔM` is
    /// only required non-negative; a heterogeneous medium may contain lossless
    /// regions alongside absorbing ones.
    ///
    /// # Errors
    /// - Any field shape ≠ `(nx,ny,nz)`, a non-finite or non-positive
    ///   `ρ`/`M_∞`/`τ` element, a negative or non-finite `ΔM` element, or a
    ///   non-finite or non-positive grid/spacing/`dt` (rejection happens
    ///   before any allocation).
    pub fn new_heterogeneous(
        grid: ViscoacousticGrid,
        rho: &Array3<f64>,
        m_inf: &Array3<f64>,
        arms: &[(Array3<f64>, Array3<f64>)],
    ) -> KwaversResult<Self> {
        let ViscoacousticGrid {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            dt,
        } = grid;

        let shape = [nx, ny, nz];
        let ok_shape = |a: &Array3<f64>| a.shape() == shape;
        if nx == 0
            || ny == 0
            || nz == 0
            || !is_positive_finite(dx)
            || !is_positive_finite(dy)
            || !is_positive_finite(dz)
            || !is_positive_finite(dt)
        {
            return Err(KwaversError::InvalidInput(
                "viscoacoustic solver requires positive finite grid, spacings, dt".to_owned(),
            ));
        }
        if !ok_shape(rho)
            || !ok_shape(m_inf)
            || rho.iter().any(|&r| !is_positive_finite(r))
            || m_inf.iter().any(|&m| !is_positive_finite(m))
        {
            return Err(KwaversError::InvalidInput(
                "ρ and M_∞ fields must be grid-shaped and positive finite".to_owned(),
            ));
        }
        if arms.iter().any(|(dm, tau)| {
            !ok_shape(dm)
                || !ok_shape(tau)
                || dm.iter().any(|&v| !(v.is_finite() && v >= 0.0))
                || tau.iter().any(|&v| !is_positive_finite(v))
        }) {
            return Err(KwaversError::InvalidInput(
                "relaxation arm fields must be grid-shaped with finite ΔM≥0 and τ>0".to_owned(),
            ));
        }

        // Heterogeneous media keep per-voxel coefficient fields unconditionally;
        // even a constant-field call site measures and behaves identically.
        let inv_rho = Coeff::Field(rho.mapv(|r| 1.0 / r));
        let arm_fields: Vec<Arm> = arms
            .iter()
            .map(|(dm, tau)| build_arm(dm, tau, dt))
            .collect();
        Ok(Self::assemble(
            ViscoacousticGrid::new(nx, ny, nz, dx, dy, dz, dt),
            inv_rho,
            Coeff::Field(m_inf.clone()),
            arm_fields,
            ActiveAxes::of(nx, ny, nz),
        ))
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
        Self::new(
            ViscoacousticGrid::new(n, 1, 1, dx, 1.0, 1.0, dt),
            rho,
            m_inf,
            arms,
        )
    }
}
