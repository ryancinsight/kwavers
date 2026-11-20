use crate::forward::viscoacoustic::solver::is_positive_finite;
use crate::forward::viscoacoustic::solver::ViscoacousticGrid;
use crate::forward::viscoacoustic::solver::ViscoacousticMemorySolver;
use kwavers_core::error::KwaversError;
use kwavers_core::error::KwaversResult;
use kwavers_medium::absorption::fit_power_law_fields;
use kwavers_medium::absorption::FitBand;
use kwavers_medium::viscoelastic::GeneralizedMaxwellModel;
use leto::Array3;

impl ViscoacousticMemorySolver {
    /// Build from a [`GeneralizedMaxwellModel`] (its `M_∞`, arms, and density)
    /// plus the grid and time step.
    /// # Errors
    /// - Propagates [`Self::new`] validation failures.
    pub fn from_generalized_maxwell(
        model: &GeneralizedMaxwellModel,
        grid: ViscoacousticGrid,
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

        Self::new(
            ViscoacousticGrid::new(nx, ny, nz, dx, dy, dz, dt),
            model.density(),
            model.equilibrium_modulus(),
            model.arms(),
        )
    }

    /// Build from a **heterogeneous power-law medium**: per-voxel density
    /// `ρ(x)`, phase velocity `c(x)` at `f_ref`, absorption coefficient `α(x)`
    /// \[Np·m⁻¹] at `f_ref`, and **per-voxel power-law exponent `γ(x)`**, so
    /// `α(x, f) = α(x)·(f/f_ref)^{γ(x)}`.
    ///
    /// A shared log-spaced relaxation-time grid is fitted per voxel by
    /// [`fit_power_law_fields`]: one non-negative strength vector per voxel on
    /// one `τₗ` grid, so a spatially varying exponent costs no extra memory
    /// fields. The fit also calibrates each voxel's equilibrium modulus so the
    /// **dispersive** phase velocity at `f_ref` equals `c(x)` — the naive
    /// `M_∞ = ρc²` runs fast by the Kramers–Krönig dispersion increment.
    ///
    /// This is the bridge from the `HuAcousticModel`/`CtMediumBuilder` tissue
    /// pipeline (book §4.5) to the broadband solver.
    ///
    /// # Errors
    /// - Field shape ≠ grid, non-positive `ρ`/`c`, negative `α`, a band the fit
    ///   rejects, or `n_arms == 0`.
    pub fn from_power_law_fields(
        grid: ViscoacousticGrid,
        rho: &Array3<f64>,
        c: &Array3<f64>,
        alpha_np_m: &Array3<f64>,
        exponent: &Array3<f64>,
        f_min: f64,
        f_max: f64,
        n_arms: usize,
        f_ref: f64,
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
        // Finite-and-positive: a bare `v > 0.0` admits `NaN`/`+∞` (both
        // compare false against zero), and a non-finite phase velocity or
        // density would poison the relaxation fit and the retained moduli.
        let positive =
            |a: &Array3<f64>| a.shape() == shape && a.iter().all(|&v| is_positive_finite(v));
        if !positive(rho) || !positive(c) || alpha_np_m.shape() != shape {
            return Err(KwaversError::InvalidInput(
                "ρ, c, α fields must be grid-shaped (ρ, c positive)".to_owned(),
            ));
        }
        if exponent.shape() != shape {
            return Err(KwaversError::InvalidInput(
                "γ field must be grid-shaped".to_owned(),
            ));
        }

        let mut band = FitBand::new(f_min, f_max, n_arms)?;
        band.n_samples = (8 * n_arms).max(64);
        let fit = fit_power_law_fields(alpha_np_m, exponent, c, rho, f_ref, &band)?;
        let arms = fit.arm_fields();

        Self::new_heterogeneous(
            ViscoacousticGrid::new(nx, ny, nz, dx, dy, dz, dt),
            rho,
            fit.equilibrium_modulus(),
            &arms,
        )
    }
}
