use super::super::config::NonlinearSWEConfig;
use super::super::material::HyperelasticModel;
use super::super::numerics::NumericsOperators;
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use kwavers_medium::Medium;

/// Nonlinear elastic wave equation solver
#[derive(Debug)]
pub struct NonlinearElasticWaveSolver {
    pub(super) grid: Grid,
    pub(super) _material: HyperelasticModel,
    pub(super) config: NonlinearSWEConfig,
    pub(super) attenuation_np_per_m: f64,
    pub(super) numerics: NumericsOperators,
}

impl NonlinearElasticWaveSolver {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        material: HyperelasticModel,
        config: NonlinearSWEConfig,
    ) -> KwaversResult<Self> {
        // Acoustic attenuation at the medium's reference frequency (Np/m),
        // not the optical Beer–Lambert μa: the two share dimensions (L⁻¹)
        // but describe different physics (KW-SWE-OPTICAL-ATTENUATION). The
        // solver carries no operating frequency, so the reference-frequency
        // value is the honest single-number attenuation; broadband content
        // needs the spectral operator, as in the PSTD path.
        let f_ref = medium.reference_frequency().into_base();
        let attenuation_np_per_m = medium
            .absorption_coefficient(0.0, 0.0, 0.0, grid, f_ref)
            .into_base()
            .max(0.0);

        let numerics = NumericsOperators::new(grid.clone());

        Ok(Self {
            grid: grid.clone(),
            _material: material,
            config,
            attenuation_np_per_m,
            numerics,
        })
    }
}
