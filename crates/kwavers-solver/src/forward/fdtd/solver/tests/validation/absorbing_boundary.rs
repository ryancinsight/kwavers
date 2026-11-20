use crate::forward::fdtd::config::FdtdConfig;
use crate::forward::fdtd::solver::FdtdSolver;
use kwavers_grid::Grid;
use kwavers_medium::HomogeneousMedium;
use kwavers_source::GridSource;

/// Spurious reflection off a PML wall, isolated by differencing against a
/// reference run whose walls are too far away to reflect within the window.
///
/// Source and receiver both sit `h` cells above the lower PML, `offset` cells
/// apart, so the specular bounce meets the wall at `atan((offset/2)/h)` from
/// the normal. The reference repeats the run with the wall `H_REF` cells away;
/// every other wall is at an identical distance and cancels in the difference,
/// leaving the near-wall reflection alone.
///
/// Returns `(peak_direct, peak_spurious, reference_self_consistency)`, the last
/// being the peak change in the reference when its wall moves another 30 cells
/// — the residual reflection contaminating the reference itself.
fn pml_grazing_reflection_sf(
    pml: usize,
    h_test: usize,
    offset: usize,
    sigma_factor: f64,
) -> (f64, f64, f64) {
    pml_grazing_reflection_full(pml, h_test, offset, 1.0, 0.0, sigma_factor)
}

fn pml_grazing_reflection(
    pml: usize,
    h_test: usize,
    offset: usize,
    kappa_max: f64,
    alpha_max: f64,
) -> (f64, f64, f64) {
    pml_grazing_reflection_full(pml, h_test, offset, kappa_max, alpha_max, 2.0)
}

fn pml_grazing_reflection_full(
    pml: usize,
    h_test: usize,
    offset: usize,
    kappa_max: f64,
    alpha_max: f64,
    sigma_factor: f64,
) -> (f64, f64, f64) {
    const NX: usize = 150;
    const H_REF: usize = 90;
    const UPPER: usize = 30;
    const F0: f64 = 1.0e6;
    const C: f64 = 1500.0;
    const RHO: f64 = 1000.0;
    const DX: f64 = C / F0 / 8.0;
    const SX: usize = 20;
    const STEPS: usize = 480;

    let rx = SX + offset;
    let run = |h: usize| -> Vec<f64> {
        let ny = pml + h + UPPER + pml;
        let y0 = pml + h;
        let grid = Grid::new(NX, ny, 1, DX, DX, DX).unwrap();
        let medium = HomogeneousMedium::new(RHO, C, 0.0, 0.0, &grid);

        // Half-width lambda/2 = 4 cells: at the h = 20 standoff the tail
        // reaching the PML is exp(-25) ~ 1e-11, so no source energy is absorbed
        // directly and the difference is reflection rather than lost drive.
        let width = C / F0 / 2.0;
        let mut p0 = leto::Array3::<f64>::zeros((NX, ny, 1));
        for i in 0..NX {
            for j in 0..ny {
                let dx = (i as f64 - SX as f64) * DX;
                let dy = (j as f64 - y0 as f64) * DX;
                p0[[i, j, 0]] = (-((dx * dx + dy * dy) / (width * width))).exp();
            }
        }
        let source = GridSource {
            p0: Some(p0),
            ..GridSource::new_empty()
        };

        let dt = 0.3 * DX / C;
        let config = FdtdConfig {
            spatial_order: 4,
            staggered_grid: true,
            enable_nonlinear: false,
            dt,
            nt: STEPS + 1,
            ..Default::default()
        };
        let mut solver = FdtdSolver::new(config, &grid, &medium, source).expect("solver");
        let cpml = kwavers_boundary::cpml::CPMLConfig {
            per_dimension: kwavers_boundary::cpml::PerDimensionPML::new(pml, pml, 0),
            thickness: pml,
            sigma_factor,
            per_dimension_alpha: kwavers_boundary::cpml::PerDimensionAlpha::uniform(sigma_factor),
            kappa_max,
            alpha_max,
            ..kwavers_boundary::cpml::CPMLConfig::default()
        };
        solver.enable_cpml(cpml, dt, C).expect("enable_cpml");

        let mut trace = Vec::with_capacity(STEPS);
        for _ in 0..STEPS {
            solver.step_forward().expect("step");
            trace.push(solver.fields.p[[rx, y0, 0]]);
        }
        trace
    };

    let peak = |v: &[f64]| v.iter().fold(0.0_f64, |a, b| a.max(b.abs()));
    let peak_diff =
        |a: &[f64], b: &[f64]| peak(&a.iter().zip(b).map(|(x, y)| x - y).collect::<Vec<_>>());

    let reference = run(H_REF);
    let test = run(h_test);
    let farther = run(H_REF - 30);

    (
        peak(&reference),
        peak_diff(&test, &reference),
        peak_diff(&farther, &reference),
    )
}

/// ## Theorem
/// The convolutional CPML absorbs a grazing-incidence wave to below 5e-3 of the
/// incident peak, which holds only while each gradient samples the σ/κ/α
/// profile at its own Yee position.
///
/// ## Method
/// Differential measurement against a reflection-free reference (see
/// [`pml_grazing_reflection`]). At `h = 20`, `offset = 110` the specular bounce
/// meets the wall 70° from the normal.
///
/// ## Measured (2026-08-17)
/// 1.23e-3 of the direct peak, reference reflection-free to 1.9e-14. Before
/// KW-BND-097 — when the pressure-gradient memory used the collocated
/// coefficients although that gradient lives half a cell away — the same
/// configuration reflected 6.31e-2, so the bound here fires ~50× over on a
/// regression of that sampling.
///
/// The CFS terms are deliberately not exercised: with the sampling corrected,
/// κ costs more in discrete reflection than it buys over a 10-cell PML
/// (κ=2 → 1.77e-3, κ=10 → 1.24e-2 against 1.23e-3 for σ-only) and α is
/// neutral (1.21e-3). See `CPMLConfig::kappa_max`.
#[test]
fn cpml_absorbs_grazing_incidence_within_the_reflection_bound() {
    let (direct, spurious, reference_residual) = pml_grazing_reflection(10, 20, 110, 1.0, 0.0);

    // The method is only valid while the reference carries no reflection of its
    // own; without this the difference would measure the reference's error.
    assert!(
        reference_residual <= 1e-8 * direct,
        "reference must be reflection-free: residual {reference_residual:.3e} against direct {direct:.3e}"
    );
    assert!(
        direct > 1e-2,
        "the direct wave must reach the receiver: peak {direct:.3e}"
    );

    let ratio = spurious / direct;
    assert!(
        ratio < 5e-3,
        "grazing reflection {ratio:.4e} exceeds the 5e-3 bound (measured 1.23e-3); \
         a jump toward 6e-2 means the staggered profile sampling regressed"
    );
}

/// ## Theorem
/// On the convolutional FDTD path, `sigma_factor = 3` absorbs grazing incidence
/// substantially better than the k-Wave default of 2.
///
/// ## Measured (2026-08-17, 70° incidence, σ-only)
/// | `sigma_factor` | 10-cell PML | 20-cell PML |
/// |---|---|---|
/// | 2 (default) | 1.23e-3 | 4.26e-7 |
/// | **3** | **4.51e-5** | **6.84e-9** |
/// | 4 | 8.77e-5 | 9.12e-9 |
/// | 5 | 1.40e-4 | 1.14e-8 |
///
/// The optimum sits at 3 for both thicknesses — 27× better than the default at
/// 10 cells, 62× at 20. Only grazing incidence is asserted: the near-normal
/// measurements have reference residuals ~0.6× their signal, so they report the
/// method's noise floor rather than reflection.
///
/// The default stays at 2 because `sigma_factor` is k-Wave's `pml_alpha`, shared
/// with the split-field PSTD path, which this measures nothing about. Changing
/// it is KW-BND-100.
#[test]
fn sigma_factor_three_outperforms_the_kwave_default_at_grazing_incidence() {
    let (direct_default, spurious_default, residual) = pml_grazing_reflection_sf(10, 20, 110, 2.0);
    let (direct_tuned, spurious_tuned, _) = pml_grazing_reflection_sf(10, 20, 110, 3.0);

    assert!(
        residual <= 1e-8 * direct_default,
        "reference must be reflection-free: residual {residual:.3e} against direct {direct_default:.3e}"
    );

    let ratio_default = spurious_default / direct_default;
    let ratio_tuned = spurious_tuned / direct_tuned;
    assert!(
        ratio_tuned < 0.2 * ratio_default,
        "sigma_factor 3 must beat the default 2 by >5x at grazing incidence: \
         default {ratio_default:.4e}, tuned {ratio_tuned:.4e} (measured 27x)"
    );
}
