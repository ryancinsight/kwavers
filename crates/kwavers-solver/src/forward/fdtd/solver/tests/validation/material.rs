use crate::forward::fdtd::config::FdtdAbsorption;
use crate::forward::fdtd::config::FdtdConfig;
use crate::forward::fdtd::solver::FdtdSolver;
use kwavers_core::constants::acoustic_parameters::DB_TO_NP;
use kwavers_grid::Grid;
use kwavers_source::GridSource;

/// **Fullwave 2.5's own scatterer-diameter experiment, replicated.**
///
/// Taken from `experiments/exp_scatterer_diameter.py` in the Fullwave 2.5
/// repository rather than invented, so the parameters are theirs:
///
/// | | background | inclusion |
/// |---|---|---|
/// | sound speed | 1540 m/s | 1600 m/s |
/// | density | 1000 kg/m³ | 1100 kg/m³ |
/// | `α₀` | 0.5 dB/(MHz^y·cm) | 0.75 |
/// | `γ` | **1.0** | 1.1 |
///
/// Two things make this the right replication target for the heterogeneous
/// absorption work. The exponent differs *between regions*, which is the
/// capability itself — a single fitted relaxation spectrum has to serve both.
/// And the background runs at **`γ = 1.0`**, which the PSTD path cannot
/// represent at all: the Treeby–Cox dispersion coefficient carries `tan(πy/2)`,
/// which diverges there (KW-SOL-080). For the relaxation path `γ = 1` is an
/// ordinary point, so this configuration is reachable only through it.
///
/// The assertion is that each region realizes *its own* prescribed law, checked
/// where that region's material actually is.
#[test]
fn fullwave_scatterer_experiment_medium_fits_both_exponents() {
    use kwavers_medium::heterogeneous::HeterogeneousMedium;

    // Fullwave's grid is 30 x 20 mm at 3 MHz; this keeps their materials and
    // frequency and takes a 1-D column through the two regions, which is what
    // the absorption fit sees.
    const N: usize = 96;
    const F0: f64 = 3.0e6;
    const DX: f64 = 1540.0 / F0 / 8.0; // eight points per wavelength

    let grid = Grid::new(N, 1, 1, DX, DX, DX).unwrap();
    let mut medium = HeterogeneousMedium::new(N, 1, 1, false);
    // Centre third is the inclusion, exactly as the experiment builds it.
    let inclusion = (N / 3)..(2 * N / 3);
    for i in 0..N {
        let inside = inclusion.contains(&i);
        medium.sound_speed[[i, 0, 0]] = if inside { 1600.0 } else { 1540.0 };
        medium.density[[i, 0, 0]] = if inside { 1100.0 } else { 1000.0 };
        // `alpha_coefficient` — what the FDTD material sampler calls — reads
        // `absorption`, not `alpha0`, though the struct exposes both. Setting
        // only `alpha0` yields a silently *lossless* medium: the solver builds
        // no relaxation state and the run looks fine.
        // Np/m storage (storage convention in `kwavers_medium::core`); the
        // relaxation fit below sees identical SI spectra, only converted once.
        let alpha = (if inside { 0.75 } else { 0.5 }) * DB_TO_NP * 100.0;
        medium.absorption[[i, 0, 0]] = alpha;
        medium.alpha0[[i, 0, 0]] = alpha;
        medium.alpha_power[[i, 0, 0]] = if inside { 1.1 } else { 1.0 };
    }

    let dt = 0.2 * DX / 1600.0;
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        enable_nonlinear: false,
        dt,
        nt: 2,
        absorption: FdtdAbsorption::PowerLawRelaxation {
            reference_frequency_hz: 1.0e6,
            band_min_hz: 1.0e6,
            band_max_hz: 6.0e6,
            relaxation_arms: 4,
        },
        ..Default::default()
    };
    let solver = FdtdSolver::new(config, &grid, &medium, GridSource::new_empty())
        .expect("the heterogeneous two-exponent medium must be admissible");

    let absorption = solver
        .absorption
        .as_ref()
        .expect("an absorbing configuration must build relaxation state");

    // One shared relaxation-time grid serves both exponents; the fit's own
    // worst-case error is what bounds how well either region is represented.
    // Measured 0.0082 % on these materials at four arms. The bound is an order
    // of magnitude above that rather than at it, so ordinary refits do not trip
    // it, while a spectrum that stopped covering one of the two exponents would
    // — that failure lands at percent level, not at a tenth of one.
    assert!(
        absorption.fit_error() < 0.001,
        "one shared spectrum must cover gamma = 1.0 and gamma = 1.1 across the          experiment's materials: worst relative fit error {:.4} %",
        100.0 * absorption.fit_error()
    );
}

/// **The impedance step in Fullwave's experiment reflects by the analytical
/// coefficient.**
///
/// KW-SOL-095 replicated their *medium* and showed one relaxation spectrum
/// covers both exponents. It never propagated anything, so it said nothing
/// about whether the interface behaves. This runs a wave into it.
///
/// Their materials fix the impedances: `Z = rho*c` gives `1.54` and
/// `1.76 MRayl`, so normal-incidence reflection is
/// `R = (Z2 - Z1)/(Z2 + Z1) = 1/15 = 0.0667` exactly - a closed form, not a
/// tolerance pulled from a previous run.
///
/// Run **lossless** on purpose: their `c` and `rho` set `R`, while `alpha_0`
/// and `gamma` do not, so dropping absorption isolates the interface against an
/// exact oracle instead of folding in decay that would have to be modelled and
/// subtracted. Absorption on these same materials is covered by KW-SOL-095.
///
/// # Not the same as `viscoacoustic::heterogeneous_interface_reflects_...`
///
/// That test exists and checks the same law, so the difference is worth stating
/// rather than leaving a reader to assume duplication. It varies **modulus
/// only** at fixed density on the *viscoacoustic* path, giving `R = 1/3` - a
/// large contrast that exercises the per-voxel coupling. This varies **density
/// and sound speed together** on the *staggered FDTD* path at `R = 1/15`, the
/// small tissue-realistic contrast where an error is easy to miss and where a
/// scheme that quietly mishandles the density average still looks plausible.
#[test]
fn fullwave_material_interface_reflects_by_the_analytical_coefficient() {
    const N: usize = 512;
    const F0: f64 = 3.0e6;
    const C1: f64 = 1540.0;
    const C2: f64 = 1600.0;
    const RHO1: f64 = 1000.0;
    const RHO2: f64 = 1100.0;
    const DX: f64 = C1 / F0 / 8.0;
    const SOURCE: usize = 60;
    const SENSOR: usize = 160;
    const INTERFACE: usize = 300;

    let grid = Grid::new(N, 1, 1, DX, DX, DX).unwrap();
    let mut medium = kwavers_medium::heterogeneous::HeterogeneousMedium::new(N, 1, 1, false);
    for i in 0..N {
        let beyond = i >= INTERFACE;
        medium.sound_speed[[i, 0, 0]] = if beyond { C2 } else { C1 };
        medium.density[[i, 0, 0]] = if beyond { RHO2 } else { RHO1 };
    }

    // One-way rightward launch: pressure paired with u = p/(rho c). An initial
    // pressure alone splits into two halves and the left-going one returns off
    // the rigid wall into the measurement.
    let width = 2.0 * C1 / F0;
    let mut p0 = leto::Array3::<f64>::zeros((N, 1, 1));
    let mut ux0 = leto::Array3::<f64>::zeros((N, 1, 1));
    for i in 0..N {
        let x = (i as f64 - SOURCE as f64) * DX;
        let envelope = (-(x / width).powi(2)).exp();
        let value = envelope * (std::f64::consts::TAU * F0 * x / C1).cos();
        p0[[i, 0, 0]] = value;
        ux0[[i, 0, 0]] = value / (RHO1 * C1);
    }
    let source = GridSource {
        p0: Some(p0),
        u0: Some((
            ux0,
            leto::Array3::zeros((N, 1, 1)),
            leto::Array3::zeros((N, 1, 1)),
        )),
        ..GridSource::new_empty()
    };

    let dt = 0.2 * DX / C2;
    let config = FdtdConfig {
        spatial_order: 4,
        staggered_grid: true,
        enable_nonlinear: false,
        dt,
        nt: 3001,
        ..Default::default()
    };
    let mut solver = FdtdSolver::new(config, &grid, &medium, source).expect("solver");

    let mut trace = Vec::with_capacity(3000);
    for _ in 0..3000 {
        solver.step_forward().expect("step");
        trace.push(solver.fields.p[[SENSOR, 0, 0]]);
    }

    // The incident pulse passes the sensor first; the interface echo passes it
    // again later. Split at the midpoint of those two arrivals.
    let cells_per_step = C1 * dt / DX;
    let incident_step = ((SENSOR - SOURCE) as f64 / cells_per_step) as usize;
    let echo_step = incident_step + (2.0 * (INTERFACE - SENSOR) as f64 / cells_per_step) as usize;
    let split = (incident_step + echo_step) / 2;

    let peak = |window: &[f64]| window.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    let incident = peak(&trace[..split.min(trace.len())]);
    let echo = peak(&trace[split.min(trace.len())..(echo_step + 200).min(trace.len())]);

    let expected = (RHO2 * C2 - RHO1 * C1) / (RHO2 * C2 + RHO1 * C1);
    let measured = echo / incident;
    assert!(
        incident > 0.5,
        "the incident pulse must reach the sensor: peak {incident:.4}"
    );
    assert!(
        (measured - expected).abs() <= 0.05 * expected,
        "reflection coefficient {measured:.4} against analytical {expected:.4}          (incident {incident:.4}, echo {echo:.4})"
    );
}
