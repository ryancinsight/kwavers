//! Therapy Execution and Field Generation
//!
//! This module handles the core therapy execution loop including acoustic field generation,
//! field updates, and step-by-step therapy delivery. It implements focused ultrasound
//! field generation using Gaussian beam approximation and manages the temporal evolution
//! of therapy sessions.
//!
//! ## Acoustic Field Generation
//!
//! Uses Gaussian beam approximation for focused ultrasound field generation:
//! - Focal point targeting based on therapy configuration
//! - Beam width and intensity profiles
//! - Distance-dependent attenuation
//!
//! ## Temperature Output
//!
//! All temperature fields returned by this module are in **degrees Celsius (°C)**,
//! consistent with the `IntensityTracker::update_thermal_dose` contract which
//! applies CEM43 thresholds at 37 °C and 43 °C.
//!
//! ## References
//!
//! - O'Neil (1949): "Gaussian beam propagation in focused ultrasound"
//! - IEC 62359:2010: "Field characterization methods"
//! - Pennes (1948): "Analysis of tissue and arterial blood temperatures"
//! - Nyborg (1981): "Heat generation by ultrasound in a relaxing medium"

use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_TISSUE};
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use kwavers_core::constants::tissue_thermal::SPECIFIC_HEAT_TISSUE;
use kwavers_core::error::KwaversResult;
use kwavers_core::utils::iterators::{for_each_indexed_mut, for_each_indexed_pair_mut};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use leto::Array2;
use leto::Array3;

use super::super::config::AcousticTherapyParams;
use super::super::state::AcousticField;

/// Generate acoustic field for therapy.
///
/// Creates a focused acoustic field using Gaussian beam approximation.
/// The field is centered at the specified focal depth with a Gaussian intensity profile.
///
/// # Arguments
///
/// - `grid`: Computational grid for spatial discretization
/// - `acoustic_params`: Therapy acoustic parameters (frequency, pressure, focal depth, etc.)
///
/// # Returns
///
/// Acoustic field with pressure and velocity components
///
/// # Field Model
///
/// Uses Gaussian beam approximation: P(r) = P₀·exp(−r²/w²), where r is the
/// distance from the focal point and w = 5 mm is the beam width.
///
/// Acoustic radiation force creates particle velocity v = P/(ρ₀·c₀) along the
/// propagation direction. For a plane-wave approximation the x-component is
/// `vx = p/(ρ₀·c₀)` and the transverse components are zero.
///
/// ## Limitation
///
/// The Gaussian approximation is valid for low-intensity diagnostic levels.
/// For HIFU (>1 kW/cm²), it does not capture:
/// - Shock formation and nonlinear harmonic generation (KZK equation,
///   Zabolotskaya & Khokhlov 1969; Lee & Hamilton 1995)
/// - Cavitation inception and Bjerknes force on bubbles
/// - Thermal dose accumulation with nonlinear heating (Sapareto & Dewey 1984)
///
/// Set `acoustic_params.use_nonlinear_field = true` with
/// `TherapyIntegrationModality::HIFU` to use the KZK nonlinear solver
/// ([`generate_kzk_acoustic_field`]) instead of this Gaussian estimator.
///
/// # References
///
/// - O'Neil (1949): "Theory of focusing radiators"
/// - Hasegawa & Yosioka (1975): "Acoustic radiation pressure on compressible spheres"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
/// # See also
///
/// - [`generate_kzk_acoustic_field`] — nonlinear KZK solver for HIFU
///
pub fn generate_acoustic_field(
    grid: &Grid,
    acoustic_params: &AcousticTherapyParams,
) -> KwaversResult<AcousticField> {
    let (nx, ny, nz) = grid.dimensions();
    let mut pressure = Array3::<f64>::zeros((nx, ny, nz));
    let velocity_zero = Array3::<f64>::zeros((nx, ny, nz));

    // Gaussian beam approximation: P(r) = P₀ · exp(−r²/w²)
    // Focal point is along the x-axis at the configured focal depth.
    let focal_x = acoustic_params.focal_depth;
    let beam_width_sq = 0.005_f64 * 0.005; // (5 mm)²

    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;
    let pnp = acoustic_params.pnp;

    for_each_indexed_mut(pressure.view_mut(), |(i, j, k), p| {
        let x = i as f64 * dx - focal_x;
        let y = j as f64 * dy;
        let z = k as f64 * dz;
        let r_sq = x * x + y * y + z * z;
        let beam_profile = (-r_sq / beam_width_sq).exp();
        *p = pnp * beam_profile;
    });

    // Plane-wave approximation for the axial velocity component:
    // v_x = p / (ρ₀·c₀).  Transverse components are zero.
    // Reference impedance uses the nominal soft-tissue value
    // ρ_water · c_tissue ≈ 1.54 MRayl (Hill & ter Haar 2004, §2.3) sourced from SSOT.
    let z_soft_tissue = DENSITY_WATER_NOMINAL * SOUND_SPEED_TISSUE;
    let velocity_x = pressure.mapv(|p| p / z_soft_tissue);

    Ok(AcousticField {
        pressure,
        velocity_x,
        velocity_y: velocity_zero.clone(),
        velocity_z: velocity_zero,
    })
}

/// Generate acoustic field using the KZK nonlinear solver with focusing.
///
/// Creates a pressure field by propagating a focused Gaussian source plane
/// from z = 0 through the full computational grid using the KZK equation
/// (Khokhlov–Zabolotskaya–Kuznetsov) with Strang operator splitting.
/// Captures nonlinear effects (harmonic generation, shock formation, extra
/// absorption) and geometric focusing that the linear Gaussian-beam estimator
/// [`generate_acoustic_field`] omits.
///
/// # Arguments
///
/// - `grid`: Computational grid for spatial discretization.  The KZK solver
///   maps therapy coordinates: axial (x) → propagation direction, transverse
///   (y, z) → the 2D source plane.  Grids where the transverse extent
///   exceeds ~0.3× the axial extent may trigger a paraxial angle warning and
///   fall back to the collimated (plugin-based) KZK solver.
/// - `acoustic_params`: Therapy acoustic parameters (frequency, pressure,
///   focal depth, etc.)
/// - `medium`: Medium properties (density, sound speed, absorption,
///   nonlinearity)
///
/// # Returns
///
/// Acoustic field with pressure and velocity components
///
/// # Field Model
///
/// A Gaussian-apodized source plane at the transducer face (z = 0) is
/// initialised with a **quadratic focusing phase**:
///
/// ```text
/// p(x,y,τ) = A(x,y) · sin(ω₀τ − k·(x²+y²)/(2·F))
/// ```
///
/// where `F = acoustic_params.focal_depth` is the geometric focal distance.
/// The field is propagated stepwise via the complex-valued KZK solver with
/// Strang splitting: D(Δz/2)·A(Δz/2)·N(Δz)·A(Δz/2)·D(Δz/2).
///
/// The output pressure field stores the **RMS pressure** `p_rms(x,y)` at
/// each axial plane, averaged over one acoustic period.  This matches the
/// `p² ∝ heating` dependence of thermal therapy models.
///
/// ## Fallback
///
/// When the grid violates the KZK paraxial angle constraint (transverse
/// extent too large relative to axial), the solver logs a warning and falls
/// back to the real-valued `KzkSolverPlugin`, which propagates a collimated
/// (unfocused) beam.  This preserves backward compatibility for square
/// test grids while offering focusing on physically appropriate grids.
///
/// # References
///
/// - Zabolotskaya & Khokhlov (1969): "Quasi-plane waves in the nonlinear
///   acoustics of confined beams"
/// - Tavakkoli et al. (1998): "A new algorithm for computational simulation
///   of focused ultrasound in inhomogeneous tissue."
/// - Lee & Hamilton (1995): "Time-domain modeling of pulsed finite-amplitude
///   sound beams"
/// # Errors
/// - Returns [`Err`] if an internal constraint is violated.
///
pub fn generate_kzk_acoustic_field(
    grid: &Grid,
    acoustic_params: &AcousticTherapyParams,
    medium: &dyn Medium,
) -> KwaversResult<AcousticField> {
    use kwavers_physics::acoustics::wave_propagation::nonlinear::kzk::KZKSolverTrait;
    use kwavers_solver::forward::nonlinear::kzk::{KZKConfig, KZKSolver};

    let (nx, ny, nz) = grid.dimensions();
    // nx = axial (propagation direction)
    // ny, nz = transverse

    // Source beam: 5 mm 1/e² radius at the transducer face.
    let beam_width_sq = 0.005_f64 * 0.005;

    // Medium properties at the source centre (homogeneous approximation).
    let c0 = kwavers_medium::sound_speed_at(medium, 0.0, 0.0, 0.0, grid);
    let rho0 = kwavers_medium::density_at(medium, 0.0, 0.0, 0.0, grid);
    let b_over_a = medium.nonlinearity_parameter(0.0, 0.0, 0.0, grid);
    let alpha0 = medium.alpha_coefficient(0.0, 0.0, 0.0, grid);
    let alpha_power = medium.alpha_power(0.0, 0.0, 0.0, grid);

    // KZK time discretisation: 20 samples per fundamental period, CFL-safe.
    let frequency = acoustic_params.frequency;
    // dt limited by both CFL (c0·dt/dz < 0.5) and harmonic Nyquist (dt < 1/(2·10·f₀)).
    let dt_cfl = 0.3 * grid.dx / c0;
    let dt_nyquist = 1.0 / (2.0 * 10.0 * frequency);
    let dt = dt_cfl.min(dt_nyquist);
    // nt: at least 10 periods of the fundamental, rounded up to a power-of-two.
    let nt = ((10.0 / (frequency * dt)).ceil() as usize)
        .max(128)
        .next_power_of_two();

    // Build KZK config, mapping therapy coordinates → KZK coordinates.
    //   Therapy:  axial = x (nx),  transverse = (y, z)  = (ny, nz)
    //   KZK:      axial = z (nz),  transverse = (x, y)  = (nx, ny)
    let config = KZKConfig {
        nx: ny,      // KZK transverse × (therapy y)
        ny: nz,      // KZK transverse y (therapy z)
        nz: nx,      // KZK axial steps (therapy x)
        dx: grid.dy, // transverse spacing
        dz: grid.dx, // axial step
        dt,
        nt,
        c0,
        rho0,
        b_over_a,
        alpha0,
        alpha_power,
        include_diffraction: true,
        include_absorption: true,
        include_nonlinearity: true,
        frequency,
    };

    // Attempt to create the complex KZK solver.  If validation fails (e.g.,
    // the paraxial angle limit is exceeded for this grid), fall back to the
    // collimated plugin solver (backward-compatible).
    let mut solver = match KZKSolver::new(config) {
        Ok(s) => s,
        Err(msg) => {
            tracing::warn!(
                "KZK solver validation failed: {msg}.  \
                 Falling back to collimated KZK solver plugin."
            );
            return generate_kzk_collimated(grid, acoustic_params, medium);
        }
    };

    // Build the focused source plane (transverse amplitude map).
    let dx_kzk = grid.dy;
    let mut source = Array2::<f64>::zeros((ny, nz));
    let pnp = acoustic_params.pnp;
    for j in 0..nz {
        for i in 0..ny {
            let x = (i as f64 - ny as f64 / 2.0) * dx_kzk;
            let y = (j as f64 - nz as f64 / 2.0) * dx_kzk;
            let r2 = x * x + y * y;
            source[[i, j]] = pnp * (-r2 / beam_width_sq).exp();
        }
    }
    solver.set_focused_source(source, frequency, acoustic_params.focal_depth);

    // Propagate step-by-step, capturing RMS pressure at each axial plane.
    let mut volume = Array3::<f64>::zeros((nx, ny, nz));
    for iz in 0..nx {
        if iz > 0 {
            solver.step();
        }
        let rms = solver.current_field(); // shape (ny, nz) in KZK = (ny, nz) in therapy
        for j in 0..nz {
            for i in 0..ny {
                volume[[iz, i, j]] = rms[[i, j]];
            }
        }
    }

    // Plane-wave approximation: v_x = p / (ρ₀·c₀).
    let z_soft_tissue = DENSITY_WATER_NOMINAL * SOUND_SPEED_TISSUE;
    let velocity_x = volume.mapv(|p| p / z_soft_tissue);
    let velocity_zero = Array3::<f64>::zeros((nx, ny, nz));

    Ok(AcousticField {
        pressure: volume,
        velocity_x,
        velocity_y: velocity_zero.clone(),
        velocity_z: velocity_zero,
    })
}

/// Fallback KZK path using the real-valued plugin solver (collimated beam).
///
/// Used when the grid does not satisfy the complex KZK solver's paraxial
/// angle constraint.  The beam propagates without geometric focusing but
/// still captures nonlinear effects (harmonic generation, shock formation).
fn generate_kzk_collimated(
    grid: &Grid,
    acoustic_params: &AcousticTherapyParams,
    medium: &dyn Medium,
) -> KwaversResult<AcousticField> {
    use kwavers_solver::forward::nonlinear::kzk_solver_plugin::KzkSolverPlugin;

    let (nx, ny, nz) = grid.dimensions();
    let beam_width_sq = 0.005_f64 * 0.005;
    let max_frequency = acoustic_params.frequency * 10.0;
    const NUM_HARMONICS: usize = 10;

    let mut kzk = KzkSolverPlugin::new();
    kzk.initialize_operators(grid, medium, max_frequency)?;

    let mut source = Array3::<f64>::zeros((nx, ny, NUM_HARMONICS));
    for i in 0..nx {
        for j in 0..ny {
            let x = i as f64 * grid.dx;
            let y = j as f64 * grid.dy;
            let r_sq = x * x + y * y;
            source[[i, j, 0]] = acoustic_params.pnp * (-r_sq / beam_width_sq).exp();
        }
    }

    let pressure_volume = kzk.propagate_volume(&source, grid, medium)?;

    let z_soft_tissue = DENSITY_WATER_NOMINAL * SOUND_SPEED_TISSUE;
    let velocity_x = pressure_volume.mapv(|p| p / z_soft_tissue);
    let velocity_zero = Array3::<f64>::zeros((nx, ny, nz));

    Ok(AcousticField {
        pressure: pressure_volume,
        velocity_x,
        velocity_y: velocity_zero.clone(),
        velocity_z: velocity_zero,
    })
}

/// Calculate acoustic absorption heating.
///
/// Returns a 3-D array of **Celsius (°C)** temperatures representing the
/// instantaneous temperature distribution after one therapy step of duration `dt`.
///
/// # Physics Model
///
/// Acoustic absorption heating: Q = α · p² / (ρ₀ · c₀)
/// where α = 0.5 Np/m (soft tissue, Nyborg 1981), ρ₀ = 1000 kg/m³, c₀ = 1540 m/s.
///
/// Temperature rise: ΔT = Q · exp(−r/L) · dt / (ρ₀ · c_p) with characteristic
/// focal length L = 10 mm and specific heat capacity c_p = 3600 J/(kg·K).
///
/// The ambient temperature T₀ = 37 °C is the baseline for CEM43 evaluation.
///
/// # Returns
///
/// 3-D array of temperatures in **degrees Celsius**, with
/// `T_min = 37.0 °C` (no heating outside focal zone) and
/// `T_max = 37.0 + ΔT_peak`.
///
/// # References
///
/// - Pennes (1948): "Analysis of tissue and arterial blood temperatures"
/// - Nyborg (1981): "Heat generation by ultrasound in a relaxing medium"
/// - Sapareto & Dewey (1984): "Thermal dose determination in cancer therapy"
pub fn calculate_acoustic_heating(
    acoustic_field: &AcousticField,
    grid: &Grid,
    dt: f64,
    focal_depth: f64,
) -> Array3<f64> {
    // Tissue constants for soft tissue (Nyborg 1981).
    const ALPHA_NP_M: f64 = 0.5; // absorption coefficient (Np/m)
    const RHO: f64 = DENSITY_WATER_NOMINAL;
    const C0: f64 = SOUND_SPEED_TISSUE;
    let c_p = SPECIFIC_HEAT_TISSUE; // J/(kg·K) — Duck (1990) / ICRP 2002 soft tissue
    const L_FOCAL: f64 = 0.01; // focal characteristic length (10 mm)

    // Q = α p² / (ρ₀ c₀); ΔT = Q dt / (ρ₀ c_p)
    // Combined: ΔT = α p² dt / (ρ₀² c₀ c_p)
    let heating_scale = ALPHA_NP_M * dt / (RHO * RHO * C0 * c_p);

    let dx = grid.dx;
    let dy = grid.dy;
    let dz = grid.dz;

    let mut temperature =
        Array3::<f64>::from_elem(acoustic_field.pressure.shape(), BODY_TEMPERATURE_C);

    for_each_indexed_pair_mut(
        temperature.view_mut(),
        acoustic_field.pressure.view(),
        |(i, j, k), t, &p| {
            // Radial distance from focal point (on the x-axis).
            let x = i as f64 * dx - focal_depth;
            let y = j as f64 * dy;
            let z = k as f64 * dz;
            let r = (x * x + y * y + z * z).sqrt();
            let distance_factor = (-r / L_FOCAL).exp();
            *t = BODY_TEMPERATURE_C + heating_scale * p * p * distance_factor;
        },
    );

    temperature
}
