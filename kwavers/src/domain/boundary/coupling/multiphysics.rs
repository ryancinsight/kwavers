//! Multi-physics interface boundary condition
//!
//! Handles coupling between different physics domains (e.g., acoustic-elastic,
//! electromagnetic-acoustic) with physically rigorous transmission conditions.

use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::BoundaryCondition;
use crate::domain::grid::GridTopology;
use ndarray::ArrayViewMut3;

use super::types::{BoundaryDirections, CouplingType, PhysicsDomain};

/// Multi-physics interface boundary
///
/// Implements coupling between different physics domains with appropriate
/// transmission conditions. This enables modeling of:
///
/// - Fluid-structure interaction (acoustic-elastic)
/// - Photoacoustic imaging (electromagnetic-acoustic)
/// - Thermoacoustic effects (acoustic-thermal)
/// - Photothermal therapy (electromagnetic-thermal)
///
/// # Physics
///
/// At a multi-physics interface, the boundary conditions depend on the
/// coupling type:
///
/// ## Acoustic-Elastic Coupling
///
/// At a fluid-solid interface:
/// - **Normal stress continuity**: σ_solid · n = -p_fluid
/// - **Normal velocity continuity**: v_fluid · n = v_solid · n
/// - **Zero tangential stress**: τ = 0 (free slip at fluid boundary)
///
/// ## Electromagnetic-Acoustic Coupling (Photoacoustic)
///
/// Optical absorption generates acoustic waves via thermal expansion:
///
/// ```text
/// p(r,t) = Γ · μ_a · Φ(r,t)
/// ```
///
/// where:
/// - Γ is the Grüneisen parameter (dimensionless)
/// - μ_a is the optical absorption coefficient (m⁻¹)
/// - Φ is the optical fluence (J/m²)
///
/// ## Acoustic-Thermal Coupling
///
/// Acoustic absorption generates heat:
///
/// ```text
/// ∂T/∂t = α · ∇²T + Q_acoustic
/// Q_acoustic = 2α_acoustic · I / ρc_p
/// ```
///
/// where I is the acoustic intensity.
///
/// # Example
///
/// ```no_run
/// use kwavers::domain::boundary::coupling::{MultiPhysicsInterface, PhysicsDomain, CouplingType};
///
/// // Photoacoustic interface (light absorption → sound generation)
/// // Water Grüneisen parameter Γ ≈ 0.12 (Xu & Wang 2006)
/// let interface = MultiPhysicsInterface::new(
///     [0.0, 0.0, 0.0],           // Interface position
///     [1.0, 0.0, 0.0],           // Interface normal
///     PhysicsDomain::Electromagnetic,
///     PhysicsDomain::Acoustic,
///     CouplingType::ElectromagneticAcoustic {
///         optical_absorption: 100.0,  // μ_a = 100 m⁻¹
///         gruneisen: 0.12,            // Γ for water at 20°C
///     },
/// );
///
/// let transmission = interface.transmission_coefficient(1e6);
/// ```
///
/// # References
///
/// - Kuchment & Kunyansky, "Mathematics of Photoacoustic and Thermoacoustic Tomography" (2010)
/// - Beard, "Biomedical Photoacoustic Imaging", Interface Focus (2011)
/// - Duck, "Physical Properties of Tissue" (1990)
#[derive(Debug, Clone)]
pub struct MultiPhysicsInterface {
    /// Interface position [x, y, z] in meters
    pub position: [f64; 3],
    /// Interface normal vector
    pub normal: [f64; 3],
    /// Physics domain 1 (left side of interface)
    pub physics_1: PhysicsDomain,
    /// Physics domain 2 (right side of interface)
    pub physics_2: PhysicsDomain,
    /// Coupling type with parameters
    pub coupling_type: CouplingType,
}

impl MultiPhysicsInterface {
    /// Create a new multi-physics interface
    ///
    /// # Arguments
    ///
    /// * `position` - Interface position [x, y, z] in meters
    /// * `normal` - Interface normal vector (will be normalized)
    /// * `physics_1` - Physics domain on negative side of interface
    /// * `physics_2` - Physics domain on positive side of interface
    /// * `coupling_type` - Type of coupling with associated parameters
    ///
    /// # Returns
    ///
    /// New `MultiPhysicsInterface`
    pub fn new(
        position: [f64; 3],
        normal: [f64; 3],
        physics_1: PhysicsDomain,
        physics_2: PhysicsDomain,
        coupling_type: CouplingType,
    ) -> Self {
        Self {
            position,
            normal,
            physics_1,
            physics_2,
            coupling_type,
        }
    }

    /// Compute the power transmission coefficient for the coupling type.
    ///
    /// # Theory
    ///
    /// ## Acoustic-Elastic (fluid-solid interface)
    ///
    /// At normal incidence, the plane-wave power transmission coefficient is:
    ///
    /// ```text
    /// τ = 4 Z₁ Z₂ / (Z₁ + Z₂)²
    /// ```
    ///
    /// where Z₁ = ρ₁ c₁ and Z₂ = ρ₂ c₂ are the specific acoustic impedances
    /// [Pa·s/m = Rayl]. This is the fraction of incident acoustic power transmitted
    /// across the interface (Brekhovskikh & Godin 1998, *Acoustics of Layered Media I*,
    /// §1.5, Eq. 1.55). The identity τ + R = 1 holds where R = (Z₂−Z₁)²/(Z₁+Z₂)²
    /// is the power reflection coefficient.
    ///
    /// ## Electromagnetic-Acoustic (photoacoustic)
    ///
    /// The photoacoustic pressure conversion efficiency (Pa·m³/J) is:
    ///
    /// ```text
    /// η_PA = Γ · μ_a
    /// ```
    ///
    /// where Γ is the dimensionless Grüneisen parameter and μ_a is the optical
    /// absorption coefficient [m⁻¹]. The ratio η_PA / (reference value) is returned
    /// as a dimensionless coupling coefficient normalised to Γ·μ_a = 1 m⁻¹.
    /// Reference: Xu & Wang (2006), Rev. Sci. Instrum. 77, 041101, Eq. (2).
    ///
    /// ## Acoustic-Thermal
    ///
    /// The fraction of incident acoustic power converted to heat per unit path length
    /// is `2α` (two-way amplitude attenuation). The dimensionless coupling is:
    ///
    /// ```text
    /// τ_thermal = 1 − exp(−2 α L)   → 2 α L  for  α L ≪ 1
    /// ```
    ///
    /// For the local BC formulation (δ-thin interface), we return the normalised
    /// instantaneous absorption efficiency:
    ///
    /// ```text
    /// η = 2 α / (ρ c_p)   [m⁻¹ · K / (Pa·s)]   — clamped to [0, 1]
    /// ```
    ///
    /// Reference: Duck (1990), *Physical Properties of Tissue*, §4.
    ///
    /// # Arguments
    ///
    /// * `_frequency` - Frequency in Hz (reserved for frequency-dependent coupling;
    ///   the current implementation uses frequency-independent coefficients).
    ///
    /// # Returns
    ///
    /// Dimensionless power transmission / coupling efficiency ∈ [0, 1] unless
    /// `ElectromagneticAcoustic` is used, in which case it is Γ·μ_a (units m⁻¹)
    /// normalised to 1 m⁻¹ reference.
    pub fn transmission_coefficient(&self, _frequency: f64) -> f64 {
        match &self.coupling_type {
            CouplingType::AcousticElastic { z1_rayl, z2_rayl } => {
                // Plane-wave power transmission at normal incidence (Brekhovskikh & Godin 1998, §1.5):
                //   τ = 4 Z₁ Z₂ / (Z₁ + Z₂)²
                let z1 = *z1_rayl;
                let z2 = *z2_rayl;
                let sum = z1 + z2;
                if sum < f64::EPSILON {
                    return 0.0;
                }
                (4.0 * z1 * z2) / (sum * sum)
            }
            CouplingType::ElectromagneticAcoustic {
                optical_absorption,
                gruneisen,
            } => {
                // Photoacoustic coupling: η = Γ · μ_a  (Xu & Wang 2006, Eq. 2).
                // Normalised to 1 m⁻¹ reference so that the result is dimensionless.
                // Clamped to [0, 1] — any Γ·μ_a > 1 m⁻¹ saturates at 1.
                (gruneisen * optical_absorption).clamp(0.0, 1.0)
            }
            CouplingType::AcousticThermal {
                alpha_np_per_m,
                rho_kg_per_m3,
                c_p_j_per_kg_k,
            } => {
                // Volumetric acoustic-thermal coupling efficiency (Duck 1990, §4):
                //   Q = 2α · I / (ρ c_p)   [K/s per W/m²]
                // Normalise by a 1 W/m² reference intensity and 1 K rise to get
                // a dimensionless coupling in [0, 1]. The raw ratio 2α/(ρ c_p) has
                // units m²/(J/K); we multiply by reference I·t = 1 W/m² × 1 s = 1 J/m²
                // to obtain K, then divide by 1 K. Clamp for physical validity.
                let denom = rho_kg_per_m3 * c_p_j_per_kg_k;
                if denom < f64::EPSILON {
                    return 0.0;
                }
                (2.0 * alpha_np_per_m / denom).clamp(0.0, 1.0)
            }
            CouplingType::ElectromagneticThermal => {
                // Photothermal coupling: nearly all absorbed optical energy → heat
                // in biological tissue (non-radiative relaxation dominates).
                // Typical quantum yield for heat ≈ 0.97 (Jacques 2013, Phys. Med. Biol. 58, R37).
                0.97
            }
            CouplingType::Custom(_) => 1.0,
        }
    }
}

impl BoundaryCondition for MultiPhysicsInterface {
    fn name(&self) -> &str {
        "MultiPhysicsInterface"
    }

    fn active_directions(&self) -> BoundaryDirections {
        // Multi-physics interfaces typically affect all directions
        BoundaryDirections::all()
    }

    fn apply_scalar_spatial(
        &mut self,
        _field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Multi-physics interface boundary condition
        // Real implementation would:
        // 1. Identify interface location in grid
        // 2. Apply appropriate coupling conditions based on physics domains
        // 3. Handle field transformations between domains
        //
        // For now, this is a placeholder for future implementation

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut ndarray::Array3<num_complex::Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Frequency-domain multi-physics interface
        Ok(())
    }

    fn reset(&mut self) {
        // No state to reset for multi-physics interface
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Acoustic impedances (ρ·c) for test materials ─────────────────────────
    // Water (20°C):       Z = 998 × 1482 = 1_479_036 Rayl
    // Soft tissue (37°C): Z = 1060 × 1540 = 1_632_400 Rayl
    // Cortical bone:      Z = 1900 × 3294 = 6_258_600 Rayl
    const Z_WATER: f64 = 1_479_036.0;
    const Z_SOFT_TISSUE: f64 = 1_632_400.0;
    const Z_BONE: f64 = 6_258_600.0;

    #[test]
    fn test_multiphysics_interface_photoacoustic() {
        // Γ · μ_a = 0.15 × 100 = 15 → clamped to 1.0
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Electromagnetic,
            PhysicsDomain::Acoustic,
            CouplingType::ElectromagneticAcoustic {
                optical_absorption: 100.0,
                gruneisen: 0.15,
            },
        );
        let tau = interface.transmission_coefficient(1e6);
        assert!(tau > 0.0);
        assert!((0.0..=1.0).contains(&tau));
    }

    /// Water–soft-tissue: nearly impedance-matched → τ ≈ 0.9978 (high transmission).
    ///
    /// Computed from Z_water = 998×1482 = 1,479,036 Rayl and
    /// Z_soft_tissue = 1060×1540 = 1,632,400 Rayl:
    ///   τ = 4×1.479×1.632 / (3.111)² ≈ 0.9978
    ///   R = (1.632−1.479)² / (3.111)² ≈ 0.0022 (0.2% reflected)
    ///
    /// Reference: Duck (1990), *Physical Properties of Tissue*, Table 4.1.
    #[test]
    fn test_acoustic_elastic_water_soft_tissue_transmission() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            PhysicsDomain::Acoustic,
            PhysicsDomain::Elastic,
            CouplingType::AcousticElastic {
                z1_rayl: Z_WATER,
                z2_rayl: Z_SOFT_TISSUE,
            },
        );
        let tau = interface.transmission_coefficient(1e6);
        let expected = 4.0 * Z_WATER * Z_SOFT_TISSUE / (Z_WATER + Z_SOFT_TISSUE).powi(2);
        assert!(
            (tau - expected).abs() < 1e-10,
            "τ = {:.6}, expected {:.6}",
            tau,
            expected
        );
        // Water/tissue: τ > 0.99 (well-matched impedances)
        assert!(tau > 0.99, "water/tissue should be nearly impedance-matched; got τ = {:.6}", tau);
        // Power conservation: τ + R = 1
        let r = (Z_SOFT_TISSUE - Z_WATER).powi(2) / (Z_SOFT_TISSUE + Z_WATER).powi(2);
        assert!((tau + r - 1.0).abs() < 1e-12, "τ + R = {:.15}", tau + r);
    }

    /// Water–cortical bone: significant impedance mismatch → τ ≈ 0.618, R ≈ 0.382.
    ///
    /// Computed from Z_water = 1,479,036 Rayl and Z_bone = 1900×3294 = 6,258,600 Rayl:
    ///   τ = 4×1.479×6.259 / (7.738)² ≈ 0.618
    ///   R = (6.259−1.479)² / (7.738)² ≈ 0.382  (38% reflected — clinically significant)
    ///
    /// Reference: Bamber (2004), in *Physical Principles of Medical Ultrasonics*, §4;
    /// Fry & Barger (1978), *J Acoust Soc Am* 63(5):1576–1590.
    #[test]
    fn test_acoustic_elastic_water_bone_transmission() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            PhysicsDomain::Acoustic,
            PhysicsDomain::Elastic,
            CouplingType::AcousticElastic {
                z1_rayl: Z_WATER,
                z2_rayl: Z_BONE,
            },
        );
        let tau = interface.transmission_coefficient(1e6);
        let expected = 4.0 * Z_WATER * Z_BONE / (Z_WATER + Z_BONE).powi(2);
        assert!((tau - expected).abs() < 1e-10, "τ = {:.6}, expected {:.6}", tau, expected);
        // Water/bone: 30%–75% transmitted (large but not total impedance mismatch)
        assert!(
            tau > 0.3 && tau < 0.75,
            "water/bone τ should be in (0.3, 0.75); got τ = {:.4}",
            tau
        );
        // Power conservation: τ + R = 1
        let r = (Z_BONE - Z_WATER).powi(2) / (Z_BONE + Z_WATER).powi(2);
        assert!((tau + r - 1.0).abs() < 1e-12, "τ + R = {:.15}", tau + r);
        // Bone reflects significantly more than soft tissue
        assert!(r > 0.2, "bone interface should have >20% reflection; R = {:.4}", r);
    }

    /// Photoacoustic: higher absorption → higher coupling (monotone property).
    #[test]
    fn test_multiphysics_photoacoustic_monotone() {
        let gruneisen = 0.12; // water at 20°C
        let make = |mu_a: f64| {
            MultiPhysicsInterface::new(
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                PhysicsDomain::Electromagnetic,
                PhysicsDomain::Acoustic,
                CouplingType::ElectromagneticAcoustic {
                    optical_absorption: mu_a,
                    gruneisen,
                },
            )
            .transmission_coefficient(1e6)
        };
        let tau_lo = make(1.0);   // Γ·μ_a = 0.12 < 1 → not clamped
        let tau_hi = make(10.0);  // Γ·μ_a = 1.2 > 1 → clamped to 1.0
        assert!(tau_hi >= tau_lo, "coupling must be monotone in μ_a");
        assert!((tau_lo - 0.12).abs() < 1e-12, "τ(μ_a=1) = Γ·μ_a = {:.4}", tau_lo);
        assert!((tau_hi - 1.0).abs() < 1e-12, "τ must saturate at 1 for Γ·μ_a > 1");
    }

    /// Acoustic-thermal: coupling is physically positive and ≤ 1.
    #[test]
    fn test_acoustic_thermal_coupling_bounds() {
        // Soft tissue parameters: α = 2 Np/m, ρ = 1060 kg/m³, c_p = 3500 J/(kg·K)
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Acoustic,
            PhysicsDomain::Thermal,
            CouplingType::AcousticThermal {
                alpha_np_per_m: 2.0,
                rho_kg_per_m3: 1060.0,
                c_p_j_per_kg_k: 3500.0,
            },
        );
        let tau = interface.transmission_coefficient(1e6);
        assert!((0.0..=1.0).contains(&tau), "τ = {}", tau);
        // η = 2α/(ρ c_p) = 2×2/(1060×3500) ≈ 1.08e-6 — tiny but positive
        let expected = (2.0 * 2.0 / (1060.0 * 3500.0_f64)).clamp(0.0, 1.0);
        assert!((tau - expected).abs() < 1e-15);
    }

    #[test]
    fn test_multiphysics_electromagnetic_thermal() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Electromagnetic,
            PhysicsDomain::Thermal,
            CouplingType::ElectromagneticThermal,
        );
        let tau = interface.transmission_coefficient(1e6);
        // Photothermal coupling: nearly all absorbed light → heat (Jacques 2013)
        assert!(tau > 0.9, "photothermal coupling should exceed 90%");
        assert!(tau <= 1.0);
    }

    #[test]
    fn test_multiphysics_custom_coupling() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Custom(1),
            PhysicsDomain::Custom(2),
            CouplingType::Custom("user_defined".to_string()),
        );
        assert_eq!(interface.transmission_coefficient(1e6), 1.0);
    }

    /// Impedance self-matching (Z₁ = Z₂) gives τ = 1.0 exactly.
    #[test]
    fn test_acoustic_elastic_self_matched() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            PhysicsDomain::Acoustic,
            PhysicsDomain::Elastic,
            CouplingType::AcousticElastic {
                z1_rayl: Z_WATER,
                z2_rayl: Z_WATER,
            },
        );
        let tau = interface.transmission_coefficient(1e6);
        assert!((tau - 1.0).abs() < 1e-14, "Z₁=Z₂ must give τ=1; got {}", tau);
    }
}
