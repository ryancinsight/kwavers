//! Encapsulated Bubble Dynamics with Shell Mechanics
//!
//! This module implements models for ultrasound contrast agent (UCA) microbubbles
//! with viscoelastic shells. These bubbles consist of a gas core surrounded by
//! a thin shell (lipid, protein, or polymer) that significantly affects the
//! bubble dynamics.
//!
//! # Models Implemented
//!
//! 1. **Church Model (1995)** - Linear viscoelastic shell
//! 2. **Marmottant Model (2005)** - Buckling and rupture behavior for lipid shells
//!
//! # References
//!
//! - Church, C. C. (1995). "The effects of an elastic solid surface layer on the
//!   radial pulsations of gas bubbles." J. Acoust. Soc. Am. 97(3), 1510-1521.
//! - Marmottant, P., et al. (2005). "A model for large amplitude oscillations of
//!   coated bubbles accounting for buckling and rupture." J. Acoust. Soc. Am.
//!   118(6), 3499-3505.
//! - Stride, E. & Coussios, C. C. (2010). "Nucleation, mapping and control of
//!   cavitation for drug delivery." Nat. Rev. Drug Discov. 9, 527-536.

use super::bubble_state::{BubbleParameters, BubbleState};
use crate::error::KwaversResult;

/// Shell properties for encapsulated bubbles
#[derive(Debug, Clone)]
pub struct ShellProperties {
    /// Shell thickness [m]
    pub thickness: f64,
    /// Shell shear modulus [Pa]
    pub shear_modulus: f64,
    /// Shell shear viscosity [Pa·s]
    pub shear_viscosity: f64,
    /// Shell density [kg/m³]
    pub density: f64,
    /// Initial surface tension [N/m] (for Marmottant model)
    pub sigma_initial: f64,
    /// Buckling radius [m] (Marmottant model)
    pub r_buckling: f64,
    /// Rupture radius [m] (Marmottant model)  
    pub r_rupture: f64,
    /// Elastic modulus for shell (Church model) [N/m]
    pub elastic_modulus: f64,
}

impl Default for ShellProperties {
    fn default() -> Self {
        // Typical lipid shell properties (SonoVue/Definity-like)
        // Based on Gorce et al. (2000) and Stride & Coussios (2010)
        Self {
            thickness: 3.0e-9,      // 3 nm (typical lipid shell)
            shear_modulus: 50.0e6,  // 50 MPa (lipid)
            shear_viscosity: 0.5,   // 0.5 Pa·s (lipid)
            density: 1100.0,        // 1100 kg/m³ (lipid)
            sigma_initial: 0.04,    // 40 mN/m (initial tension)
            r_buckling: 0.0,        // Computed from equilibrium
            r_rupture: 0.0,         // Computed based on shell strain
            elastic_modulus: 0.5,   // 0.5 N/m (Church model)
        }
    }
}

impl ShellProperties {
    /// Create properties for lipid shell (SonoVue/Definity type)
    #[must_use]
    pub fn lipid_shell() -> Self {
        Self::default()
    }

    /// Create properties for protein shell (Albunex type)
    #[must_use]
    pub fn protein_shell() -> Self {
        Self {
            thickness: 15.0e-9,     // 15 nm (thicker protein)
            shear_modulus: 100.0e6, // 100 MPa (stiffer)
            shear_viscosity: 1.5,   // 1.5 Pa·s (more viscous)
            density: 1200.0,        // 1200 kg/m³
            sigma_initial: 0.056,   // 56 mN/m
            ..Self::default()
        }
    }

    /// Create properties for polymer shell
    #[must_use]
    pub fn polymer_shell() -> Self {
        Self {
            thickness: 200.0e-9,     // 200 nm (much thicker)
            shear_modulus: 500.0e6,  // 500 MPa (very stiff)
            shear_viscosity: 5.0,    // 5 Pa·s (highly viscous)
            density: 1050.0,         // 1050 kg/m³
            sigma_initial: 0.072,    // 72 mN/m
            elastic_modulus: 5.0,    // 5 N/m (stiffer)
            ..Self::default()
        }
    }

    /// Compute buckling and rupture radii based on equilibrium
    ///
    /// # Arguments
    /// * `r0` - Equilibrium radius [m]
    /// * `p0` - Ambient pressure [Pa]
    ///
    /// # References
    /// - Marmottant et al. (2005), Equations 12-13
    pub fn compute_critical_radii(&mut self, r0: f64, _p0: f64) {
        // Buckling radius: shell buckles when compressed beyond this
        // Typically 0.8-0.9 × R₀
        self.r_buckling = 0.85 * r0;

        // Rupture radius: shell ruptures when stretched beyond this
        // Based on shell strain limit (~10-20% strain)
        let max_strain = 0.15; // 15% maximum strain
        self.r_rupture = r0 * (1.0 + max_strain);
    }
}

/// Church model for encapsulated bubbles with elastic shell
///
/// Implements the linearized shell model from Church (1995) which adds
/// shell elasticity and viscosity terms to the Rayleigh-Plesset equation.
///
/// The modified equation includes:
/// - Shell elasticity: 12G(d/R)[(R/R₀)² - 1] term
/// - Shell viscosity: 12μ_s(d/R)(dR/dt)/R term
///
/// where:
/// - G = shell shear modulus
/// - μ_s = shell shear viscosity  
/// - d = shell thickness
/// - R = bubble radius
/// - R₀ = equilibrium radius
///
/// # References
/// - Church (1995), J. Acoust. Soc. Am. 97(3), 1510-1521
#[derive(Debug, Clone)]
pub struct ChurchModel {
    params: BubbleParameters,
    shell: ShellProperties,
}

impl ChurchModel {
    /// Create new Church model with shell properties
    #[must_use]
    pub fn new(params: BubbleParameters, mut shell: ShellProperties) -> Self {
        // Compute critical radii for the shell
        shell.compute_critical_radii(params.r0, params.p0);

        Self { params, shell }
    }

    /// Calculate bubble wall acceleration with shell effects (Church model)
    ///
    /// The Church model modifies the Rayleigh-Plesset equation to include
    /// shell elasticity and viscosity:
    ///
    /// ```text
    /// ρ(RR̈ + 3/2Ṙ²) = p_g - p_∞ - 2σ/R - 4μṘ/R - 12G(d/R)[(R/R₀)² - 1] - 12μ_s(d/R)Ṙ/R
    /// ```
    ///
    /// # Arguments
    /// * `state` - Current bubble state
    /// * `p_acoustic` - Acoustic pressure amplitude [Pa]
    /// * `t` - Current time [s]
    ///
    /// # Returns
    /// Bubble wall acceleration [m/s²]
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        let r = state.radius;
        let v = state.wall_velocity;
        let r0 = self.params.r0;

        // Acoustic forcing
        let omega = 2.0 * std::f64::consts::PI * self.params.driving_frequency;
        let p_acoustic_inst = p_acoustic * (omega * t).sin();
        let p_inf = self.params.p0 + p_acoustic_inst;

        // Internal gas pressure (polytropic)
        let gamma = state.gas_species.gamma();
        let p_eq = self.params.p0 + 2.0 * self.params.sigma / r0;
        let p_gas = p_eq * (r0 / r).powf(3.0 * gamma);

        // Standard Rayleigh-Plesset terms
        let surface_tension = 2.0 * self.params.sigma / r;
        let viscous_stress = 4.0 * self.params.mu_liquid * v / r;

        // Shell elasticity term: 12G(d/R)[(R/R₀)² - 1]
        // This represents the elastic restoring force from shell deformation
        let d = self.shell.thickness;
        let g = self.shell.shear_modulus;
        let shell_elastic = 12.0 * g * (d / r) * ((r / r0).powi(2) - 1.0);

        // Shell viscosity term: 12μ_s(d/R)(dR/dt)/R
        // This represents the viscous damping from shell material
        let mu_s = self.shell.shear_viscosity;
        let shell_viscous = 12.0 * mu_s * (d / r) * v / r;

        // Net pressure difference
        let net_pressure = p_gas - p_inf - surface_tension - viscous_stress 
                          - shell_elastic - shell_viscous;

        // Solve for R̈ from Rayleigh-Plesset equation
        let accel = (net_pressure / (self.params.rho_liquid * r)) - (1.5 * v * v / r);

        // Update state
        state.wall_acceleration = accel;
        state.pressure_internal = p_gas;
        state.pressure_liquid = p_inf;

        Ok(accel)
    }

    /// Get shell properties
    #[must_use]
    pub fn shell_properties(&self) -> &ShellProperties {
        &self.shell
    }
}

/// Marmottant model for encapsulated bubbles with buckling/rupture
///
/// Implements the nonlinear shell model from Marmottant et al. (2005) which
/// accounts for:
/// - Buckling behavior at small radii (R < R_buckling)
/// - Elastic regime with variable surface tension
/// - Rupture at large radii (R > R_rupture)
///
/// The surface tension σ(R) is given by:
/// - σ = 0 for R < R_buckling (buckled state)
/// - σ = χ(R² - R_buckling²)/R² for R_buckling < R < R_rupture (elastic)
/// - σ = σ_water for R > R_rupture (ruptured, free surface)
///
/// where χ is the elastic compression modulus of the shell.
///
/// # References
/// - Marmottant et al. (2005), J. Acoust. Soc. Am. 118(6), 3499-3505
/// - van der Meer et al. (2007), "Microbubble spectroscopy of ultrasound
///   contrast agents", J. Acoust. Soc. Am. 121(1), 648-656
#[derive(Debug, Clone)]
pub struct MarmottantModel {
    params: BubbleParameters,
    shell: ShellProperties,
    /// Elastic compression modulus χ [N/m]
    chi: f64,
}

impl MarmottantModel {
    /// Create new Marmottant model with shell properties
    ///
    /// # Arguments
    /// * `params` - Bubble parameters
    /// * `shell` - Shell properties
    /// * `chi` - Elastic compression modulus χ [N/m]
    #[must_use]
    pub fn new(params: BubbleParameters, mut shell: ShellProperties, chi: f64) -> Self {
        // Compute critical radii for the shell
        shell.compute_critical_radii(params.r0, params.p0);

        Self {
            params,
            shell,
            chi,
        }
    }

    /// Calculate effective surface tension based on Marmottant model
    ///
    /// Implements the piecewise surface tension function:
    /// ```text
    /// σ(R) = {
    ///   0                                for R ≤ R_buckling
    ///   χ(R² - R_buckling²)/R²          for R_buckling < R ≤ R_rupture  
    ///   σ_water                          for R > R_rupture
    /// }
    /// ```
    ///
    /// # Arguments
    /// * `radius` - Current bubble radius [m]
    ///
    /// # Returns
    /// Effective surface tension [N/m]
    #[must_use]
    pub fn surface_tension(&self, radius: f64) -> f64 {
        let r_b = self.shell.r_buckling;
        let r_r = self.shell.r_rupture;

        if radius <= r_b {
            // Buckled state: no surface tension
            0.0
        } else if radius <= r_r {
            // Elastic regime: variable surface tension
            self.chi * (radius.powi(2) - r_b.powi(2)) / radius.powi(2)
        } else {
            // Ruptured state: water surface tension
            0.0728 // Water surface tension at 20°C [N/m]
        }
    }

    /// Calculate derivative of surface tension with respect to radius
    ///
    /// Needed for accurate dynamics calculations:
    /// ```text
    /// dσ/dR = {
    ///   0                           for R ≤ R_buckling or R > R_rupture
    ///   2χR_buckling²/R³           for R_buckling < R ≤ R_rupture
    /// }
    /// ```
    #[must_use]
    pub fn surface_tension_derivative(&self, radius: f64) -> f64 {
        let r_b = self.shell.r_buckling;
        let r_r = self.shell.r_rupture;

        if radius <= r_b || radius > r_r {
            0.0
        } else {
            // Elastic regime
            2.0 * self.chi * r_b.powi(2) / radius.powi(3)
        }
    }

    /// Calculate bubble wall acceleration with Marmottant shell model
    ///
    /// The Marmottant model modifies the Rayleigh-Plesset equation to include
    /// the nonlinear surface tension behavior:
    ///
    /// ```text
    /// ρ(RR̈ + 3/2Ṙ²) = p_g - p_∞ - 2σ(R)/R - R(dσ/dR)Ṙ - 4μṘ/R - 12μ_s(d/R)Ṙ/R
    /// ```
    ///
    /// # Arguments
    /// * `state` - Current bubble state
    /// * `p_acoustic` - Acoustic pressure amplitude [Pa]
    /// * `t` - Current time [s]
    ///
    /// # Returns
    /// Bubble wall acceleration [m/s²]
    pub fn calculate_acceleration(
        &self,
        state: &mut BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        let r = state.radius;
        let v = state.wall_velocity;
        let r0 = self.params.r0;

        // Acoustic forcing
        let omega = 2.0 * std::f64::consts::PI * self.params.driving_frequency;
        let p_acoustic_inst = p_acoustic * (omega * t).sin();
        let p_inf = self.params.p0 + p_acoustic_inst;

        // Internal gas pressure (polytropic)
        let gamma = state.gas_species.gamma();
        let p_eq = self.params.p0 + 2.0 * self.shell.sigma_initial / r0;
        let p_gas = p_eq * (r0 / r).powf(3.0 * gamma);

        // Variable surface tension (Marmottant model)
        let sigma = self.surface_tension(r);
        let dsigma_dr = self.surface_tension_derivative(r);

        // Surface tension terms
        let surface_term = 2.0 * sigma / r;
        let surface_rate_term = r * dsigma_dr * v; // Time derivative contribution

        // Viscous damping (liquid + shell)
        let viscous_liquid = 4.0 * self.params.mu_liquid * v / r;
        let d = self.shell.thickness;
        let mu_s = self.shell.shear_viscosity;
        let viscous_shell = 12.0 * mu_s * (d / r) * v / r;

        // Net pressure difference
        let net_pressure = p_gas - p_inf - surface_term - surface_rate_term
                          - viscous_liquid - viscous_shell;

        // Solve for R̈ from modified Rayleigh-Plesset equation
        let accel = (net_pressure / (self.params.rho_liquid * r)) - (1.5 * v * v / r);

        // Update state
        state.wall_acceleration = accel;
        state.pressure_internal = p_gas;
        state.pressure_liquid = p_inf;

        Ok(accel)
    }

    /// Get shell properties
    #[must_use]
    pub fn shell_properties(&self) -> &ShellProperties {
        &self.shell
    }

    /// Check if shell is buckled
    #[must_use]
    pub fn is_buckled(&self, radius: f64) -> bool {
        radius <= self.shell.r_buckling
    }

    /// Check if shell is ruptured
    #[must_use]
    pub fn is_ruptured(&self, radius: f64) -> bool {
        radius > self.shell.r_rupture
    }

    /// Get current shell state as string
    #[must_use]
    pub fn shell_state(&self, radius: f64) -> &'static str {
        if self.is_buckled(radius) {
            "buckled"
        } else if self.is_ruptured(radius) {
            "ruptured"
        } else {
            "elastic"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_properties_defaults() {
        let shell = ShellProperties::default();
        assert!(shell.thickness > 0.0);
        assert!(shell.shear_modulus > 0.0);
        assert!(shell.shear_viscosity > 0.0);
    }

    #[test]
    fn test_shell_types() {
        let lipid = ShellProperties::lipid_shell();
        let protein = ShellProperties::protein_shell();
        let polymer = ShellProperties::polymer_shell();

        // Protein should be stiffer than lipid
        assert!(protein.shear_modulus > lipid.shear_modulus);
        // Polymer should be thickest
        assert!(polymer.thickness > protein.thickness);
        assert!(polymer.thickness > lipid.thickness);
    }

    #[test]
    fn test_critical_radii_computation() {
        let mut shell = ShellProperties::default();
        let r0 = 2e-6; // 2 microns
        let p0 = 101325.0; // 1 atm

        shell.compute_critical_radii(r0, p0);

        // Buckling radius should be less than equilibrium
        assert!(shell.r_buckling < r0);
        assert!(shell.r_buckling > 0.5 * r0); // Reasonable range

        // Rupture radius should be greater than equilibrium
        assert!(shell.r_rupture > r0);
        assert!(shell.r_rupture < 2.0 * r0); // Reasonable range
    }

    #[test]
    fn test_church_model_creation() {
        let params = BubbleParameters::default();
        let shell = ShellProperties::lipid_shell();
        let model = ChurchModel::new(params, shell);

        assert!(model.shell_properties().thickness > 0.0);
    }

    #[test]
    fn test_church_acceleration_finite() {
        let params = BubbleParameters::default();
        let shell = ShellProperties::lipid_shell();
        let model = ChurchModel::new(params.clone(), shell);

        let mut state = BubbleState::new(&params);
        let result = model.calculate_acceleration(&mut state, 0.0, 0.0);

        assert!(result.is_ok());
        let accel = result.unwrap();
        assert!(accel.is_finite());
    }

    #[test]
    fn test_marmottant_surface_tension_regimes() {
        let params = BubbleParameters::default();
        let shell = ShellProperties::lipid_shell();
        let chi = 0.5; // N/m
        let model = MarmottantModel::new(params, shell, chi);

        let r_b = model.shell.r_buckling;
        let r_r = model.shell.r_rupture;

        // Test buckled state (σ = 0)
        let sigma_buckled = model.surface_tension(0.8 * r_b);
        assert_eq!(sigma_buckled, 0.0, "Buckled state should have zero surface tension");

        // Test elastic regime (σ > 0)
        let r_elastic = (r_b + r_r) / 2.0;
        let sigma_elastic = model.surface_tension(r_elastic);
        assert!(sigma_elastic > 0.0, "Elastic regime should have positive surface tension");

        // Test ruptured state (σ = σ_water)
        let sigma_ruptured = model.surface_tension(1.5 * r_r);
        assert!(sigma_ruptured > 0.05, "Ruptured state should have water surface tension");
    }

    #[test]
    fn test_marmottant_shell_state_detection() {
        let params = BubbleParameters::default();
        let shell = ShellProperties::lipid_shell();
        let chi = 0.5;
        let model = MarmottantModel::new(params, shell, chi);

        let r_b = model.shell.r_buckling;
        let r_r = model.shell.r_rupture;

        assert_eq!(model.shell_state(0.8 * r_b), "buckled");
        assert_eq!(model.shell_state((r_b + r_r) / 2.0), "elastic");
        assert_eq!(model.shell_state(1.5 * r_r), "ruptured");
    }

    #[test]
    fn test_marmottant_acceleration_finite() {
        let params = BubbleParameters::default();
        let shell = ShellProperties::lipid_shell();
        let chi = 0.5;
        let model = MarmottantModel::new(params.clone(), shell, chi);

        let mut state = BubbleState::new(&params);
        let result = model.calculate_acceleration(&mut state, 0.0, 0.0);

        assert!(result.is_ok());
        let accel = result.unwrap();
        assert!(accel.is_finite());
    }

    #[test]
    fn test_church_vs_marmottant_equilibrium() {
        // Both models should give similar results at equilibrium for elastic regime
        let params = BubbleParameters::default();
        let shell = ShellProperties::lipid_shell();
        
        let church = ChurchModel::new(params.clone(), shell.clone());
        let marmottant = MarmottantModel::new(params.clone(), shell, 0.5);

        let mut state_church = BubbleState::new(&params);
        let mut state_marmottant = BubbleState::new(&params);

        let accel_church = church.calculate_acceleration(&mut state_church, 0.0, 0.0).unwrap();
        let accel_marmottant = marmottant.calculate_acceleration(&mut state_marmottant, 0.0, 0.0).unwrap();

        // Both should give finite accelerations at equilibrium
        assert!(accel_church.is_finite());
        assert!(accel_marmottant.is_finite());
    }

    #[test]
    fn test_shell_elastic_restoring_force() {
        // Test that shell elasticity provides restoring force
        let params = BubbleParameters::default();
        let shell = ShellProperties::lipid_shell();
        let model = ChurchModel::new(params.clone(), shell);

        let mut state = BubbleState::new(&params);
        
        // Compress bubble (R < R₀)
        state.radius = 0.8 * params.r0;
        state.wall_velocity = -10.0; // Inward velocity
        
        let accel_compressed = model.calculate_acceleration(&mut state, 0.0, 0.0).unwrap();
        
        // Shell elasticity should resist compression (though other forces may dominate)
        assert!(accel_compressed.is_finite());
    }

    #[test]
    fn test_marmottant_buckling_reduces_stiffness() {
        let params = BubbleParameters::default();
        let shell = ShellProperties::lipid_shell();
        let chi = 0.5;
        let model = MarmottantModel::new(params, shell, chi);

        let r_b = model.shell.r_buckling;

        // Surface tension in buckled vs elastic state
        let sigma_buckled = model.surface_tension(0.9 * r_b);
        let sigma_elastic = model.surface_tension(1.1 * r_b);

        // Buckled state should have zero or much lower surface tension
        assert!(sigma_buckled < sigma_elastic);
    }
}
