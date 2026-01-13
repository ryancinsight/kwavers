/// Shell properties for encapsulated bubbles
#[derive(Debug, Clone)]
pub struct ShellProperties {
    /// Shell thickness \[m\]
    pub thickness: f64,
    /// Shell shear modulus \[Pa\]
    pub shear_modulus: f64,
    /// Shell shear viscosity [Pa·s]
    pub shear_viscosity: f64,
    /// Shell density [kg/m³]
    pub density: f64,
    /// Initial surface tension [N/m] (for Marmottant model)
    pub sigma_initial: f64,
    /// Buckling radius \[m\] (Marmottant model)
    pub r_buckling: f64,
    /// Rupture radius \[m\] (Marmottant model)  
    pub r_rupture: f64,
    /// Elastic modulus for shell (Church model) [N/m]
    pub elastic_modulus: f64,
}

impl Default for ShellProperties {
    fn default() -> Self {
        // Typical lipid shell properties (SonoVue/Definity-like)
        // Based on Gorce et al. (2000) and Stride & Coussios (2010)
        Self {
            thickness: 3.0e-9,     // 3 nm (typical lipid shell)
            shear_modulus: 50.0e6, // 50 MPa (lipid)
            shear_viscosity: 0.5,  // 0.5 Pa·s (lipid)
            density: 1100.0,       // 1100 kg/m³ (lipid)
            sigma_initial: 0.04,   // 40 mN/m (initial tension)
            r_buckling: 0.0,       // Computed from equilibrium
            r_rupture: 0.0,        // Computed based on shell strain
            elastic_modulus: 0.5,  // 0.5 N/m (Church model)
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
            thickness: 200.0e-9,    // 200 nm (much thicker)
            shear_modulus: 500.0e6, // 500 MPa (very stiff)
            shear_viscosity: 5.0,   // 5 Pa·s (highly viscous)
            density: 1050.0,        // 1050 kg/m³
            sigma_initial: 0.072,   // 72 mN/m
            elastic_modulus: 5.0,   // 5 N/m (stiffer)
            ..Self::default()
        }
    }

    /// Compute buckling and rupture radii based on equilibrium
    ///
    /// # Arguments
    /// * `r0` - Equilibrium radius \[m\]
    /// * `p0` - Ambient pressure \[Pa\]
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
}
