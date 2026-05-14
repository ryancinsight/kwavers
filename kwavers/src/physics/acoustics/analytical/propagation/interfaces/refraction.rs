//! Refraction calculations for wave propagation

/// Refraction calculator
#[derive(Debug)]
pub struct RefractionCalculator {
    // Implementation details
}

/// Refraction angles
#[derive(Debug, Clone)]
pub struct RefractionAngles {
    /// Incident angle \[radians\]
    pub incident: f64,
    /// Refracted angle \[radians\]
    pub refracted: f64,
    /// Critical angle for total internal reflection \[radians\]
    pub critical: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// RefractionAngles stores all fields correctly, including None critical angle.
    #[test]
    fn stores_fields_with_no_critical_angle() {
        let a = RefractionAngles {
            incident: PI / 6.0,
            refracted: PI / 4.0,
            critical: None,
        };
        assert!((a.incident - PI / 6.0).abs() < 1e-15);
        assert!((a.refracted - PI / 4.0).abs() < 1e-15);
        assert!(a.critical.is_none());
    }

    /// RefractionAngles stores Some critical angle.
    #[test]
    fn stores_critical_angle_when_present() {
        let theta_c = (1500.0_f64 / 3400.0).asin(); // water–bone critical angle ≈ 26.2°
        let a = RefractionAngles {
            incident: 0.1,
            refracted: 0.0,
            critical: Some(theta_c),
        };
        let stored = a.critical.expect("critical angle must be Some");
        assert!(
            (stored - theta_c).abs() < 1e-15,
            "stored critical={stored}, expected={theta_c}"
        );
    }

    /// Clone produces an equal copy including the Option field.
    #[test]
    fn clone_produces_equal_values() {
        let original = RefractionAngles {
            incident: 0.3,
            refracted: 0.6,
            critical: Some(0.4),
        };
        let cloned = original.clone();
        assert!((original.incident - cloned.incident).abs() < 1e-15);
        assert!((original.refracted - cloned.refracted).abs() < 1e-15);
        assert_eq!(
            original.critical.map(|v| (v * 1e12) as i64),
            cloned.critical.map(|v| (v * 1e12) as i64)
        );
    }

    /// Snell's law: sin(θ_t) = (c2/c1) · sin(θ_i); verify stored angles satisfy it.
    ///
    /// Water (c1=1500) → fat (c2=1450): θ_i = 20° → θ_t = arcsin((1450/1500)·sin(20°)).
    #[test]
    fn snell_law_holds_for_water_fat_interface() {
        let c1 = 1500.0_f64;
        let c2 = 1450.0_f64;
        let theta_i = 20.0_f64.to_radians();
        let theta_t = ((c2 / c1) * theta_i.sin()).asin();

        let a = RefractionAngles {
            incident: theta_i,
            refracted: theta_t,
            critical: None,
        };

        // Verify Snell's law: c1·sin(θ_t) ≈ c2·sin(θ_i)
        let lhs = c1 * a.refracted.sin();
        let rhs = c2 * a.incident.sin();
        assert!(
            (lhs - rhs).abs() < 1e-10,
            "Snell: c1·sin(θ_t)={lhs:.6}, c2·sin(θ_i)={rhs:.6}"
        );
    }
}
