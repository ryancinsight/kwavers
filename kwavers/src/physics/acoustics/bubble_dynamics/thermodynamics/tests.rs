use super::mass_transfer::MassTransferModel;
use super::vapor_pressure::{ThermodynamicsCalculator, VaporPressureModel};
use crate::core::constants::{P_ATM, T_CRITICAL_WATER};

#[test]
fn test_vapor_pressure_models() {
    let calc = ThermodynamicsCalculator::default();

    // Test at 100°C (373.15 K) - should be ~1 atm
    let t = 373.15;
    let p_wagner = calc.wagner_equation(t);
    assert!((p_wagner - P_ATM).abs() / P_ATM < 0.01); // Within 1%

    // Test Antoine equation
    let calc_antoine = ThermodynamicsCalculator::new(VaporPressureModel::Antoine);
    let p_antoine = calc_antoine.vapor_pressure(t);
    assert!((p_antoine - P_ATM).abs() / P_ATM < 0.02); // Within 2%

    // Test at 25°C (298.15 K) - should be ~3.17 kPa
    let t_room = 298.15;
    let p_room = calc.vapor_pressure(t_room);
    assert!((p_room - 3170.0).abs() < 100.0); // Within 100 Pa
}

#[test]
fn test_clausius_clapeyron() {
    let calc = ThermodynamicsCalculator::new(VaporPressureModel::ClausiusClapeyron);

    // Test at reference point
    let p_ref = calc.vapor_pressure(373.15);
    assert!((p_ref - P_ATM).abs() / P_ATM < 0.01);

    // Test temperature dependence
    let p_low = calc.vapor_pressure(353.15); // 80°C
    let p_high = calc.vapor_pressure(393.15); // 120°C

    assert!(p_low < P_ATM);
    assert!(p_high > P_ATM);
}

#[test]
fn test_mass_transfer() {
    let model = MassTransferModel::new(0.04); // Typical accommodation coefficient

    let temperature = 300.0; // K
    let p_vapor = 2000.0; // Pa (undersaturated)
    let surface_area = 1e-6; // m² (1 mm² bubble)

    let rate = model.mass_transfer_rate(temperature, p_vapor, surface_area);

    // Should be positive (evaporation) since p_vapor < p_sat
    assert!(rate > 0.0);
}

#[test]
fn test_enthalpy_vaporization() {
    let calc = ThermodynamicsCalculator::default();

    // At 100°C
    let h_vap_100 = calc.enthalpy_vaporization(373.15);
    assert!((h_vap_100 - 40660.0).abs() < 100.0); // Should match reference

    // Should decrease with temperature
    let h_vap_150 = calc.enthalpy_vaporization(423.15); // 150°C
    assert!(h_vap_150 < h_vap_100);

    // Should be zero at critical point
    let h_vap_crit = calc.enthalpy_vaporization(T_CRITICAL_WATER);
    assert!(h_vap_crit.abs() < 1.0);
}
