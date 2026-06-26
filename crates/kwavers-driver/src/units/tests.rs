//! Consolidated tests for the `units` slice (Phase 4k carve-out): the `Nm` integer-length newtype,
//! the prefix factories, same-unit + scalar arithmetic, the cross-unit dimensional algebra (Ohm's
//! law etc.), temperature conversions, and the SI-suffix `Display`. Moved verbatim from the flat
//! `src/units.rs` `mod tests` block; `super::*` resolves the slice facade.

use super::*;

#[test]
fn nm_mm_roundtrip_is_exact_at_micron() {
    assert_eq!(Nm::from_mm(1.27), Nm(1_270_000));
    assert_eq!(Nm(1_270_000).to_mm(), 1.27);
}

#[test]
fn nm_arithmetic_works() {
    let a = Nm::from_mm(2.5);
    let b = Nm::from_mm(1.5);
    assert_eq!(a + b, Nm::from_mm(4.0));
    assert_eq!(a - b, Nm::from_mm(1.0));
    assert_eq!(b * 3, Nm::from_mm(4.5));
    assert_eq!(-b, Nm(-1_500_000));
    assert_eq!(b.abs(), Nm::from_mm(1.5));
}

#[test]
fn hz_khz_mhz_round_trip() {
    let f = Hz::from_khz(2.5);
    assert!((f.value() - 2_500.0).abs() < 1e-9);
    assert!((f.to_khz() - 2.5).abs() < 1e-9);
    let g = Hz::from_mhz(2.0);
    assert!((g.value() - 2e6).abs() < 1e-9);
    assert!((g.to_mhz() - 2.0).abs() < 1e-9);
    let h = Hz::from_ghz(1.5);
    assert!((h.value() - 1.5e9).abs() < 1e-6);
}

#[test]
fn hz_addition_is_same_unit() {
    let a = Hz::from_khz(1.0);
    let b = Hz::from_khz(2.0);
    assert!((a + b).to_khz() - 3.0 < 1e-12);
}

#[test]
fn ohm_amp_volt_is_ohms_law() {
    let r = Ohm::from(8.0);
    let i = Amp::from(1.5);
    let v = r * i; // Ohm × Amp = Volt
    assert!((v.value() - 12.0).abs() < 1e-12);
}

#[test]
fn volt_amp_watt_is_joules_per_second() {
    let v = Volt::from(150.0);
    let i = Amp::from(1.5);
    let p = v * i; // Volt × Amp = Watt
    assert!((p.value() - 225.0).abs() < 1e-12);
}

#[test]
fn volt_per_ohm_is_amp() {
    let v = Volt::from(150.0);
    let r = Ohm::from(100.0);
    let i = v / r;
    assert!((i.value() - 1.5).abs() < 1e-12);
}

#[test]
fn farad_volt_is_coulomb() {
    let c = Farad::from_pf(50.0);
    let v = Volt::from(150.0);
    let q = c * v;
    assert!((q.value() - 7.5e-9).abs() < 1e-18);
}

#[test]
fn temperature_celsius_kelvin_conversion() {
    // 25 °C = 298.15 K
    let k = Kelvin::from_celsius(25.0);
    assert!((k.value() - 298.15).abs() < 1e-12);
    assert!((k.to_celsius() - 25.0).abs() < 1e-12);
    // Backwards: Kelvin → Celsius
    let c = Celsius::from_kelvin(310.15);
    assert!((c.value() - 37.0).abs() < 1e-12);
    // Fahrenheit round-trip: 25 °C = 77 °F
    let f = Celsius::from(25.0).to_fahrenheit();
    assert!((f - 77.0).abs() < 1e-9);
}

#[test]
fn display_shows_unit_suffix() {
    // Default Rust `{}` f64 formatting is the shortest round-trip decimal —
    // scientific notation only for values too small/large to print otherwise.
    assert_eq!(Hz::from(50.0).to_string(), "50 Hz");
    assert_eq!(Hz::from_mhz(2.0).to_string(), "2000000 Hz");
    assert_eq!(Ohm::from(50.0).to_string(), "50 Ω");
    assert_eq!(Watt::from_mw(2250.0).to_string(), "2.25 W");
    assert_eq!(Volt::from(150.0).to_string(), "150 V");
    assert_eq!(Amp::from(1.5).to_string(), "1.5 A");
    // 50 pF is below the threshold where Rust uses scientific notation →
    // the natural decimal representation prints as the exact round-trip form.
    assert_eq!(Farad::from_pf(50.0).to_string(), "0.00000000005 F");
    assert_eq!(Henry::from_nh(10.0).to_string(), "0.00000001 H");
    assert_eq!(Kelvin::from_celsius(25.0).to_string(), "298.15 K");
    assert_eq!(Celsius::from(37.0).to_string(), "37 °C"); // Convenience: a hand-rolled pretty-printer picks the most readable prefix.
                                                          // Kept as a free function to avoid baking auto-scaling into `Display`,
                                                          // which is round-trip stable in spec. `{:8.3}` on 5e-11 renders as
                                                          // 3 leading spaces + `0.000` (because the value is below the precision).
    assert_eq!(
        format!("{:8.3} {}", Farad::from_pf(50.0).value(), "F"),
        "   0.000 F"
    );
}

#[test]
fn scalar_mul_freq() {
    let f = Hz::from(50.0);
    assert!((f * 3.0).value() - 150.0 < 1e-12);
    assert!((2.0 * f).value() - 100.0 < 1e-12);
}

#[test]
fn watt_div_amp_is_volt() {
    let p = Watt::from(225.0);
    let i = Amp::from(1.5);
    let v = p / i; // Watt / Amp = Volt
    assert!((v.value() - 150.0).abs() < 1e-12);
}
