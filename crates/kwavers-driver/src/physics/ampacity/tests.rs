use super::*;
use crate::geom::Nm;

#[test]
fn ipc2221_canonical_point() {
    let w = ipc2221_min_width(1.0, 10.0, 1.0, false).to_mm();
    assert!((w - 0.30).abs() < 0.05, "expected ~0.30 mm, got {w:.3}");
    assert!(
        ipc2221_min_width(2.0, 10.0, 1.0, false) > ipc2221_min_width(1.0, 10.0, 1.0, false)
    );
    assert!(ipc2221_min_width(1.0, 10.0, 1.0, true) > ipc2221_min_width(1.0, 10.0, 1.0, false));
}

#[test]
fn skin_depth_at_2mhz_is_about_46um() {
    let d = skin_depth_m(2.0e6);
    assert!(
        (d - 46.0e-6).abs() < 3.0e-6,
        "expected ~46 µm, got {:.1} µm",
        d * 1e6
    );
    // 1 oz copper (35 µm) at 2 MHz: t < δ ⇒ AC factor near unity (skin effect mild).
    let f = ac_resistance_factor(35e-6, 2.0e6);
    assert!(
        (1.0..1.25).contains(&f),
        "1 oz at 2 MHz should be ~1.0–1.2, got {f:.3}"
    );
    // Heavy copper at high frequency shows a clear AC penalty.
    assert!(ac_resistance_factor(300e-6, 50.0e6) > 1.5);
}

#[test]
fn annular_ring_meets_floor() {
    // 0.46 mm via / 0.2 mm drill ⇒ 0.13 mm ring, exactly the fab floor.
    assert!((annular_ring_mm(0.46, 0.2) - 0.13).abs() < 1e-9);
    assert!(annular_ring_mm(0.46, 0.2) >= 0.13);
}

#[test]
fn pth_aspect_ratio_flags_thin_drills() {
    // 1.6 mm board, 0.2 mm drill ⇒ 8:1 (OK ≤10); 0.1 mm drill ⇒ 16:1 (risky).
    assert!((pth_aspect_ratio(1.6e-3, 0.2e-3) - 8.0).abs() < 1e-9);
    assert!(pth_aspect_ratio(1.6e-3, 0.1e-3) > 10.0);
}

#[test]
fn electromigration_mttf_degrades_with_current_and_heat() {
    // 1 A in 0.25 mm 1 oz: J = I/(W·t) = 1/(0.25·0.0348) ≈ 115 A/mm².
    let j = current_density_a_per_mm2(1.0, 0.25e-3, 1.0);
    assert!((j - 115.0).abs() < 5.0, "expected ~115 A/mm², got {j:.1}");
    // Doubling current density (×4 in J^2) and raising temperature both cut MTTF below 1.
    let worse = black_mttf_relative(j, 300.0, 2.0 * j, 350.0, 2.0, 0.9);
    assert!(
        worse < 1.0,
        "higher J and T must shorten EM lifetime, got {worse:.3}"
    );
    // Same conditions ⇒ ratio 1.
    assert!((black_mttf_relative(j, 300.0, j, 300.0, 2.0, 0.9) - 1.0).abs() < 1e-9);
}

#[test]
fn resistance_scales_inversely_with_width() {
    // 100 mm of 0.25 mm, 1 oz track: R = ρL/(W·t) ≈ 1.68e-8·0.1/(2.5e-4·3.48e-5) ≈ 0.193 Ω.
    let r = track_resistance(0.1, 0.25e-3, 1.0);
    assert!((r - 0.193).abs() < 0.01, "expected ~0.193 Ω, got {r:.3}");
    // Doubling width halves resistance.
    assert!((track_resistance(0.1, 0.5e-3, 1.0) - r / 2.0).abs() < 1e-6);
}

#[test]
fn detects_undersized_hv_track() {
    use crate::board::{Board, LayerId, NetClassKind, Track};
    use crate::geom::{GridSpec, Point};
    let spec =
        GridSpec::cover(Nm::from_mm(20.0), Nm::from_mm(20.0), Nm::from_mm(0.5), 2).unwrap();
    let mut b = Board::new(spec);
    let vpp = b.add_net("VPP", NetClassKind::Hv);
    b.tracks.push(Track {
        start: Point::new(Nm::from_mm(2.0), Nm::from_mm(2.0)),
        end: Point::new(Nm::from_mm(8.0), Nm::from_mm(2.0)),
        width: Nm::from_mm(0.25),
        layer: LayerId(0),
        net: vpp,
    });
    // 3 A needs ~0.7 mm; a 0.25 mm track is a deficit.
    let d = ampacity_check(&b, |_| 3.0, 10.0, 1.0);
    assert_eq!(d.len(), 1);
    assert!(d[0].required_mm > d[0].actual_mm);
    // At 0.5 A the 0.25 mm track is adequate.
    assert!(ampacity_check(&b, |_| 0.5, 10.0, 1.0).is_empty());
}
