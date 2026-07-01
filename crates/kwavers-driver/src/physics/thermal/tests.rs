//! Consolidated thermal + ir_drop tests (Phase 3b).
//!
//! 6 thermal tests from `src/thermal.rs` + 3 ir_drop tests relocated from `src/pdn.rs`.

use super::{
    ir_drop, junction_temperature_k, solve_board, solve_poisson, temperature_derated_resistance,
    thermal_time_constant_s, thermal_via_conductance, transient_rise_k,
};
use crate::board::{Board, LayerId, NetClassKind, Track};
use crate::geom::{GridSpec, Nm, Point};

#[test]
fn transient_thermal_reaches_steady_state() {
    // τ = R_th·C_th; one τ ⇒ 63 %, five τ ⇒ >99 %.
    let tau = thermal_time_constant_s(20.0, 0.5); // 10 s
    assert!((tau - 10.0).abs() < 1e-9);
    assert!((transient_rise_k(22.0, tau, tau) - 22.0 * 0.632).abs() < 0.1);
    assert!(transient_rise_k(22.0, tau, 5.0 * tau) > 22.0 * 0.99);
    // By the paper's 18 s window (1.8 τ here) it is ~83 % settled.
    assert!(transient_rise_k(22.0, tau, 18.0) > 22.0 * 0.8);
}

#[test]
fn thermal_vias_add_conductance() {
    // One 0.3 mm via, 25 µm plating, 1.6 mm board: R = L/(kA) ⇒ G ≈ 5.9e-3 W/K.
    let g1 = thermal_via_conductance(1, 0.3e-3, 25e-6, 1.6e-3);
    assert!(
        (g1 - 5.89e-3).abs() < 1e-3,
        "expected ~5.9e-3 W/K, got {g1:.2e}"
    );
    // A stitched array of 9 vias is 9× the conductance (parallel).
    assert!((thermal_via_conductance(9, 0.3e-3, 25e-6, 1.6e-3) - 9.0 * g1).abs() < 1e-9);
    assert_eq!(thermal_via_conductance(0, 0.3e-3, 25e-6, 1.6e-3), 0.0);
}

#[test]
fn manufactured_solution_recovers_sine_field() {
    // T*(x,y) = sin(πx/L) sin(πy/W), zero on the Dirichlet boundary.
    // f* = (π²/L² + π²/W²) T*  must reproduce T* to O(h²).
    let spec = GridSpec::cover(Nm::from_mm(40.0), Nm::from_mm(30.0), Nm::from_mm(1.0), 1).unwrap();
    let (nx, ny) = (spec.nx, spec.ny);
    let h = spec.pitch.to_mm() * 1.0e-3;
    let l = (nx as f64 - 1.0) * h;
    let w = (ny as f64 - 1.0) * h;
    let kk = std::f64::consts::PI * std::f64::consts::PI * (1.0 / (l * l) + 1.0 / (w * w));

    let exact = |ix: usize, iy: usize| {
        (std::f64::consts::PI * ix as f64 * h / l).sin()
            * (std::f64::consts::PI * iy as f64 * h / w).sin()
    };
    let mut f = vec![0.0f64; nx * ny];
    for iy in 0..ny {
        for ix in 0..nx {
            f[iy * nx + ix] = kk * exact(ix, iy);
        }
    }
    let field = solve_poisson(spec, &f, 20.0, 1.6e-3, 0.0, 6000);

    let mut max_err = 0.0f64;
    for iy in 1..ny - 1 {
        for ix in 1..nx - 1 {
            let e = (field.temp[iy * nx + ix] - exact(ix, iy)).abs();
            max_err = max_err.max(e);
        }
    }
    // Truncation error of the 5-point Laplacian for this mode is ~ (kh)²/12 ≈ 0.4%; allow 1%.
    assert!(
        max_err < 0.01,
        "MMS error {max_err:.4} exceeds the O(h²) truncation bound"
    );
}

#[test]
fn peak_rises_with_power_and_sits_at_the_source() {
    use crate::place::component::Placement;
    use crate::place::footprint::{PadDef, Role};
    use crate::place::rotation::Rot;

    let spec = GridSpec::cover(Nm::from_mm(40.0), Nm::from_mm(40.0), Nm::from_mm(1.0), 1).unwrap();

    let lib = vec![crate::place::footprint::FootprintDef::new(
        "HOT",
        (Nm::from_mm(6.0), Nm::from_mm(6.0)),
        Role::ActiveIc,
        vec![PadDef {
            offset: Point::new(Nm(0), Nm(0)),
            size: (Nm::from_mm(0.5), Nm::from_mm(0.5)),
            layers: vec![LayerId(0)],
            power_pin: false,
        }],
    )];
    let comp = |x, y| crate::place::component::Component {
        fp: 0,
        nets: vec![None],
        refdes: "U".into(),
        placement: Placement {
            pos: Point::new(Nm::from_mm(x), Nm::from_mm(y)),
            rot: Rot::R0,
        },
        assoc_ic: None,
        locked: false,
        ..Default::default()
    };
    let watts = |_: &crate::place::footprint::FootprintDef| 2.0;
    let field = solve_board(
        spec,
        &[comp(20.0, 20.0)],
        &lib,
        watts,
        20.0,
        1.6e-3,
        10.0,
        4000,
    );
    assert!(
        field.peak() > 0.0,
        "a 2 W source must raise the temperature"
    );

    // Two sources spread apart yield a lower peak than both stacked at the centre.
    let stacked = solve_board(
        spec,
        &[comp(20.0, 20.0), comp(20.0, 20.0)],
        &lib,
        watts,
        20.0,
        1.6e-3,
        10.0,
        4000,
    );
    let spread = solve_board(
        spec,
        &[comp(12.0, 20.0), comp(28.0, 20.0)],
        &lib,
        watts,
        20.0,
        1.6e-3,
        10.0,
        4000,
    );
    assert!(
        spread.peak() < stacked.peak(),
        "spreading power devices ({:.2} K) must lower peak vs stacking them ({:.2} K)",
        spread.peak(),
        stacked.peak()
    );
}

#[test]
fn junction_temperature_sums_the_thermal_chain() {
    // T_ambient = 298 K (25 °C), board rise = 22 K, θ_jc = 40 K/W, P = 2 W.
    // T_j = 298 + 22 + 40·2 = 400 K (127 °C).
    let tj = junction_temperature_k(298.0, 22.0, 40.0, 2.0);
    assert!((tj - 400.0).abs() < 1e-9, "expected 400 K, got {tj:.1}");
    // Zero dissipation: T_j = ambient + board rise only.
    assert!((junction_temperature_k(298.0, 5.0, 40.0, 0.0) - 303.0).abs() < 1e-9);
    // Doubling power doubles the θ_jc contribution.
    let tj2 = junction_temperature_k(298.0, 22.0, 40.0, 4.0);
    assert!((tj2 - tj - 80.0).abs() < 1e-9);
}

#[test]
fn copper_resistance_rises_with_temperature() {
    // 1 Ω at 293 K (20 °C). At 393 K (120 °C): ΔT = 100 K, α = 3.93e-3 → R = 1·(1+0.393) ≈ 1.393 Ω.
    let r = temperature_derated_resistance(1.0, 293.0, 393.0, 3.93e-3);
    assert!((r - 1.393).abs() < 0.001, "expected ~1.393 Ω, got {r:.4}");
    // At reference temperature: R stays exactly R_dc.
    assert!((temperature_derated_resistance(0.5, 293.0, 293.0, 3.93e-3) - 0.5).abs() < 1e-12);
    // Below reference: resistance decreases.
    assert!(temperature_derated_resistance(1.0, 293.0, 253.0, 3.93e-3) < 1.0);
}

// ----- ir_drop tests relocated from src/pdn.rs -----

fn board_with_chain(width_mm: f64) -> (Board, crate::board::NetId, Point) {
    let spec = GridSpec::cover(Nm::from_mm(60.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut b = Board::new(spec);
    let vpp = b.add_net("VPP", NetClassKind::Hv);
    // A straight chain of 0.5 mm segments from x=0 to x=50 mm at y=5.
    let mut x = 0.0;
    while x < 50.0 {
        b.tracks.push(Track {
            start: Point::new(Nm::from_mm(x), Nm::from_mm(5.0)),
            end: Point::new(Nm::from_mm(x + 0.5), Nm::from_mm(5.0)),
            width: Nm::from_mm(width_mm),
            layer: LayerId(0),
            net: vpp,
        });
        x += 0.5;
    }
    (b, vpp, Point::new(Nm::from_mm(0.0), Nm::from_mm(5.0)))
}

#[test]
fn ir_drop_matches_series_resistance() {
    // 50 mm of 0.25 mm 1 oz track, 1 A: V = I·R, R = ρL/(W·t) ≈ 1.68e-8·0.05/(2.5e-4·3.48e-5)
    // ≈ 0.0965 Ω ⇒ ~0.097 V at the far end.
    let (b, vpp, supply) = board_with_chain(0.25);
    let d = ir_drop(&b, vpp, supply, 1.0, 1.0, 40000).unwrap();
    assert!(
        (d.max_drop_v - 0.0965).abs() < 0.01,
        "expected ~0.097 V series IR drop, got {:.4}",
        d.max_drop_v
    );
}

#[test]
fn wider_track_drops_less() {
    let (b1, vpp, supply) = board_with_chain(0.25);
    let (b2, vpp2, supply2) = board_with_chain(0.5);
    let d1 = ir_drop(&b1, vpp, supply, 1.0, 1.0, 40000).unwrap();
    let d2 = ir_drop(&b2, vpp2, supply2, 1.0, 1.0, 40000).unwrap();
    assert!(
        d2.max_drop_v < d1.max_drop_v * 0.6,
        "doubling width must roughly halve IR drop"
    );
}

#[test]
fn ir_drop_uses_power_oz_tracks_have_different_drop() {
    // Sanity: with twice the copper weight (2 oz vs 1 oz), the same current yields a smaller
    // voltage drop — i.e., the ir_drop solver respects copper_oz as a parameter.
    let (b, vpp, supply) = board_with_chain(0.5);
    let d1 = ir_drop(&b, vpp, supply, 1.0, 1.0, 40000).unwrap();
    let d2 = ir_drop(&b, vpp, supply, 1.0, 2.0, 40000).unwrap();
    assert!(
        d2.max_drop_v < d1.max_drop_v,
        "2-oz copper must yield smaller IR drop than 1-oz ({:.4} vs {:.4})",
        d2.max_drop_v,
        d1.max_drop_v
    );
}

#[test]
fn ir_drop_boundary_zero_current() {
    // With zero load current every node stays at supply potential ⇒ max drop = 0 V exactly.
    let (b, vpp, supply) = board_with_chain(0.25);
    let d = ir_drop(&b, vpp, supply, 0.0, 1.0, 100).unwrap();
    assert_eq!(d.max_drop_v, 0.0, "zero current must produce zero IR drop");
}

#[test]
fn ir_drop_boundary_no_tracks_returns_none() {
    // An empty board has no routed tracks on the net ⇒ function returns None.
    let spec = GridSpec::cover(Nm::from_mm(60.0), Nm::from_mm(10.0), Nm::from_mm(0.5), 2).unwrap();
    let mut b = Board::new(spec);
    let vpp = b.add_net("VPP", NetClassKind::Hv);
    let supply = Point::new(Nm::from_mm(0.0), Nm::from_mm(5.0));
    assert!(
        ir_drop(&b, vpp, supply, 1.0, 1.0, 100).is_none(),
        "a net with no routed tracks must produce None"
    );
}
