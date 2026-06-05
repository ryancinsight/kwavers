use super::*;

// ─── Epstein–Plesset free-bubble dissolution ─────────────────────────────────

#[test]
fn saturated_no_surface_tension_does_not_dissolve() {
    // f = 1 and σ = 0 ⇒ no concentration gradient ⇒ dR/dt = 0.
    let mut p = GasDiffusionParams::air_in_water(1.0);
    p.surface_tension = 0.0;
    let mut model = EpsteinPlessetDissolution::new(p);
    model.surface_tension = false;
    let rate = model.radius_rate(2e-6, 1e-3);
    assert!(
        rate.abs() < 1e-12,
        "saturated σ=0 bubble must be static; rate={rate}"
    );
}

#[test]
fn undersaturated_bubble_dissolves() {
    let model = EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(0.0));
    assert!(
        model.radius_rate(2e-6, 1e-3) < 0.0,
        "degassed liquid ⇒ dissolves"
    );
}

#[test]
fn supersaturated_bubble_grows() {
    // f = 2 (supersaturated) with negligible surface tension at large R ⇒ growth.
    let model = EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(2.0));
    assert!(
        model.radius_rate(20e-6, 1e-3) > 0.0,
        "supersaturated ⇒ grows"
    );
}

#[test]
fn surface_tension_dissolves_even_when_saturated() {
    // Classic Epstein–Plesset result: a free bubble dissolves in a SATURATED
    // liquid because of the Laplace overpressure (surface tension on).
    let model = EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(1.0));
    assert!(
        model.radius_rate(2e-6, 1e-3) < 0.0,
        "saturated free bubble must still dissolve via surface tension"
    );
}

#[test]
fn closed_form_dissolution_time_matches_numeric_quasi_static() {
    // Quasi-static, σ = 0, undersaturated: τ = R₀²/(2 D L (1−f)) should match the
    // numerically integrated dissolution time to a few percent.
    let mut p = GasDiffusionParams::air_in_water(0.0);
    p.surface_tension = 0.0;
    let mut model = EpsteinPlessetDissolution::quasi_static(p);
    model.surface_tension = false;
    let r0 = 2e-6;
    let closed = model.dissolution_time(r0).expect("closed form for f<1");
    let num = dissolution_time_numeric(&model, r0, 1e-9).expect("numeric dissolves");
    let rel = (num - closed).abs() / closed;
    assert!(
        rel < 0.05,
        "numeric {num:.4e}s vs closed {closed:.4e}s rel={rel:.3}"
    );
}

#[test]
fn dissolution_time_physically_reasonable_for_micron_bubble() {
    // A 1–2 µm air bubble in degassed/saturated water dissolves on a
    // milliseconds-to-tens-of-ms timescale (sets the residual-shielding τ_d).
    let model = EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(0.5));
    let tau = dissolution_time_numeric(&model, 2e-6, 1e-9).expect("dissolves");
    assert!(
        (1e-4..1.0).contains(&tau),
        "τ_d for 2 µm bubble should be ~0.1 ms–1 s; got {tau:.4e} s"
    );
}

#[test]
fn larger_bubble_dissolves_slower() {
    let model = EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(0.0));
    let t1 = dissolution_time_numeric(&model, 1e-6, 1e-9).unwrap();
    let t2 = dissolution_time_numeric(&model, 2e-6, 1e-9).unwrap();
    assert!(
        t2 > t1,
        "τ_d grows with R₀² (Epstein–Plesset); t1={t1:.3e}, t2={t2:.3e}"
    );
}

// ─── Shelled-microbubble (Sarkar permeation) ─────────────────────────────────

#[test]
fn shell_slows_dissolution_vs_free_bubble() {
    let p = GasDiffusionParams::air_in_water(0.0);
    let free = EpsteinPlessetDissolution::new(p);
    let shelled = ShellPermeationDissolution::lipid_shell(p);
    let r = 2e-6;
    let t = 1e-3;
    assert!(
        shelled.radius_rate(r, t).abs() < free.radius_rate(r, t).abs(),
        "shell permeation resistance must slow dissolution"
    );
}

#[test]
fn impermeable_shell_stabilizes_bubble() {
    let p = GasDiffusionParams::air_in_water(0.0);
    let shelled = ShellPermeationDissolution::new(p, 0.0); // k_s = 0
    assert_eq!(
        shelled.radius_rate(2e-6, 1e-3),
        0.0,
        "impermeable shell ⇒ no dissolution"
    );
}

#[test]
fn shelled_dissolution_time_exceeds_free() {
    let p = GasDiffusionParams::air_in_water(0.3);
    let free = EpsteinPlessetDissolution::new(p);
    let shelled = ShellPermeationDissolution::new(p, 1.0e-6);
    let tf = dissolution_time_numeric(&free, 2e-6, 1e-9).unwrap();
    let ts = dissolution_time_numeric(&shelled, 2e-6, 1e-9).unwrap();
    assert!(
        ts > tf,
        "coated MB persists longer: free={tf:.3e}s, shelled={ts:.3e}s"
    );
}

// ─── Integrator / trajectory ─────────────────────────────────────────────────

#[test]
fn trajectory_is_monotone_decreasing_and_volume_ratio_bounded() {
    let model = EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(0.0));
    let traj = integrate_dissolution(&model, 2e-6, 1e-5, 1.0, 1e-9);
    assert!(traj.dissolution_time.is_some(), "should fully dissolve");
    for k in 1..traj.radius.len() {
        assert!(
            traj.radius[k] <= traj.radius[k - 1] + 1e-15,
            "radius must not increase while dissolving"
        );
    }
    let vr = traj.volume_ratio();
    assert!((vr[0] - 1.0).abs() < 1e-9, "initial volume ratio = 1");
    assert!(vr.iter().all(|&v| (0.0..=1.0 + 1e-9).contains(&v)));
}

#[test]
fn integrator_rejects_bad_input() {
    let model = EpsteinPlessetDissolution::new(GasDiffusionParams::air_in_water(0.0));
    let traj = integrate_dissolution(&model, -1.0, 1e-5, 1.0, 1e-9);
    assert!(traj.time.is_empty() && traj.dissolution_time.is_none());
}
