//! Multi-scale planar inversion schedule and stability diagnostics.

/// Return the frequency-continuation schedule and print its numerical contract.
pub(super) fn configure(dt: f64, t_transit: f64) -> &'static [(f64, usize)] {
    // CPML absorption is adequate from 40 kHz for the 30 mm absorbing layer.
    // Frequency continuation starts below the 150 kHz cycle-skipping limit and
    // carries each converged model into the next scale.
    const SCALES: &[(f64, usize)] = &[(40_000.0, 10), (80_000.0, 12), (150_000.0, 15)];

    println!("  dt              : {:.1} ns", dt * 1e9);
    println!(
        "  Scales          : {} → {} → {} kHz  (10-12-15 iterations)",
        SCALES[0].0 * 1e-3,
        SCALES[1].0 * 1e-3,
        SCALES[2].0 * 1e-3
    );
    for &(frequency, iterations) in SCALES {
        let nt = ((t_transit * 1.2 + 3.0 / frequency) / dt).ceil() as usize;
        let half_period = 1.0 / (2.0 * frequency) * 1e6;
        println!(
            "    f₀={:.0} kHz: T/2={:.1} μs, Δt_skull=5.4 μs → {}, nt={}, {} iter",
            frequency * 1e-3,
            half_period,
            if half_period > 5.4 { "OK" } else { "WARN" },
            nt,
            iterations
        );
    }

    SCALES
}
