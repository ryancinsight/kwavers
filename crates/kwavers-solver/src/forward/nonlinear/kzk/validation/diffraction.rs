//! Diffraction validation tests for KZK equation implementation.

#[cfg(test)]
mod tests {
    use super::super::super::*;
    use crate::forward::nonlinear::kzk::constants::{DEFAULT_BEAM_WAIST, DEFAULT_FREQUENCY};
    use leto::Array2;
    use std::f64::consts::PI;

    /// Transverse grid points per axis.
    const GRID: usize = 64;
    /// Transverse spacing (m); GRID × DX = 32 mm of aperture.
    const DX: f64 = 0.5e-3;
    /// Axial steps taken to reach the Rayleigh distance.
    const STEPS: usize = 10;

    /// Relative tolerance on the measured 1/e² beam radius.
    ///
    /// ## Derivation
    ///
    /// With `w = √2·w₀ = 7.0711 mm` and `h = DX = 0.5 mm`:
    ///
    /// 1. **Threshold interpolation.**  The crossing of `E(r) ∝ exp(−2r²/w²)`
    ///    with `E₀/e²` is located by linear interpolation between adjacent
    ///    nodes.  The root displacement of a linear interpolant is bounded by
    ///    `(h²/8)·|E″/E′|`, and at `r = w` that ratio is `3/w`, giving
    ///    `Δw/w ≤ 3h²/(8w²) = 1.9e−3`.  This term dominates.
    /// 2. **Periodic wraparound.**  The DFT propagates the periodised source.
    ///    At the measurement radius the nearest image contributes an amplitude
    ///    `exp(−((L−w)/w)²) = exp(−12.4) = 4.0e−6` against a local amplitude
    ///    `e⁻¹`, i.e. a `2.2e−5` intensity perturbation.  With
    ///    `|d ln E / d ln r| = 4` at `r = w` this moves the radius by
    ///    `5.5e−6`.
    /// 3. **Source aliasing.**  The sampled Gaussian's spectrum at the Nyquist
    ///    wavenumber is `exp(−k²w₀²/4) = exp(−247)`; unrepresentable.
    /// 4. **Axial discretisation.**  None.  Diffraction is the only active
    ///    operator, and the spectral propagator composes exactly
    ///    (`D(Δz/2)·D(Δz/2) = D(Δz)`), so the field at `STEPS·Δz` is the exact
    ///    parabolic solution regardless of `STEPS`.
    /// 5. **Round-off.**  40 length-4096 2-D transforms at `3·log₂(N)·ε`
    ///    (Higham 2002 §24.1) give `3.2e−13`.
    ///
    /// Sum ≈ `1.9e−3`.  The bound below carries a factor-5 margin over that,
    /// covering the second-order term in the interpolation-error constant.
    const RADIUS_TOLERANCE: f64 = 1.0e-2;

    /// Relative tolerance on the on-axis intensity ratio.
    ///
    /// ## Derivation
    ///
    /// The on-axis value has no interpolation error.  The residual is the
    /// periodisation of the truncated Gaussian: the source is cut at
    /// `L/2 = 16 mm`, where its amplitude is `exp(−(16/5)²) = 3.6e−5`.  The
    /// propagator is unitary, so that residual bounds the on-axis perturbation
    /// at `3.6e−5`.  Round-off adds `3.2e−13`.  The bound below carries a
    /// factor-25 margin.
    const ON_AXIS_TOLERANCE: f64 = 1.0e-3;

    /// Solver configuration reaching exactly one Rayleigh distance in
    /// [`STEPS`] axial steps, with diffraction as the only active operator.
    ///
    /// `nz` is set to `2·STEPS` rather than `STEPS`: [`validate_config`] caps
    /// the parabolic half-angle `atan(nx·dx / (2·nz·dz))` at 0.3 rad, and a
    /// 32 mm aperture over a single Rayleigh distance sits just above that
    /// limit.  Declaring twice the axial extent and propagating half of it
    /// keeps the angle at 0.16 rad without changing the measured physics.
    fn rayleigh_config() -> (KZKConfig, f64) {
        let c0 = KZKConfig::default().c0;
        let wavelength = c0 / DEFAULT_FREQUENCY;
        // Rayleigh range for the amplitude convention exp(−r²/w₀²):
        //   z_R = k·w₀²/2 = π·w₀²/λ
        let rayleigh_distance = PI * DEFAULT_BEAM_WAIST * DEFAULT_BEAM_WAIST / wavelength;
        let dz = rayleigh_distance / STEPS as f64;

        let config = KZKConfig {
            nx: GRID,
            ny: GRID,
            nz: 2 * STEPS,
            // The parabolic propagator acts identically on every retarded-time
            // slice, so the transverse profile is τ-independent and nt only
            // has to provide a non-null slice.  dt satisfies the CFL cap
            // c₀·dt/dz ≤ 0.5 with three orders of magnitude to spare.
            nt: 4,
            dx: DX,
            dz,
            dt: 1e-7,
            include_nonlinearity: false,
            include_absorption: false,
            include_diffraction: true,
            ..Default::default()
        };
        (config, rayleigh_distance)
    }

    /// Circular Gaussian source `A(r) = exp(−r²/w₀²)` centred on the grid.
    fn gaussian_source(config: &KZKConfig) -> Array2<f64> {
        let mut source = Array2::zeros((config.nx, config.ny));
        for j in 0..config.ny {
            for i in 0..config.nx {
                let x = (i as f64 - config.nx as f64 / 2.0) * config.dx;
                let y = (j as f64 - config.ny as f64 / 2.0) * config.dx;
                let r2 = x.mul_add(x, y * y);
                source[[i, j]] = (-r2 / (DEFAULT_BEAM_WAIST * DEFAULT_BEAM_WAIST)).exp();
            }
        }
        source
    }

    /// Transverse envelope intensity `Σ_τ |p(x, y, τ)|²`.
    ///
    /// The internal field is `p(x, y, τ) = sin(ω₀τ)·U(x, y)`, so the modulus
    /// carries the beam envelope `|U|` at every retarded time.  The real-part
    /// observables (`get_intensity`, `get_peak_pressure`) instead report
    /// `Re[U]`, whose radial phase `arg U(r) = r²/(2w₀²) − π/4` modulates the
    /// profile and shifts the apparent 1/e² radius by +11% at the Rayleigh
    /// distance.  The width oracle therefore reads the envelope directly; the
    /// second assertion in the test below covers the real-part path with its
    /// own closed form.
    fn envelope_intensity(solver: &KZKSolver, config: &KZKConfig) -> Array2<f64> {
        let mut envelope = Array2::zeros((config.nx, config.ny));
        for i in 0..config.nx {
            for j in 0..config.ny {
                let mut sum = 0.0_f64;
                for t in 0..config.nt {
                    let p = solver.pressure[[i, j, t]];
                    sum = p.re.mul_add(p.re, p.im.mul_add(p.im, sum));
                }
                envelope[[i, j]] = sum;
            }
        }
        envelope
    }

    /// Radius (m) at which `envelope` falls to `1/e²` of its on-axis value,
    /// scanning outward along +x from the beam centre and interpolating the
    /// crossing linearly between the bracketing nodes.
    fn one_over_e_squared_radius(envelope: &Array2<f64>, config: &KZKConfig) -> f64 {
        let centre_i = config.nx / 2;
        let centre_j = config.ny / 2;
        let peak = envelope[[centre_i, centre_j]];
        let threshold = peak / (std::f64::consts::E * std::f64::consts::E);

        for i in centre_i + 1..config.nx {
            let current = envelope[[i, centre_j]];
            if current < threshold {
                let previous = envelope[[i - 1, centre_j]];
                let fraction = (threshold - current) / (previous - current);
                return ((i - centre_i) as f64 - fraction) * config.dx;
            }
        }
        panic!(
            "beam edge not found within the {} mm half-aperture: the beam \
             fills the grid and the width oracle cannot be evaluated",
            (config.nx - centre_i) as f64 * config.dx * 1000.0
        );
    }

    /// Gaussian-beam diffraction oracle at the Rayleigh distance.
    ///
    /// ## Analytical solution
    ///
    /// The KZK diffraction sub-step is the parabolic equation
    /// `∂U/∂z = (i/2k)∇⊥²U`.  For the source `U(r, 0) = exp(−r²/w₀²)` its exact
    /// solution is
    ///
    /// ```text
    /// U(r, z) = 1/(1 + iζ) · exp(−r²/(w₀²(1 + iζ))),    ζ = z/z_R
    /// ```
    ///
    /// with `z_R = k·w₀²/2 = π·w₀²/λ`.  Two consequences are asserted:
    ///
    /// - `|U(r, z)| = (1 + ζ²)^(−1/2)·exp(−r²/w(z)²)` with
    ///   `w(z) = w₀·√(1 + ζ²)`, so the 1/e² intensity radius at `ζ = 1` is
    ///   exactly `√2·w₀`.
    /// - `Re[U(0, z)] = Re[1/(1 + iζ)] = 1/(1 + ζ²)`, so the on-axis
    ///   real-part intensity that [`KZKSolver::get_intensity`] reports falls
    ///   by exactly `1/(1 + ζ²)² = 1/4` at `ζ = 1`.
    ///
    /// ## Workload
    ///
    /// 64² transverse points, 4 retarded-time slices, 10 axial steps.  The
    /// spectral propagator composes exactly, so a coarse axial step reaches the
    /// same field as a fine one; the reduction is in the instrument, not in the
    /// regime, and the test runs inside the default per-test budget.
    ///
    /// ## References
    ///
    /// - Siegman AE (1986). Lasers. University Science Books, §17.1
    ///   (Gaussian-beam q-parameter propagation).
    /// - Lee Y-S, Hamilton MF (1995). J. Acoust. Soc. Am. 97(2), 906–917.
    ///   DOI: 10.1121/1.412000
    #[test]
    fn gaussian_beam_spreads_to_root_two_waist_at_the_rayleigh_distance() {
        let (config, rayleigh_distance) = rayleigh_config();
        let mut solver = KZKSolver::new(config.clone()).expect("Rayleigh-distance solver");
        solver.set_source(gaussian_source(&config), DEFAULT_FREQUENCY);

        let source_radius =
            one_over_e_squared_radius(&envelope_intensity(&solver, &config), &config);
        let source_on_axis = solver.get_intensity()[[config.nx / 2, config.ny / 2]];

        for _ in 0..STEPS {
            solver.step();
        }

        // The propagated distance must be one Rayleigh range.
        let propagated = STEPS as f64 * config.dz;
        assert!(
            (propagated - rayleigh_distance).abs() / rayleigh_distance < 1e-12,
            "workload must land on the Rayleigh distance: propagated \
             {:.6} mm vs z_R = {:.6} mm",
            propagated * 1000.0,
            rayleigh_distance * 1000.0
        );

        // Oracle 1: envelope radius w(z_R) = √2·w₀.
        let measured_radius =
            one_over_e_squared_radius(&envelope_intensity(&solver, &config), &config);
        let expected_radius = DEFAULT_BEAM_WAIST * std::f64::consts::SQRT_2;
        let radius_error = (measured_radius - expected_radius).abs() / expected_radius;
        assert!(
            radius_error < RADIUS_TOLERANCE,
            "Gaussian-beam spreading: expected w(z_R) = {:.4} mm, got {:.4} mm \
             (relative error {radius_error:.3e} exceeds the derived bound \
             {RADIUS_TOLERANCE:.1e}); source radius was {:.4} mm",
            expected_radius * 1000.0,
            measured_radius * 1000.0,
            source_radius * 1000.0
        );

        // The source must actually have started at w₀, else the ratio above
        // could be met by a beam that was already too wide.
        let source_error = (source_radius - DEFAULT_BEAM_WAIST).abs() / DEFAULT_BEAM_WAIST;
        assert!(
            source_error < RADIUS_TOLERANCE,
            "source plane must start at w₀ = {:.4} mm, got {:.4} mm \
             (relative error {source_error:.3e})",
            DEFAULT_BEAM_WAIST * 1000.0,
            source_radius * 1000.0
        );

        // Oracle 2: on-axis real-part intensity ratio 1/(1 + ζ²)² = 1/4.
        let on_axis = solver.get_intensity()[[config.nx / 2, config.ny / 2]];
        let measured_ratio = on_axis / source_on_axis;
        let expected_ratio = 0.25;
        let ratio_error = (measured_ratio - expected_ratio).abs() / expected_ratio;
        assert!(
            ratio_error < ON_AXIS_TOLERANCE,
            "on-axis intensity at z_R: expected I/I₀ = 1/(1 + ζ²)² = {expected_ratio}, \
             got {measured_ratio:.9} (relative error {ratio_error:.3e} exceeds the \
             derived bound {ON_AXIS_TOLERANCE:.1e})"
        );
    }
}
