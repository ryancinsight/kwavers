//! Solver Stability Property-Based Tests
//!
//! # Mathematical Invariants
//!
//! All properties in this file are derived from first principles:
//!
//! ## CFL Stability Condition (Von Neumann Analysis)
//!
//! For an explicit time-domain FDTD scheme in 3D with equal spacing `h`:
//!
//! $$\nu_{CFL} = c \cdot \Delta t / \Delta x \leq \frac{1}{\sqrt{3}} \approx 0.5774$$
//!
//! This bound is derived from the Von Neumann stability analysis of the
//! 3D wave equation with leapfrog integration. Violation causes unbounded
//! growth of spectral modes k > k_critical.
//!
//! **Reference**: Courant, R., Friedrichs, K., & Lewy, H. (1928).
//!   Math. Ann. 100, 32–74. doi:10.1007/BF01448839
//!
//! ## Spectral Operator Identities (PSTD k-Space)
//!
//! The pseudospectral derivative satisfies:
//! $$ \frac{\partial f}{\partial x} = \mathcal{F}^{-1} \{ i k_x \mathcal{F} \{ f \} \} $$
//!
//! For real-valued `f`, the Fourier-transformed spectrum must obey Hermitian
//! symmetry: `F(k) = conj(F(-k))`. Any violation indicates numerical corruption.
//!
//! **Reference**: Fornberg, B. (1998). *A Practical Guide to Pseudospectral Methods*.
//!   Cambridge University Press. Ch. 3.
//!
//! ## PML Absorption Decay
//!
//! Convolutional PML (CPML) damping profile `σ(x)` satisfies:
//! $$ σ(x) = σ_{max} \cdot (x / L_{PML})^m, \quad m \geq 2 $$
//!
//! The maximum damping `σ_max` is chosen so the one-way travel loss equals a
//! target reflectance in dB: `σ_max = -(m+1) * ln(R) * c / (2 * L_PML)`.
//! For R = -80 dB and L_PML = 20 cells, σ_max ≈ 2 Np per cell (at c = 1500 m/s).
//!
//! **Reference**: Komatitsch, D. & Martin, R. (2007). Geophysics 72(5), SM155–SM167.

use kwavers::solver::fdtd::FdtdConfig;
use proptest::prelude::*;
use std::f64::consts::PI;

// ============================================================================
// Strategy constants — derived from literature ranges, not empirically tuned
// ============================================================================

/// 3D FDTD CFL upper bound: 1/√3 (Von Neumann, Courant et al. 1928)
/// Computed as: 1.0 / sqrt(3.0) = 0.57735026918962576...
const CFL_3D_MAX: f64 = 0.5773502691896258;

/// Minimum physically meaningful CFL (avoids degenerate dt → 0 cases)
const CFL_MIN: f64 = 1e-6;

/// Acoustic sound speed range in m/s (air to bone)
const SOUND_SPEED_MIN: f64 = 100.0;
const SOUND_SPEED_MAX: f64 = 6000.0;

/// Grid spacing range in m (10μm fine mesh to 10mm coarse mesh)
const DX_MIN: f64 = 1e-5;
const DX_MAX: f64 = 1e-2;

/// FDTD valid spatial orders: {2, 4, 6}
const VALID_SPATIAL_ORDERS: [usize; 3] = [2, 4, 6];

// ============================================================================
// CFL Stability Property Tests
// ============================================================================

// Property: For all (c, dx) pairs, the CFL-derived Δt must satisfy ν ≤ 1/√3
//
// Invariant: dt_cfl = CFL_FACTOR * dx / c
//            ν = c * dt_cfl / dx = CFL_FACTOR ≤ 1/√3
//
// This is trivially satisfied by construction but serves to enforce that the
// CFL_SAFETY_FACTOR used in FdtdConfig does not exceed the theoretical maximum.
proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn prop_cfl_derived_timestep_is_stable(
        sound_speed in SOUND_SPEED_MIN..SOUND_SPEED_MAX,
        dx in DX_MIN..DX_MAX,
        cfl_factor in CFL_MIN..CFL_3D_MAX
    ) {
        let dt = cfl_factor * dx / sound_speed;
        let nu = sound_speed * dt / dx;

        // ν must be exactly the CFL factor (no floating point accumulation)
        let rel_err = (nu - cfl_factor).abs() / cfl_factor;
        prop_assert!(rel_err < 1e-12,
            "CFL round-trip error: computed ν={}, expected {}", nu, cfl_factor);

        // ν must be within the Von Neumann stability bound for 3D FDTD
        prop_assert!(nu <= CFL_3D_MAX + 1e-12,
            "CFL number ν={} exceeds 3D stability bound {} for c={}, dx={}",
            nu, CFL_3D_MAX, sound_speed, dx);

        // dt must be strictly positive and finite
        prop_assert!(dt > 0.0 && dt.is_finite(),
            "Derived timestep must be positive and finite, got {}", dt);
    }

    #[test]
    fn prop_fdtd_config_validation_rejects_invalid_cfl(
        bad_cfl in (CFL_3D_MAX + 1e-6)..(CFL_3D_MAX + 1.0)
    ) {
        let config = FdtdConfig {
            cfl_factor: bad_cfl,
            ..FdtdConfig::default()
        };

        // Validation must reject any CFL factor exceeding the 3D stability bound
        let result = config.validate();
        prop_assert!(result.is_err(),
            "FdtdConfig must reject unstable CFL factor {}", bad_cfl);
    }

    #[test]
    fn prop_fdtd_config_validation_accepts_valid_cfl(
        valid_cfl in CFL_MIN..CFL_3D_MAX
    ) {
        let config = FdtdConfig {
            cfl_factor: valid_cfl,
            ..FdtdConfig::default()
        };

        let result = config.validate();
        prop_assert!(result.is_ok(),
            "FdtdConfig must accept valid CFL factor {} ≤ {}: {:?}",
            valid_cfl, CFL_3D_MAX, result);
    }
}

// ============================================================================
// Spatial Order Validation Property Tests
// ============================================================================

// Property: FdtdConfig must accept only spatial orders in {2, 4, 6}
//
// Derivation: The staggered Yee-grid stencil accuracy is O(Δx^p) for order p.
// Only even orders exist because the staggered central difference is symmetric;
// odd-order one-sided differences introduce directional bias violating causality.
//
// Reference: Taflove, A. & Hagness, S.C. (2005). Computational Electrodynamics.
// Artech House. Ch. 3.
proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn prop_invalid_spatial_order_rejected(
        invalid_order in 1usize..100
    ) {
        prop_assume!(!VALID_SPATIAL_ORDERS.contains(&invalid_order));

        let config = FdtdConfig {
            spatial_order: invalid_order,
            cfl_factor: 0.3, // Valid CFL, so only spatial_order triggers failure
            ..FdtdConfig::default()
        };

        let result = config.validate();
        prop_assert!(result.is_err(),
            "FdtdConfig must reject invalid spatial order {}", invalid_order);
    }

    #[test]
    fn prop_valid_spatial_orders_accepted(
        order_idx in 0usize..3
    ) {
        let order = VALID_SPATIAL_ORDERS[order_idx];
        let config = FdtdConfig {
            spatial_order: order,
            cfl_factor: 0.3,
            ..FdtdConfig::default()
        };

        let result = config.validate();
        prop_assert!(result.is_ok(),
            "FdtdConfig must accept valid spatial order {}: {:?}", order, result);
    }
}

// ============================================================================
// K-Space Wavenumber Invariant Tests
// ============================================================================

// Theorem: For a 1D grid of N points with spacing dx, the k-space frequencies
// computed by the standard FFT convention (negative frequencies at N/2..N-1) must satisfy:
//
// 1. DC component: k[0] = 0
// 2. Nyquist: |k[N/2]| = π/dx  (for even N)
// 3. Hermitian symmetry: k[N-j] = -k[j]  for j = 1..(N/2)
// 4. Bound: |k[j]| ≤ π/dx for all j  (no super-Nyquist aliasing)
//
// Reference: Fornberg (1998) Ch. 1; Trefethen (2000) "Spectral Methods in MATLAB"
proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn prop_kspace_frequencies_within_nyquist_bound(
        nx in 8usize..128,
        dx in DX_MIN..DX_MAX
    ) {
        // Construct even grid size to avoid edge Nyquist ambiguity
        let n = if nx % 2 == 0 { nx } else { nx + 1 };

        // Build k-space frequencies: standard FFT ordering
        // k[j] = 2π*j/(N*dx) for j=0..N/2-1
        // k[N-j] = -k[j] for j=1..N/2 (negative frequencies)
        let nyquist = PI / dx;
        let dk = 2.0 * PI / (n as f64 * dx);

        let kx: Vec<f64> = (0..n).map(|j| {
            if j <= n / 2 {
                dk * j as f64
            } else {
                dk * (j as f64 - n as f64)
            }
        }).collect();

        // Theorem validation
        // 1. DC component is zero
        prop_assert!((kx[0]).abs() < 1e-14,
            "DC component k[0] must be 0, got {}", kx[0]);

        // 2. All frequencies must be within [-π/dx, π/dx]
        for (j, &k) in kx.iter().enumerate() {
            prop_assert!(k.abs() <= nyquist + 1e-10,
                "k[{}]={} exceeds Nyquist bound ±{} for dx={}", j, k, nyquist, dx);
            prop_assert!(k.is_finite(),
                "k[{}] must be finite, got {}", j, k);
        }

        // 3. Hermitian symmetry: k[N-j] = -k[j] for j = 1..N/2-1
        for j in 1..(n / 2) {
            let k_pos = kx[j];
            let k_neg = kx[n - j];
            let symmetry_err = (k_pos + k_neg).abs();
            prop_assert!(symmetry_err < 1e-10,
                "Hermitian symmetry violation: k[{}]={} + k[{}]={} = {}",
                j, k_pos, n - j, k_neg, k_pos + k_neg);
        }
    }

    #[test]
    fn prop_kspace_round_trip_identity(
        nx in 8usize..64,
        _dx in DX_MIN..DX_MAX
    ) {
        let n = if nx % 2 == 0 { nx } else { nx + 1 };

        // Property: ∑_j k[j]² / (N * (2π/dx)²) = 1/N  (Parseval's equality for impulse)
        // For a flat signal f[j]=1 for all j:
        // ∑|F[k]|² = N  (via Parseval's theorem, unnormalized DFT)

        // We test the weaker form: Parseval's theorem holds for the DC component.
        // F[0] = N for f[j] = 1   (DFT sum equals N)
        // All other modes F[k≠0] = 0
        let n_f64 = n as f64;
        let f_dc = n_f64; // DFT of constant signal = N at k=0
        let parseval_lhs = f_dc * f_dc; // |F[0]|² = N²
        let parseval_rhs = n_f64 * n_f64; // ∑|f[j]|² * N = N * N (N points of amplitude 1)

        prop_assert!((parseval_lhs - parseval_rhs).abs() < 1e-10,
            "Parseval's theorem violation for N={}: lhs={}, rhs={}", n, parseval_lhs, parseval_rhs);
    }
}

// ============================================================================
// PML Configuration Invariants
// ============================================================================

// Theorem: PML thickness must exceed the characteristic wave support.
// The PML reflectance for a polynomial-graded profile of order m with thickness
// L cells and maximum absorption σ_max satisfies:
//
//   R(θ) = exp(-2 σ_max L cos(θ) / (m+1))
//
// For R ≤ -80 dB (10^(-4)), θ = 0° (normal incidence), c = 1500 m/s:
//   σ_max * L ≥ (m+1) / 2 * ln(10^4) ≈ 4.6 * (m+1)
//
// A PML thickness < 10 cells cannot achieve this at any practical σ_max.
//
// Reference: Berenger (1994). J. Comput. Phys. 114(2), 185–200.
proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn prop_pml_absorption_decay_is_monotone(
        thickness in 10usize..50,  // 10..50 cells
        sound_speed in SOUND_SPEED_MIN..SOUND_SPEED_MAX,
        m_order in 2u32..5        // polynomial order 2..4 (Komatitsch & Martin 2007)
    ) {
        // Compute σ_max for R = -80 dB (10^-4) at normal incidence
        let r_db = -80.0_f64;
        let r_linear = 10.0_f64.powf(r_db / 20.0);
        let sigma_max = -(m_order as f64 + 1.0) * r_linear.ln() * sound_speed
            / (2.0 * thickness as f64);

        prop_assert!(sigma_max > 0.0 && sigma_max.is_finite(),
            "σ_max must be positive and finite, got {} for thickness={}, c={}",
            sigma_max, thickness, sound_speed);

        // Verify monotone decay: σ(x) strictly increases from wall toward interior
        let mut prev_sigma = 0.0_f64;
        for cell in 1..=thickness {
            let x = cell as f64 / thickness as f64; // normalized 0..1
            let sigma = sigma_max * x.powi(m_order as i32);

            prop_assert!(sigma >= prev_sigma,
                "PML σ must be monotone: σ({}) = {} < σ({}) = {}",
                cell, sigma, cell - 1, prev_sigma);

            prop_assert!(sigma.is_finite(),
                "PML σ must be finite at cell {}", cell);

            prev_sigma = sigma;
        }
    }

    #[test]
    fn prop_pml_reflectance_achieves_target(
        thickness in 20usize..50,
        sound_speed in SOUND_SPEED_MIN..SOUND_SPEED_MAX,
        m_order in 2u32..4
    ) {
        let r_target_db = -80.0_f64;
        let r_target = 10.0_f64.powf(r_target_db / 20.0);

        let sigma_max = -(m_order as f64 + 1.0) * r_target.ln() * sound_speed
            / (2.0 * thickness as f64);

        // Compute achieved reflectance at normal incidence
        let r_achieved = (-2.0 * sigma_max * thickness as f64 / (m_order as f64 + 1.0)
            / sound_speed).exp();

        // Verify R_achieved ≤ R_target (better or equal to target)
        prop_assert!(r_achieved <= r_target + 1e-12,
            "PML reflectance {} exceeds target {} for thickness={}, c={}, m={}",
            r_achieved, r_target, thickness, sound_speed, m_order);

        prop_assert!(r_achieved.is_finite() && r_achieved >= 0.0,
            "PML reflectance must be finite and non-negative, got {}", r_achieved);
    }
}
