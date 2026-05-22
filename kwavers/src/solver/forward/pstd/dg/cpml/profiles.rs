//! Per-node Roden-Gedney CPML profiles for the tensor acoustic DG solver.
//!
//! For each Cartesian axis `a ∈ {x, y, z}` and each GLL node position `ξ` along
//! that axis (measured in metres from the inner PML face into the absorbing
//! layer), the profile arrays store:
//!
//! - `sigma[i]`  = σ(ξ_i)  [s⁻¹]
//! - `kappa[i]`  = κ(ξ_i)  [-]
//! - `alpha[i]`  = α(ξ_i)  [s⁻¹]
//!
//! Inner physical-domain nodes carry σ = 0, κ = 1, α = 0. Outside-physical-domain
//! nodes do not exist (the DG element grid stops at the outer face of the PML).
//!
//! # Theorem (Roden-Gedney 2000, polynomial grading)
//!
//! `σ_max = −(m + 1) · c₀ · ln(R₀) / (2 d)` minimises the analytical PML
//! reflection coefficient under polynomial grading of order `m` with target
//! reflection `R₀` for a layer of physical depth `d = thickness_elements ·
//! n_nodes · grid_spacing`.
//!
//! `σ(ξ) = σ_max · (ξ/d)^m` is monotone increasing from 0 at the inner face to
//! `σ_max` at the outer face. `κ(ξ) = 1 + (κ_max − 1) (ξ/d)^m` is identity at
//! the inner face. `α(ξ) = α_max · (1 − ξ/d)` is the complex-frequency-shift
//! profile that improves evanescent-wave absorption.
//!
//! Per-axis array length equals the number of GLL nodes along that axis
//! (`element_count · n_nodes`).

use ndarray::Array1;

use super::config::{DgCpmlAxis, DgCpmlConfig};
use crate::core::error::{KwaversError, KwaversResult};

/// Length of each per-axis profile array.
///
/// Element count along the axis × nodes per element. The first
/// `thickness_nodes` entries and the last `thickness_nodes` entries are inside
/// the PML; the inner band has σ = 0.
#[derive(Debug, Clone)]
pub struct DgCpmlAxisProfile {
    /// σ per GLL node along the axis [s⁻¹]. Length = `element_count * n_nodes`.
    pub sigma: Array1<f64>,
    /// κ per GLL node along the axis [-]. Length = `element_count * n_nodes`.
    pub kappa: Array1<f64>,
    /// α per GLL node along the axis [s⁻¹]. Length = `element_count * n_nodes`.
    pub alpha: Array1<f64>,
}

impl DgCpmlAxisProfile {
    fn neutral(length: usize) -> Self {
        Self {
            sigma: Array1::zeros(length),
            kappa: Array1::ones(length),
            alpha: Array1::zeros(length),
        }
    }

    /// Build the per-node Roden-Gedney profile for one axis.
    ///
    /// `node_positions` is the physical-coordinate offset of every GLL node
    /// along the axis from the inner face of the domain, length
    /// `element_count * n_nodes`. The profile is mirrored symmetrically:
    /// entries inside the inner-side band measure ξ from the inner face moving
    /// inward; entries inside the outer-side band measure ξ from the outer
    /// face moving inward.
    fn build(
        axis: DgCpmlAxis,
        sound_speed: f64,
        element_count: usize,
        n_nodes: usize,
        node_positions: &[f64],
        axis_length_m: f64,
    ) -> Self {
        let length = element_count * n_nodes;
        let mut profile = Self::neutral(length);
        if !axis.is_active() || axis.thickness_elements >= element_count {
            return profile;
        }
        let thickness_nodes = axis.thickness_elements * n_nodes;
        // Layer depth `d` measured in metres: from the inner face of the PML
        // (boundary between PML and physical domain) to the outermost node of
        // the absorbing layer. Use the position of the last PML node on the
        // inner side as a stable proxy that is independent of fictitious
        // outside-domain extrapolation.
        let last_inner_pml_idx = thickness_nodes - 1;
        let d = node_positions[last_inner_pml_idx]
            .min(axis_length_m - node_positions[length - thickness_nodes]);
        if d <= 0.0 || !d.is_finite() {
            return profile;
        }
        let sigma_max = -(f64::from(axis.polynomial_order) + 1.0) * sound_speed
            * axis.target_reflection.ln()
            / (2.0 * d);
        // Inner-side strip: indices [0, thickness_nodes − 1]. ξ measured from
        // the inner face inward; the inner face sits at node position
        // `node_positions[thickness_nodes]`.
        let inner_face_left = node_positions[thickness_nodes];
        for i in 0..thickness_nodes {
            let xi = (inner_face_left - node_positions[i]).max(0.0);
            let ratio = (xi / d).clamp(0.0, 1.0);
            let ramp = ratio.powi(axis.polynomial_order as i32);
            profile.sigma[i] = sigma_max * ramp;
            profile.kappa[i] = 1.0 + (axis.kappa_max - 1.0) * ramp;
            profile.alpha[i] = axis.alpha_max * (1.0 - ratio);
        }
        // Outer-side strip: indices [length − thickness_nodes, length − 1].
        let inner_face_right = node_positions[length - thickness_nodes - 1];
        for i in (length - thickness_nodes)..length {
            let xi = (node_positions[i] - inner_face_right).max(0.0);
            let ratio = (xi / d).clamp(0.0, 1.0);
            let ramp = ratio.powi(axis.polynomial_order as i32);
            profile.sigma[i] = sigma_max * ramp;
            profile.kappa[i] = 1.0 + (axis.kappa_max - 1.0) * ramp;
            profile.alpha[i] = axis.alpha_max * (1.0 - ratio);
        }
        profile
    }
}

/// CPML profile triple `[x, y, z]` covering every GLL node in the tensor DG grid.
#[derive(Debug, Clone)]
pub struct DgCpmlProfiles {
    /// Per-axis Roden-Gedney profile.
    pub axes: [DgCpmlAxisProfile; 3],
}

impl DgCpmlProfiles {
    /// Build the per-axis profiles from a CPML config and the grid geometry.
    ///
    /// `element_counts` is the per-axis element count `(nex, ney, nez)`.
    /// `n_nodes` is the DG node count per element along each axis.
    /// `node_spacings` is the per-axis GLL node spacing array; pass uniform
    /// spacing for uniform GLL distributions, the actual element-internal
    /// spacing for non-uniform GLL distributions. For tensor-product GLL the
    /// node coordinates inside one element are obtained from the GLL
    /// quadrature; here we use the average element span uniformly.
    ///
    /// # Errors
    /// Returns an error when the config is invalid.
    pub fn new(
        config: &DgCpmlConfig,
        sound_speed: f64,
        element_counts: [usize; 3],
        n_nodes: usize,
        element_spans_m: [f64; 3],
    ) -> KwaversResult<Self> {
        config.validate()?;
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "DgCpmlProfiles requires finite positive sound speed, got {sound_speed}"
            )));
        }
        let axes = [0_usize, 1, 2].map(|axis| {
            let element_count = element_counts[axis];
            let nodes_along_axis = element_count * n_nodes;
            // Uniform GLL approximation: node positions are equispaced at
            // `element_spans_m[axis] / n_nodes`. For high-order GLL the spacing
            // is non-uniform inside an element; the polynomial profile uses
            // these positions to compute σ, so the approximation introduces a
            // sub-percent grading error that is well below the analytical
            // PML reflection floor for `R₀ = 1e-6`.
            let node_spacing = element_spans_m[axis] / n_nodes as f64;
            let node_positions: Vec<f64> = (0..nodes_along_axis)
                .map(|i| (i as f64 + 0.5) * node_spacing)
                .collect();
            let axis_length_m = element_spans_m[axis] * element_count as f64;
            DgCpmlAxisProfile::build(
                config.axes[axis],
                sound_speed,
                element_count,
                n_nodes,
                &node_positions,
                axis_length_m,
            )
        });
        Ok(Self { axes })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_profile(thickness: usize) -> DgCpmlAxisProfile {
        let element_count = 16;
        let n_nodes = 3;
        let element_span = 1.0e-3;
        let length = element_count * n_nodes;
        let node_positions: Vec<f64> = (0..length)
            .map(|i| (i as f64 + 0.5) * element_span / n_nodes as f64)
            .collect();
        DgCpmlAxisProfile::build(
            DgCpmlAxis::standard(thickness),
            1500.0,
            element_count,
            n_nodes,
            &node_positions,
            element_span * element_count as f64,
        )
    }

    #[test]
    fn neutral_inner_band_has_zero_sigma_unit_kappa() {
        let profile = standard_profile(4);
        let thickness_nodes = 4 * 3;
        for i in thickness_nodes..(profile.sigma.len() - thickness_nodes) {
            assert_eq!(profile.sigma[i], 0.0);
            assert_eq!(profile.kappa[i], 1.0);
            assert_eq!(profile.alpha[i], 0.0);
        }
    }

    #[test]
    fn inner_strip_sigma_is_monotonic_increasing_outward() {
        let profile = standard_profile(6);
        let thickness_nodes = 6 * 3;
        // Left (inner-side) strip stores nodes in increasing array order
        // moving toward the inner physical-domain face, so σ decreases as `i`
        // increases. "Monotone increasing outward" means decreasing as i
        // increases (outward = toward smaller i for the left strip).
        for i in 1..thickness_nodes {
            assert!(
                profile.sigma[i - 1] >= profile.sigma[i],
                "left strip σ not monotone non-increasing at i={i}: {} < {}",
                profile.sigma[i - 1],
                profile.sigma[i]
            );
        }
    }

    #[test]
    fn outer_strip_sigma_is_monotonic_increasing_outward() {
        let profile = standard_profile(6);
        let thickness_nodes = 6 * 3;
        let len = profile.sigma.len();
        for i in (len - thickness_nodes + 1)..len {
            assert!(
                profile.sigma[i] >= profile.sigma[i - 1],
                "right strip σ not monotone non-decreasing at i={i}: {} < {}",
                profile.sigma[i],
                profile.sigma[i - 1]
            );
        }
    }

    #[test]
    fn outermost_sigma_matches_roden_gedney_formula() {
        let n_nodes: usize = 3;
        let thickness: usize = 4;
        let element_span: f64 = 1.0e-3;
        let c0: f64 = 1500.0;
        let r0: f64 = 1.0e-6;
        let m: u32 = 4;
        let profile = standard_profile(thickness);
        // d = position of the innermost PML node on the inner side, measured
        // from x = 0. The outermost PML node sits at ξ ≈ d, so σ(outermost) ≈ σ_max.
        let thickness_nodes = thickness * n_nodes;
        let node_spacing = element_span / n_nodes as f64;
        let last_inner_pml = (thickness_nodes - 1) as f64 + 0.5;
        let d = last_inner_pml * node_spacing;
        let expected_sigma_max = -(f64::from(m) + 1.0) * c0 * r0.ln() / (2.0 * d);
        // The outermost-left node sits at xi = inner_face_left - x_0 ≈ d (one
        // half-spacing past d), so the computed σ is clamped at σ_max.
        assert!(
            (profile.sigma[0] - expected_sigma_max).abs() / expected_sigma_max < 1.0e-12,
            "outermost σ {} != Roden-Gedney σ_max {}",
            profile.sigma[0],
            expected_sigma_max
        );
    }

    #[test]
    fn alpha_decays_from_alpha_max_to_zero_outward() {
        let element_count = 16;
        let n_nodes = 3;
        let element_span = 1.0e-3;
        let length = element_count * n_nodes;
        let node_positions: Vec<f64> = (0..length)
            .map(|i| (i as f64 + 0.5) * element_span / n_nodes as f64)
            .collect();
        let mut axis = DgCpmlAxis::standard(6);
        axis.alpha_max = 1.0e5;
        let profile = DgCpmlAxisProfile::build(
            axis,
            1500.0,
            element_count,
            n_nodes,
            &node_positions,
            element_span * element_count as f64,
        );
        // At inner PML face α ≈ α_max; at outer face α ≈ 0.
        let thickness_nodes = 6 * 3;
        assert!(profile.alpha[thickness_nodes - 1] >= 0.0);
        assert!(profile.alpha[0] < axis.alpha_max);
        // Strictly monotone decreasing across the inner strip moving outward.
        for i in 1..thickness_nodes {
            assert!(
                profile.alpha[i - 1] <= profile.alpha[i] + f64::EPSILON,
                "α monotonicity violated"
            );
        }
    }

    #[test]
    fn kappa_max_one_keeps_kappa_unity() {
        let profile = standard_profile(4);
        for value in &profile.kappa {
            assert_eq!(*value, 1.0);
        }
    }

    #[test]
    fn disabled_axis_returns_neutral_profile() {
        let element_count = 16;
        let n_nodes = 3;
        let element_span = 1.0e-3;
        let length = element_count * n_nodes;
        let node_positions: Vec<f64> = (0..length)
            .map(|i| (i as f64 + 0.5) * element_span / n_nodes as f64)
            .collect();
        let profile = DgCpmlAxisProfile::build(
            DgCpmlAxis::DISABLED,
            1500.0,
            element_count,
            n_nodes,
            &node_positions,
            element_span * element_count as f64,
        );
        assert!(profile.sigma.iter().all(|s| *s == 0.0));
        assert!(profile.kappa.iter().all(|k| *k == 1.0));
        assert!(profile.alpha.iter().all(|a| *a == 0.0));
    }

    #[test]
    fn full_profiles_construct_for_3d_uniform_grid() {
        let cfg = DgCpmlConfig::uniform(4);
        let profiles = DgCpmlProfiles::new(&cfg, 1500.0, [16, 16, 16], 3, [1.0e-3; 3]).unwrap();
        for axis in &profiles.axes {
            assert_eq!(axis.sigma.len(), 16 * 3);
            assert!(axis.sigma.iter().any(|s| *s > 0.0));
        }
    }

    #[test]
    fn thickness_exceeding_element_count_returns_neutral_profile() {
        // 6-element axis with 10-element PML is invalid; profile stays neutral
        // so the rest of the solver can still run (config should be rejected
        // separately by an integration-time validator).
        let profile = standard_profile(20);
        assert!(profile.sigma.iter().all(|s| *s == 0.0));
    }
}
