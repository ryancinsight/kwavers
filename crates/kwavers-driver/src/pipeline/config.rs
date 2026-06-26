//! Co-optimization tunables: the [`CoOpt`] config + the per-role dissipation model.

use crate::geom::Nm;
use crate::place::footprint::FootprintDef;

/// Tunables for the adversarial place↔route co-optimization loop.
#[derive(Debug, Clone, Copy)]
pub struct CoOpt {
    /// Maximum annealed feedback rounds after the seed floorplan has been judged once.
    /// Must be `>= 1`; `rounds = 0` is a programmer error that causes a panic in [`crate::pipeline::cooptimize`].
    /// [`CoOpt::default`] sets `4`.
    pub rounds: usize,
    /// Stop early after this many rounds with no improvement in the best score.
    pub patience: usize,
    /// Placement configuration.
    pub place: crate::place::PlaceConfig,
    /// Annealing schedule per round.
    pub anneal: crate::place::AnnealParams,
    /// HV creepage rule for the routing cost.
    pub creepage: crate::rules::CreepageRule,
    /// Physics-cost creepage / layer-affinity weights.
    pub creepage_weight: f64,
    /// Affinity weight.
    pub affinity_weight: f64,
    /// Weight of the fed-back congestion+weakness field in the next placement.
    pub feedback_weight: f64,
    /// Weight of the solved-thermal hotspot field folded into the placement feedback. Biases the
    /// placer to spread dissipative parts so the steady-state temperature field flattens.
    pub thermal_weight: f64,
    /// Weight of the EMI hotspot field (HV↔LV pad proximity) folded into the placement feedback.
    /// Biases the placer to separate the HV switching node from sensitive low-voltage control.
    pub emi_weight: f64,
    /// Weight of the **component-density field** folded into the placement feedback. Each
    /// component's courtyard *area* is deposited as a source and diffused by the same Poisson
    /// solver that produces the thermal field ([`crate::physics::thermal::solve_board`]); the
    /// resulting potential peaks where parts cluster, so biasing the next placement away from it
    /// spreads the **whole** BOM to fill the board — not just the dissipative parts the thermal
    /// field covers. This is the electrostatic density-equalisation of analytical placement
    /// (ePlace/RePlAce): area-as-charge, ∇²ψ = ρ, force = −∇ψ. It removes the need for hand-tuned
    /// `thermal_spacing` / courtyard-padding to force spread. `0.0` disables it (ablation).
    pub density_weight: f64,
    /// Per-footprint steady-state dissipation (W) sourcing the in-loop thermal-feedback field.
    /// Defaults to a coarse role estimate; a transducer driver overrides it with its derived pulser
    /// loss model ([`crate::driver::pulser_dissipation`]) so the placer actively spreads the real
    /// hot parts (the HV pulsers *and* their series damping resistors), not just the ICs.
    pub dissipation_w: fn(&FootprintDef) -> f64,
    /// When `true` (the default), distribute identical active-IC footprint groups into a regular
    /// grid before round 0. Set to `false` when the caller deliberately starts from a specific
    /// placement (e.g. in an ablation test that measures guidance over a clustered seed).
    pub seed_groups: bool,
    /// Maximum PathFinder negotiation iterations per routing round (default `40`). Increase for
    /// dense multi-IC boards where 40 iterations is insufficient to resolve congestion. Each extra
    /// iteration adds one full rip-up/re-route pass over all nets; cost is linear in net count.
    pub pathfinder_max_iter: usize,
}

/// Relative steady-state dissipation per placement role (watts), used **only** to scale the heat
/// sources of the thermal-feedback field that spreads parts during placement. These are coarse
/// role-class estimates (a placement bias, not a junction-temperature claim) — the diffusion that
/// turns them into a field is the MMS-validated Poisson solver in [`crate::physics::thermal`]. The HV driver
/// IC and its rails carry the switching loss; passives/connectors are effectively cold.
#[must_use]
pub(super) fn role_dissipation_w(role: crate::place::footprint::Role) -> f64 {
    use crate::place::footprint::Role;
    match role {
        Role::ActiveIc => 1.0,
        Role::Power => 0.5,
        Role::Decoupling | Role::Passive | Role::Connector => 0.0,
    }
}

/// Default per-footprint dissipation for the thermal-feedback field: the coarse role estimate.
#[must_use]
pub fn role_footprint_dissipation_w(fp: &FootprintDef) -> f64 {
    role_dissipation_w(fp.role)
}

impl Default for CoOpt {
    fn default() -> Self {
        CoOpt {
            rounds: 4,
            patience: 2,
            place: crate::place::PlaceConfig {
                board: (Nm::from_mm(100.0), Nm::from_mm(80.0)),
                margin: Nm::from_mm(1.0),
                thermal_spacing: Nm::from_mm(14.0),
                courtyard_clearance: Nm::from_mm(1.0),
                weights: crate::place::PlaceWeights::default(),
                isolation_axis: crate::place::Axis::X,
            },
            anneal: crate::place::AnnealParams {
                steps: 16_000,
                ..Default::default()
            },
            creepage: crate::rules::CreepageRule::holohv(),
            creepage_weight: 60.0,
            affinity_weight: 2.0,
            feedback_weight: 0.05,
            thermal_weight: 8.0,
            emi_weight: 6.0,
            // On par with the thermal field: a primary spreading signal that fills the board from
            // the whole BOM's area, so dense central clusters relax without per-design padding.
            density_weight: 10.0,
            dissipation_w: role_footprint_dissipation_w,
            seed_groups: true,
            pathfinder_max_iter: 40,
        }
    }
}
