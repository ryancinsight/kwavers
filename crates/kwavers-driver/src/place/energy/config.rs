//! Configuration types for the placement energy function.

use crate::geom::{GridSpec, Nm, Point};

/// The orientation of the LV↔HV isolation barrier placed by the
/// [`EnergyTerms::isolation_drift`] term. Each variant names which board axis the barrier line
/// is **perpendicular to** — `Axis::X` parks LV on the low-x edge and HV on the high-x edge
/// (a vertical centerline barrier); `Axis::Y` does the same along the y axis (a horizontal
/// centerline barrier).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Axis {
    /// Barrier runs vertically: LV on the **x-min** edge, HV on the **x-max** edge.
    #[default]
    X,
    /// Barrier runs horizontally: LV on the **y-min** edge, HV on the **y-max** edge.
    Y,
}

/// Relative weights of the placement terms. Defaults make overlap and edge violations dominant
/// (they are near-hard constraints) and tune the softer terms below them.
#[derive(Debug, Clone, Copy)]
pub struct PlaceWeights {
    /// Courtyard overlap (near-hard: must drive to ~0).
    pub overlap: f64,
    /// Courtyard crossing the edge keep-in margin (near-hard).
    pub edge: f64,
    /// Periphery preference: connectors→edge, active ICs→core.
    pub periphery: f64,
    /// Decoupling cap to its IC's nearest power pin.
    pub decoupling: f64,
    /// Resistor-like terminator proximity to the active IC pad it shares a net with.
    pub termination: f64,
    /// Net half-perimeter wirelength.
    pub hpwl: f64,
    /// Thermal spread between active ICs.
    pub thermal: f64,
    /// Connector blockage of package cooling corridors.
    pub airflow_blockage: f64,
    /// Board macro-utilization: penalises large empty board regions with no nearby component centre.
    pub utilization: f64,
    /// Similar-component orientation-axis mismatch.
    pub alignment: f64,
    /// Functional-region cohesion for components sharing non-global nets.
    pub regional: f64,
    /// Logical signal-path crossings between local net flight lines.
    pub flow_crossing: f64,
    /// Unrelated package courtyards blocking local net flight-line routing channels.
    pub channel_blockage: f64,
    /// Continuous spread penalty for multiple instances of the same active-IC footprint.
    /// Penalises `board_diagonal / (min_pairwise_distance_mm + 1)` so the incentive to spread
    /// identical ICs never fully vanishes, even once they exceed `thermal_spacing`.
    pub ic_spread: f64,
    /// LV↔HV isolation-barrier drift: penalty proportional to a component's distance from its
    /// domain's expected edge along [`PlaceConfig::isolation_axis`]. **Default `0.0`** so existing
    /// designs (nobody was tagging components until now) are unaffected until an example
    /// explicitly opts in by setting `weights.isolation_drift > 0` and tagging components via
    /// [`crate::place::component::Component::with_isolation_domain`].
    pub isolation_drift: f64,
    /// Component courtyard overlapping a fixed board mechanical-feature keepout (fiducial / mounting
    /// hole). A fiducial must sit in clear copper for pick-and-place vision and a hole must keep its
    /// barrel clear, so a part may not be floor-planned on top of one. Shares
    /// [`crate::io::mechanical_features`] with the router keepout and emission, so the placer avoids
    /// exactly the features that get drilled/printed. **Default `0.0`** (opt-in): turning it on
    /// perturbs the annealer's solution, which can shift a previously-tuned dense board into new
    /// collisions, so each board opts in (`weights.mech_keepout > 0`) once its floorplan has room —
    /// the same discipline as [`Self::isolation_drift`].
    pub mech_keepout: f64,
}

impl Default for PlaceWeights {
    fn default() -> Self {
        PlaceWeights {
            overlap: 50.0,
            edge: 50.0,
            periphery: 1.0,
            // Decoupling proximity is a PDN constraint (loop inductance ∝ distance), so weight it well
            // above wirelength so bypass caps are pulled tight to their IC's power pin — within the
            // few-mm budget that keeps them effective through the switching band (verify::decoupling).
            decoupling: 18.0,
            termination: 12.0,
            hpwl: 0.5,
            thermal: 0.5,
            airflow_blockage: 1.0,
            utilization: 0.03,
            alignment: 0.2,
            regional: 0.08,
            flow_crossing: 1.0,
            channel_blockage: 1.0,
            ic_spread: 2.0,
            // Opt-in; default 0.0 leaves all existing placements unchanged. Examples that want a
            // floorplanned LV/HV split bump this to e.g. 1.0 and tag HV components explicitly.
            isolation_drift: 0.0,
            // Opt-in; default 0.0 leaves existing tuned placements unchanged. A board with floorplan
            // room bumps this to e.g. 50.0 (near-hard, on par with overlap/edge) to push parts off its
            // fiducial/mounting-hole keepouts. Same discipline as `isolation_drift`.
            mech_keepout: 0.0,
        }
    }
}

/// A routing-congestion map fed back from a previous routing pass, used to drive the next
/// placement away from the regions the router struggled with (place↔route co-optimization).
#[derive(Debug, Clone)]
pub struct CongestionField {
    /// The grid the field is sampled on (the routing grid).
    pub spec: GridSpec,
    /// Per in-plane column congestion (from [`crate::route::Grid::congestion_field`]).
    pub per_column: Vec<f32>,
    /// Weight of the congestion term in the placement energy.
    pub weight: f64,
}

impl CongestionField {
    /// Congestion sampled at a board point (0 if out of range).
    #[must_use]
    pub fn at(&self, p: Point) -> f64 {
        let (ix, iy) = self.spec.cell_of(p);
        self.per_column[iy * self.spec.nx + ix] as f64
    }
}

/// Fixed placement context (board extent + margins + weights).
#[derive(Debug, Clone, Copy)]
pub struct PlaceConfig {
    /// Board `(width, height)`.
    pub board: (Nm, Nm),
    /// Keep-in margin from each edge (components stay inside it).
    pub margin: Nm,
    /// Target minimum spacing between active ICs (thermal).
    pub thermal_spacing: Nm,
    /// Minimum clearance kept between component courtyards (assembly DFM + keeps pads of adjacent
    /// parts from colliding on the routing grid). Courtyards are inflated by half this in the
    /// overlap term, so the placer pushes parts apart until they hold the clearance.
    pub courtyard_clearance: Nm,
    /// Term weights.
    pub weights: PlaceWeights,
    /// LV↔HV isolation-barrier orientation. The barrier line is **perpendicular** to this axis
    /// and parked at the appropriate edge of the board: `Axis::X` ⇒ LV drifts toward `x = 0`,
    /// HV toward `x = board.width`; `Axis::Y` ⇒ LV drifts toward `y = 0`, HV toward
    /// `y = board.height`. Drives [`EnergyTerms::isolation_drift`]. Default
    /// [`Axis::X`] matches the canonical "left=LV, right=HV" floorplan published in the
    /// article's isolation-barrier section.
    pub isolation_axis: Axis,
}

impl Default for PlaceConfig {
    fn default() -> Self {
        // Sensible a-priori defaults for isolated unit tests that don't bother to construct one.
        // Components are forbidden to cross the board edge (zero margin) and the isolation axis
        // is the canonical left-right split.
        Self {
            board: (Nm::from_mm(40.0), Nm::from_mm(40.0)),
            margin: Nm::from_mm(0.0),
            thermal_spacing: Nm::from_mm(0.0),
            courtyard_clearance: Nm::from_mm(0.0),
            weights: PlaceWeights::default(),
            isolation_axis: Axis::X,
        }
    }
}

/// Per-term energy breakdown (unweighted terms + the weighted total), for inspection and tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnergyTerms {
    /// Total courtyard overlap area (mm²).
    pub overlap: f64,
    /// Total edge-margin overshoot (mm).
    pub edge: f64,
    /// Periphery preference penalty (mm).
    pub periphery: f64,
    /// Decoupling proximity penalty (mm).
    pub decoupling: f64,
    /// Termination-resistor proximity penalty (mm).
    pub termination: f64,
    /// Net wirelength (mm).
    pub hpwl: f64,
    /// Thermal-spread penalty (mm).
    pub thermal: f64,
    /// Connector blockage penalty in hot-package airflow corridors.
    pub airflow_blockage: f64,
    /// Routing-congestion penalty (from fed-back congestion, if any).
    pub congestion: f64,
    /// Board macro-utilization penalty (average sample-to-nearest-component distance, mm).
    pub utilization: f64,
    /// Similar-component orientation-axis mismatch count.
    pub alignment: f64,
    /// Functional-region cohesion penalty (component-centre HPWL over shared non-global nets, mm).
    pub regional: f64,
    /// Logical signal-path crossing penalty for local nets.
    pub flow_crossing: f64,
    /// Routing-channel blockage penalty for local nets.
    pub channel_blockage: f64,
    /// Spread penalty for co-located same-footprint active ICs (`board_diag / (min_dist_mm + 1)`).
    pub ic_spread: f64,
    /// LV↔HV isolation-barrier drift penalty (mm projected onto [`PlaceConfig::isolation_axis`]).
    /// Sum of LV components' axis projection plus HV components' distance to the axis-max edge;
    /// fires only when `weights.isolation_drift > 0` and components have been tagged with
    /// [`crate::place::footprint::IsolationDomain`].
    pub isolation_drift: f64,
    /// Component-courtyard overlap with board mechanical-feature keepouts (fiducials, mounting
    /// holes), mm².
    pub mech_keepout: f64,
    /// Weighted sum the annealer minimises.
    pub total: f64,
}
