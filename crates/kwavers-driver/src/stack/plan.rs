//! Single-board stack optimisation: the thermal/height/capacity trade that picks the **fewest
//! boards** driving a target channel count. The dimension above one board — the *number of boards*.

/// Physical constraints on the stack.
#[derive(Debug, Clone, Copy)]
pub struct StackConstraints {
    /// Maximum allowed per-board steady-state temperature rise (K).
    pub dt_max_k: f64,
    /// Enclosure height budget for the board stack (mm).
    pub height_max_mm: f64,
    /// Board-to-board pitch in the stack — the stacking-connector height (mm).
    pub board_pitch_mm: f64,
    /// Maximum channels a single tile's driver can serve.
    pub channel_cap: usize,
}

/// A chosen stack configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StackPlan {
    /// Number of stacked driver boards.
    pub boards: usize,
    /// Channels on the most-loaded tile (channels are balanced across boards).
    pub channels_per_tile: usize,
    /// Peak per-board steady-state temperature rise (K) for that load.
    pub peak_rise_k: f64,
    /// Total stack height (mm) = `boards · board_pitch`.
    pub stack_height_mm: f64,
    /// Whether every constraint is satisfied.
    pub feasible: bool,
    /// Which constraint bounds the result: `"thermal"`, `"height"`, `"capacity"`, or `"ok"`.
    pub limiter: &'static str,
}

impl StackPlan {
    /// Generate manufacturing and operational recommendations based on the plan.
    #[must_use]
    pub fn recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();
        if self.boards > 1 {
            recs.push(format!(
                "Convective cooling is restricted in a {}-board stack. Active cooling (forced air) is recommended to mitigate the 15% per-board thermal penalty.",
                self.boards
            ));
            recs.push("Ensure vertical stacking connectors (J_STACK) are shrouded and keyed to prevent reverse-insertion shorts.".to_string());
        }
        if !self.feasible {
            match self.limiter {
                "thermal" => {
                    recs.push("Thermal limit exceeded. Recommendations: (1) Reduce active duty cycle, (2) Add a bottom-side TIM/heatsink path, (3) Use integrated pulsers (e.g. STHV1600L) to lower per-channel power dissipation.".to_string());
                }
                "height" => {
                    recs.push("Stack height exceeds enclosure limit. Recommendations: (1) Use higher-density tiles to reduce board count, (2) Reduce connector board-to-board pitch, (3) Increase the enclosure height limit.".to_string());
                }
                "capacity" => {
                    recs.push("Driver capacity limit exceeded. Recommendation: Choose a driver IC with higher channel density or use multiple parallel controllers.".to_string());
                }
                _ => {}
            }
        }
        recs
    }
}

/// Per-board temperature rise (K) for `channels` channels at `per_channel_w` each through a
/// board-to-ambient thermal resistance `theta_k_per_w` with a vertical stacking penalty:
/// `ΔT = n · p_ch · θ · (1.0 + 0.15 · (N - 1))`.
#[must_use]
pub fn board_rise_k(
    channels: usize,
    per_channel_w: f64,
    theta_k_per_w: f64,
    total_boards: usize,
) -> f64 {
    let penalty = 1.0 + 0.15 * (total_boards.saturating_sub(1) as f64);
    channels as f64 * per_channel_w * theta_k_per_w * penalty
}

/// Choose the **fewest boards** that drive `total_channels` within the thermal, height, and driver
/// constraints. Channels are balanced (`ceil(total/boards)` on the busiest tile). Scans board count
/// upward from the capacity floor and returns the first thermally-and-dimensionally feasible plan;
/// if none fits, returns the best attempt with `feasible = false` and the binding `limiter`.
#[must_use]
pub fn optimize_stack(
    total_channels: usize,
    per_channel_w: f64,
    theta_k_per_w: f64,
    c: &StackConstraints,
) -> StackPlan {
    // SAFETY contract: callers must supply a non-empty design; return a clearly-infeasible plan
    // rather than panicking so library users receive a typed result they can inspect and report.
    if total_channels == 0 || c.channel_cap == 0 {
        return StackPlan {
            boards: 0,
            channels_per_tile: 0,
            peak_rise_k: 0.0,
            stack_height_mm: 0.0,
            feasible: false,
            limiter: "zero channels or zero channel capacity",
        };
    }
    let floor = total_channels.div_ceil(c.channel_cap); // fewest boards the driver allows
    let max_boards = total_channels; // 1 channel/board is the thermal extreme
    let plan_for = |boards: usize| {
        let cpt = total_channels.div_ceil(boards);
        let rise = board_rise_k(cpt, per_channel_w, theta_k_per_w, boards);
        let height = boards as f64 * c.board_pitch_mm;
        (cpt, rise, height)
    };

    let mut best: Option<StackPlan> = None;
    for boards in floor..=max_boards {
        let (cpt, rise, height) = plan_for(boards);
        let thermal_ok = rise <= c.dt_max_k;
        let height_ok = height <= c.height_max_mm;
        let plan = StackPlan {
            boards,
            channels_per_tile: cpt,
            peak_rise_k: rise,
            stack_height_mm: height,
            feasible: thermal_ok && height_ok,
            limiter: if !height_ok {
                "height"
            } else if !thermal_ok {
                "thermal"
            } else {
                "ok"
            },
        };
        if plan.feasible {
            return plan; // fewest boards that satisfy everything
        }
        // Track the least-bad attempt: prefer lower temperature rise (thermal is the safety limit).
        if best.as_ref().map(|b| rise < b.peak_rise_k).unwrap_or(true) {
            best = Some(plan);
        }
        // Once height alone is the blocker (adding boards only makes height worse), stop.
        if height > c.height_max_mm && rise <= c.dt_max_k {
            break;
        }
    }
    let mut p = best.expect("scan is non-empty");
    // If the driver capacity alone forces the floor and that floor is over temperature, name it.
    if p.boards == floor
        && p.peak_rise_k > c.dt_max_k
        && floor == total_channels.div_ceil(c.channel_cap)
    {
        p.limiter = "capacity";
    }
    p
}
