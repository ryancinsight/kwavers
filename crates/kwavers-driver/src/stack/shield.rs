//! The complete controller-plus-HV shield stack: the stack-level plan (top controller + enough
//! driver tiles), the physical board instances and local→global channel maps, and the assembled
//! electrical/mechanical manifest.

use std::fmt::Write as _;

use crate::manifest::DriverManifest;

use super::compatibility::verify_stack_pair;
use super::manifest::StackBoardManifest;
use super::plan::{optimize_stack, StackConstraints};
use super::role::StackBoardRole;

/// Stack-level plan including the top controller shield plus lower HV driver shields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShieldStackPlan {
    /// Number of controller boards.
    pub controller_boards: usize,
    /// Number of driver tiles.
    pub driver_tiles: usize,
    /// Total boards in the shield stack.
    pub total_boards: usize,
    /// Channels on the most-loaded driver tile.
    pub channels_per_tile: usize,
    /// Peak driver-tile temperature rise in kelvin.
    pub peak_driver_rise_k: f64,
    /// Total stack height including the controller board.
    pub stack_height_mm: f64,
    /// Whether the complete shield stack fits constraints.
    pub feasible: bool,
    /// Binding constraint.
    pub limiter: &'static str,
}

impl ShieldStackPlan {
    /// Generate manufacturing and operational recommendations based on the shield stack plan.
    #[must_use]
    pub fn recommendations(&self) -> Vec<String> {
        let mut recs = Vec::new();
        if self.total_boards > 1 {
            recs.push(format!(
                "Vertical stack has {} boards. Active cooling (forced air) is highly recommended to counteract restricted convection.",
                self.total_boards
            ));
            recs.push("Use shrouded, keyed stacking connectors (e.g. Samtec TFM/SFM) to prevent 150 V VPP coupling into low-voltage logic.".to_string());
            recs.push("Ensure symmetric copper plane distribution across all laminated layers to prevent reflow board warpage.".to_string());
        }
        if self.peak_driver_rise_k > 30.0 {
            recs.push("Exposed thermal pad vias on HV pulsers must be filled and plated over (VIPPO) to prevent solder wicking during reflow.".to_string());
        }
        if !self.feasible {
            match self.limiter {
                "thermal" => {
                    recs.push("Thermal budget violation. Recommendations: (1) Use exposure slots to isolate HV/LV creepage paths, (2) Relocate high-wattage damping resistors away from pulser ICs.".to_string());
                }
                "height" => {
                    recs.push("Enclosure height budget violation. Recommendations: (1) Move to an integrated 16-channel pulser to collapse the board count, (2) Tighten card pitch under active cooling.".to_string());
                }
                _ => {}
            }
        }
        recs
    }
}

/// One physical board position in the complete shield stack.
#[derive(Debug, Clone, PartialEq)]
pub struct StackBoardInstance {
    /// Zero-based physical slot from bottom to top.
    pub slot: usize,
    /// Board role.
    pub role: StackBoardRole,
    /// Source KiCad board filename.
    pub board: String,
    /// Stack connector reference designator.
    pub connector: String,
    /// Board bottom Z coordinate in millimetres.
    pub z_mm: f64,
}

/// Mapping from one local HV shield to the global stack channel namespace.
#[derive(Debug, Clone, PartialEq)]
pub struct StackTileChannelMap {
    /// Zero-based HV tile index from bottom to top.
    pub tile: usize,
    /// Physical slot containing this tile.
    pub slot: usize,
    /// Transducer connector reference designator on the tile.
    pub tx_connector: String,
    /// Local tile output nets.
    pub local_tx_nets: Vec<String>,
    /// Global stack output nets driven by this tile.
    pub global_tx_nets: Vec<String>,
}

/// Complete electrical and mechanical manifest for the controller-plus-HV shield stack.
#[derive(Debug, Clone, PartialEq)]
pub struct ShieldStackAssembly {
    /// Total stack output channels.
    pub total_channels: usize,
    /// Number of HV driver shields.
    pub driver_tiles: usize,
    /// Total physical boards, including the top controller.
    pub total_boards: usize,
    /// Board-to-board pitch in millimetres.
    pub board_pitch_mm: f64,
    /// PCB thickness in millimetres.
    pub board_thickness_mm: f64,
    /// Total stack height in millimetres.
    pub stack_height_mm: f64,
    /// Canonical stack-bus pinout shared by every mated shield connector.
    pub stack_bus_pins: Vec<String>,
    /// Physical board instances from bottom to top.
    pub boards: Vec<StackBoardInstance>,
    /// Per-HV-shield local-to-global channel maps.
    pub channel_maps: Vec<StackTileChannelMap>,
}

impl ShieldStackAssembly {
    /// Serialize the complete stack assembly as deterministic key-value text.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "format=kicad-routing-shield-stack-assembly-v1");
        let _ = writeln!(s, "total_channels={}", self.total_channels);
        let _ = writeln!(s, "driver_tiles={}", self.driver_tiles);
        let _ = writeln!(s, "total_boards={}", self.total_boards);
        let _ = writeln!(s, "board_pitch_mm={:.6}", self.board_pitch_mm);
        let _ = writeln!(s, "board_thickness_mm={:.6}", self.board_thickness_mm);
        let _ = writeln!(s, "stack_height_mm={:.6}", self.stack_height_mm);
        let _ = writeln!(s, "stack_bus_pins={}", self.stack_bus_pins.join(","));
        for board in &self.boards {
            let _ = writeln!(s, "board.{}.role={}", board.slot, board.role.as_str());
            let _ = writeln!(s, "board.{}.file={}", board.slot, board.board);
            let _ = writeln!(s, "board.{}.connector={}", board.slot, board.connector);
            let _ = writeln!(s, "board.{}.z_mm={:.6}", board.slot, board.z_mm);
        }
        for map in &self.channel_maps {
            let _ = writeln!(s, "tile.{}.slot={}", map.tile, map.slot);
            let _ = writeln!(s, "tile.{}.tx_connector={}", map.tile, map.tx_connector);
            let _ = writeln!(
                s,
                "tile.{}.local_tx_nets={}",
                map.tile,
                map.local_tx_nets.join(",")
            );
            let _ = writeln!(
                s,
                "tile.{}.global_tx_nets={}",
                map.tile,
                map.global_tx_nets.join(",")
            );
        }
        s
    }
}

/// Build the complete stack-level manifest from the controller shield, one driver-shield template,
/// and the board-backed HV manifest.
pub fn assemble_shield_stack(
    controller: &StackBoardManifest,
    driver: &StackBoardManifest,
    driver_manifest: &DriverManifest,
    plan: &ShieldStackPlan,
    total_channels: usize,
    board_pitch_mm: f64,
    board_thickness_mm: f64,
) -> Result<ShieldStackAssembly, String> {
    if plan.controller_boards != 1 {
        return Err(format!(
            "shield stack requires one controller board, found {}",
            plan.controller_boards
        ));
    }
    if !plan.feasible {
        return Err(format!("shield stack plan is infeasible: {}", plan.limiter));
    }
    let compat = verify_stack_pair(controller, driver);
    if !compat.pass {
        return Err(format!(
            "stack connector compatibility failed: {}",
            compat.mismatches.join("; ")
        ));
    }
    let local_channels = driver_manifest.channel_count();
    if local_channels != plan.channels_per_tile {
        return Err(format!(
            "driver manifest exposes {local_channels} TX nets but plan expects {} per tile",
            plan.channels_per_tile
        ));
    }
    let covered_channels = plan.driver_tiles * local_channels;
    if covered_channels != total_channels {
        return Err(format!(
            "stack channel coverage is {covered_channels}, expected {total_channels}"
        ));
    }

    let mut boards = Vec::with_capacity(plan.total_boards);
    let mut channel_maps = Vec::with_capacity(plan.driver_tiles);
    for tile in 0..plan.driver_tiles {
        boards.push(StackBoardInstance {
            slot: tile,
            role: StackBoardRole::Driver,
            board: driver.board.clone(),
            connector: driver.connector.clone(),
            z_mm: tile as f64 * board_pitch_mm,
        });
        let first = tile * local_channels;
        let global_tx_nets = (first..first + local_channels)
            .map(|channel| format!("TX_{channel}"))
            .collect();
        channel_maps.push(StackTileChannelMap {
            tile,
            slot: tile,
            tx_connector: driver_manifest.tx_connector.clone(),
            local_tx_nets: driver_manifest.tx_nets.clone(),
            global_tx_nets,
        });
    }
    boards.push(StackBoardInstance {
        slot: plan.driver_tiles,
        role: StackBoardRole::Controller,
        board: controller.board.clone(),
        connector: controller.connector.clone(),
        z_mm: plan.driver_tiles as f64 * board_pitch_mm,
    });

    Ok(ShieldStackAssembly {
        total_channels,
        driver_tiles: plan.driver_tiles,
        total_boards: plan.total_boards,
        board_pitch_mm,
        board_thickness_mm,
        stack_height_mm: plan.stack_height_mm,
        stack_bus_pins: controller.pin_nets.clone(),
        boards,
        channel_maps,
    })
}

/// Optimize the full shield stack: one top controller plus enough driver shields.
#[must_use]
pub fn optimize_shield_stack(
    total_channels: usize,
    per_channel_w: f64,
    theta_k_per_w: f64,
    c: &StackConstraints,
) -> ShieldStackPlan {
    let drivers = optimize_stack(total_channels, per_channel_w, theta_k_per_w, c);
    let total_boards = drivers.boards + 1;
    let height = total_boards as f64 * c.board_pitch_mm;
    let height_ok = height <= c.height_max_mm;
    ShieldStackPlan {
        controller_boards: 1,
        driver_tiles: drivers.boards,
        total_boards,
        channels_per_tile: drivers.channels_per_tile,
        peak_driver_rise_k: drivers.peak_rise_k,
        stack_height_mm: height,
        feasible: drivers.feasible && height_ok,
        limiter: if !height_ok {
            "height"
        } else {
            drivers.limiter
        },
    }
}
