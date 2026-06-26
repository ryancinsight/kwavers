//! The stack-connector observation extracted from one generated shield board — its geometry and
//! canonicalised pin nets — serialised as deterministic key-value text for inter-board comparison.

use std::fmt::Write as _;
use std::path::Path;

use crate::board::Board;
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;

use super::role::StackBoardRole;
use super::util::{board_height_mm, board_width_mm, canonical_stack_net};

/// Stack connector observation extracted from one generated shield board.
#[derive(Debug, Clone, PartialEq)]
pub struct StackBoardManifest {
    /// Board filename.
    pub board: String,
    /// Board role in the shield stack.
    pub role: StackBoardRole,
    /// Stack connector reference designator.
    pub connector: String,
    /// Board width in millimetres.
    pub board_w_mm: f64,
    /// Board height in millimetres.
    pub board_h_mm: f64,
    /// Connector centre X coordinate in millimetres.
    pub connector_x_mm: f64,
    /// Connector centre Y coordinate in millimetres.
    pub connector_y_mm: f64,
    /// Connector rotation in degrees.
    pub connector_rot_deg: f64,
    /// Connector pin nets in pad order, canonicalized for inter-board comparison.
    pub pin_nets: Vec<String>,
}

impl StackBoardManifest {
    /// Serialize the stack board manifest as deterministic key-value text.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "format=kicad-routing-stack-board-v1");
        let _ = writeln!(s, "board={}", self.board);
        let _ = writeln!(s, "role={}", self.role.as_str());
        let _ = writeln!(s, "connector={}", self.connector);
        let _ = writeln!(s, "board_w_mm={:.6}", self.board_w_mm);
        let _ = writeln!(s, "board_h_mm={:.6}", self.board_h_mm);
        let _ = writeln!(s, "connector_x_mm={:.6}", self.connector_x_mm);
        let _ = writeln!(s, "connector_y_mm={:.6}", self.connector_y_mm);
        let _ = writeln!(s, "connector_rot_deg={:.6}", self.connector_rot_deg);
        let _ = writeln!(s, "pin_nets={}", self.pin_nets.join(","));
        s
    }

    /// Parse stack board manifest text.
    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut map = std::collections::BTreeMap::new();
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let Some((k, v)) = line.split_once('=') else {
                return Err(format!("stack manifest line has no '=': {line}"));
            };
            map.insert(k.trim(), v.trim());
        }
        if map.get("format").copied() != Some("kicad-routing-stack-board-v1") {
            return Err("stack board manifest format mismatch".into());
        }
        let get = |k: &str| {
            map.get(k)
                .copied()
                .ok_or_else(|| format!("stack manifest missing {k}"))
        };
        let parse = |k: &str| -> Result<f64, String> {
            get(k)?
                .parse::<f64>()
                .map_err(|e| format!("stack manifest {k} parse failed: {e}"))
        };
        let pin_nets = get("pin_nets")?
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        Ok(StackBoardManifest {
            board: get("board")?.to_string(),
            role: StackBoardRole::try_from(get("role")?)?,
            connector: get("connector")?.to_string(),
            board_w_mm: parse("board_w_mm")?,
            board_h_mm: parse("board_h_mm")?,
            connector_x_mm: parse("connector_x_mm")?,
            connector_y_mm: parse("connector_y_mm")?,
            connector_rot_deg: parse("connector_rot_deg")?,
            pin_nets,
        })
    }

    /// Read a stack board manifest from disk.
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
        Self::from_text(&text)
    }
}

/// Build a stack-board manifest from a generated board and its placed connector.
pub fn stack_board_manifest_from_board(
    board_path: &Path,
    role: StackBoardRole,
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
    connector: &str,
) -> Result<StackBoardManifest, String> {
    let c = comps
        .iter()
        .find(|c| c.refdes == connector)
        .ok_or_else(|| format!("missing stack connector {connector}"))?;
    let fp = &lib[c.fp];
    if c.nets.len() != fp.pads.len() {
        return Err(format!(
            "{connector} net count {} does not match pad count {}",
            c.nets.len(),
            fp.pads.len()
        ));
    }
    let pin_nets = c
        .nets
        .iter()
        .map(|n| {
            n.map(|id| canonical_stack_net(&board.nets[id.0 as usize].name).to_string())
                .unwrap_or_else(|| "NC".to_string())
        })
        .collect();
    Ok(StackBoardManifest {
        board: board_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("board.kicad_pcb")
            .to_string(),
        role,
        connector: connector.to_string(),
        board_w_mm: board_width_mm(board),
        board_h_mm: board_height_mm(board),
        connector_x_mm: c.placement.pos.x.to_mm(),
        connector_y_mm: c.placement.pos.y.to_mm(),
        connector_rot_deg: c.placement.rot.degrees(),
        pin_nets,
    })
}
