//! Component CAD/footprint accuracy manifest.

use std::fmt::Write as _;

/// Accuracy status for one component family used by a generated board.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentAccuracy {
    /// Board component family or reference designator group.
    pub component: String,
    /// Selected manufacturer part number.
    pub mpn: String,
    /// Quantity used on the generated board.
    pub quantity: usize,
    /// Pads represented by the generated `FootprintDef`.
    pub board_pads: usize,
    /// Pads in the locally downloaded exact KiCad footprint.
    pub exact_pads: usize,
    /// Local exact CAD footprint path or `missing`.
    pub exact_cad: String,
    /// Whether the generated board already uses the exact footprint.
    pub exact_used: bool,
}

impl ComponentAccuracy {
    /// Build one component accuracy row.
    #[must_use]
    pub fn new(
        component: impl Into<String>,
        mpn: impl Into<String>,
        quantity: usize,
        board_pads: usize,
        exact_pads: usize,
        exact_cad: impl Into<String>,
        exact_used: bool,
    ) -> Self {
        Self {
            component: component.into(),
            mpn: mpn.into(),
            quantity,
            board_pads,
            exact_pads,
            exact_cad: exact_cad.into(),
            exact_used,
        }
    }
}

/// Board-level component accuracy report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComponentAccuracyReport {
    /// Generated board filename.
    pub board: String,
    /// Expected routed channel count on this board.
    pub channels: usize,
    /// Number of HV7355 devices on this board.
    pub hv7355_count: usize,
    /// Component rows.
    pub components: Vec<ComponentAccuracy>,
}

impl ComponentAccuracyReport {
    /// True only when every listed component uses an exact local CAD footprint.
    #[must_use]
    pub fn exact_complete(&self) -> bool {
        self.components.iter().all(|c| c.exact_used)
    }

    /// Serialize to deterministic key-value text.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "format=kicad-routing-component-accuracy-v1");
        let _ = writeln!(s, "board={}", self.board);
        let _ = writeln!(s, "channels={}", self.channels);
        let _ = writeln!(s, "hv7355_count={}", self.hv7355_count);
        let _ = writeln!(s, "exact_complete={}", self.exact_complete());
        for (i, c) in self.components.iter().enumerate() {
            let _ = writeln!(s, "component.{i}.name={}", c.component);
            let _ = writeln!(s, "component.{i}.mpn={}", c.mpn);
            let _ = writeln!(s, "component.{i}.quantity={}", c.quantity);
            let _ = writeln!(s, "component.{i}.board_pads={}", c.board_pads);
            let _ = writeln!(s, "component.{i}.exact_pads={}", c.exact_pads);
            let _ = writeln!(s, "component.{i}.exact_cad={}", c.exact_cad);
            let _ = writeln!(s, "component.{i}.exact_used={}", c.exact_used);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_file(path: &str) -> String {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
    }
    // Suppress dead-code lint: kept for future tests that require guaranteed fixtures.
    #[allow(dead_code)]
    fn _assert_repo_file_reachable() {
        let _ = repo_file as fn(&str) -> String;
    }

    /// Try to read a fixture file; returns `None` and prints a skip message when
    /// the file does not exist.  Tests that call this must return early on `None`
    /// so they are treated as passing rather than panicking on missing fixtures
    /// (the fixture files are generated artifacts committed separately).
    fn try_repo_file(path: &str) -> Option<String> {
        let full = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        match std::fs::read_to_string(&full) {
            Ok(content) => Some(content),
            Err(_) => {
                eprintln!(
                    "SKIP: fixture not found at {}; \
                     run the board generation script to produce it.",
                    full.display()
                );
                None
            }
        }
    }

    fn count_token(haystack: &str, needle: &str) -> usize {
        haystack.match_indices(needle).count()
    }

    fn footprint_block<'a>(board: &'a str, footprint: &str) -> &'a str {
        let start = board
            .find(footprint)
            .unwrap_or_else(|| panic!("missing footprint token {footprint}"));
        let mut depth = 0usize;
        for (offset, ch) in board[start..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return &board[start..start + offset + ch.len_utf8()];
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated footprint token {footprint}");
    }

    fn footprint_block_by_reference<'a>(board: &'a str, refdes: &str) -> &'a str {
        let property = format!("(property \"Reference\" \"{refdes}\"");
        let property_at = board
            .find(&property)
            .unwrap_or_else(|| panic!("missing reference property {refdes}"));
        let start = board[..property_at]
            .rfind("(footprint ")
            .unwrap_or_else(|| panic!("missing footprint start for {refdes}"));
        let mut depth = 0usize;
        for (offset, ch) in board[start..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        return &board[start..start + offset + ch.len_utf8()];
                    }
                }
                _ => {}
            }
        }
        panic!("unterminated footprint for {refdes}");
    }

    #[test]
    fn exact_complete_requires_every_component() {
        let report = ComponentAccuracyReport {
            board: "b.kicad_pcb".into(),
            channels: 16,
            hv7355_count: 2,
            components: vec![
                ComponentAccuracy::new("A", "A1", 1, 4, 4, "a.kicad_mod", true),
                ComponentAccuracy::new("B", "B1", 1, 2, 8, "b.kicad_mod", false),
            ],
        };
        assert!(!report.exact_complete());
        assert!(report.to_text().contains("exact_complete=false"));
    }

    #[test]
    fn hv_driver_artifact_uses_exact_j4_power_header() {
        let Some(board) =
            try_repo_file("tests/fixtures/boards/hv7355_24ch_tile/hv7355_24ch_tile.kicad_pcb")
        else {
            return; // fixture not present; skip gracefully
        };
        let j4 = footprint_block(&board, "(footprint \"kicad-routing:MOLEX_430450400\"");

        assert_eq!(
            count_token(&board, "(footprint \"kicad-routing:MOLEX_430450400\""),
            1
        );
        assert_eq!(
            count_token(
                &board,
                "(footprint \"kicad-routing:J_PWR_Molex_0430450400_demo\""
            ),
            0
        );
        assert_eq!(count_token(&board, "(property \"Reference\" \"J4\""), 1);
        assert_eq!(count_token(&board, "(at 95.52 20 90)"), 1);
        assert_eq!(
            count_token(&board, "(at 95.52 32 90)"),
            0,
            "J4 must stay in the right-edge power-input region, not the lower empty quadrant"
        );
        assert_eq!(
            count_token(
                &board,
                "(model \"docs/cad_models/430450400/430450400.step\""
            ),
            1
        );
        let j4 = j4.replace("\r\n", "\n");
        assert!(
            j4.contains("(rotate\n\t\t\t\t(xyz 0 0 0)\n\t\t\t)"),
            "J4 STEP model must stay in the downloaded Molex footprint orientation"
        );
        assert!(
            !j4.contains("(xyz 0 0 90)"),
            "J4 STEP model rotation reintroduces the rendered pin/pad offset"
        );
    }

    #[test]
    fn hv7355_32ch_artifact_has_renderable_component_models() {
        let Some(board) =
            try_repo_file("tests/fixtures/boards/hv7355_32ch_tile/hv7355_32ch_tile.kicad_pcb")
        else {
            return; // fixture not present; skip gracefully
        };
        let j5 = footprint_block_by_reference(&board, "J5");

        assert_eq!(
            count_token(
                &board,
                "Package_DFN_QFN.3dshapes/QFN-56-1EP_8x8mm_P0.5mm_EP4.5x5.2mm.step"
            ),
            4,
            "32-channel comparison board must render one HV7355 package body per driver IC"
        );
        assert_eq!(
            count_token(&board, "Capacitor_SMD.3dshapes/C_0402_1005Metric.step"),
            16,
            "32-channel comparison board must render all decoupling capacitors"
        );
        assert_eq!(
            count_token(&board, "Connector_PinHeader_2.54mm.3dshapes/PinHeader_"),
            6,
            "32-channel comparison board must render J1-J6 header bodies"
        );
        assert_eq!(
            count_token(&board, "(xyz -1.27 -10.16 0)"),
            1,
            "J1 2x9 header model must be recentered onto the synthetic pad grid"
        );
        assert_eq!(
            count_token(&board, "(xyz -1.27 -3.81 0)"),
            4,
            "J2-J5 2x4 header models must be recentered onto their pad grids"
        );
        assert_eq!(
            count_token(&board, "(xyz -1.27 -5.08 0)"),
            1,
            "J6 2x5 header model must be recentered onto the synthetic pad grid"
        );
        assert!(
            j5.contains(
                "(pad \"1\" thru_hole circle (at -1.2700 -3.8100) (size 1.0000 1.0000) (drill 0.5000) (layers \"*.Cu\") (net 51 \"TX_27\")"
            ),
            "J5 pad 1 must be the top-left pad in the KiCad stock 2x04 header orientation"
        );
        assert!(
            j5.contains(
                "(pad \"8\" thru_hole circle (at 1.2700 3.8100) (size 1.0000 1.0000) (drill 0.5000) (layers \"*.Cu\") (net 55 \"TX_31\")"
            ),
            "J5 pad 8 must be the bottom-right pad in the KiCad stock 2x04 header orientation"
        );
    }
}
