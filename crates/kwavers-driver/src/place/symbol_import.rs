//! Import pin name↔number maps from KiCad `.kicad_sym` symbol files.
//!
//! The `.kicad_mod` footprint importer gives each pad a **number** and an exact position; the schematic
//! **symbol** gives the number↔**function name** map (`"1" → "IN0"`, `"33" → "VPP"`). Together they let
//! the optimiser wire a net to a pin by *function* on the genuine manufacturer footprint — the last
//! piece needed to build a fabrication-real netlist instead of a positional abstraction.
//!
//! Parser: a focused scanner for the `(name "…")` / `(number "…")` token pairs that appear (in that
//! order) inside each `(pin …)` of the symbol. No external dependency.
//!
//! Phase 2c: the inline `mod tests { ... }` block moved to `crate::place::tests` (the consolidated
//! slice-wide test surface). `quoted_events` stays private — the byte-tracking pinning tests at
//! `crate::place::tests` exercise it indirectly through `import_symbol_pinmap` (and through the
//! `pub(super)` parser items at `crate::place::footprint_import` for the dedicated sexpr contracts).
//!
//! Evidence tier: value-semantic tests against the committed vendor symbols (HV7355, ISO7740) assert
//! the exact pin count and known name↔number pairs.

/// Pin name↔number map for one symbol (a part's function pinout).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PinMap {
    /// `(function name, pad number)` pairs in file order. A name may repeat (e.g. several `GND` pins).
    pub pins: Vec<(String, String)>,
}

impl PinMap {
    /// The first pad number carrying function `name`, if any.
    #[must_use]
    pub fn number_of(&self, name: &str) -> Option<&str> {
        self.pins
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, num)| num.as_str())
    }

    /// Every pad number carrying function `name` (a power/ground net spans many pins).
    #[must_use]
    pub fn numbers_of(&self, name: &str) -> Vec<&str> {
        self.pins
            .iter()
            .filter(|(n, _)| n == name)
            .map(|(_, num)| num.as_str())
            .collect()
    }

    /// The function name on pad `number`, if any.
    #[must_use]
    pub fn name_of(&self, number: &str) -> Option<&str> {
        self.pins
            .iter()
            .find(|(_, num)| num == number)
            .map(|(n, _)| n.as_str())
    }

    /// Number of pins.
    #[must_use]
    pub fn len(&self) -> usize {
        self.pins.len()
    }

    /// Whether the map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.pins.is_empty()
    }
}

/// Collect the quoted string immediately following each occurrence of `pat` (which ends in `"`),
/// tagged with its byte position so name/number events can be interleaved in file order.
///
/// **Phase 1d polish**: on an unclosed-quote the function returns
/// [`crate::error::Manifest::Parse`] with the byte offset pointing at the opening `"`,
/// so a contributor trimming a half-finished vendor symbol gets a typed diagnostic in
/// `cargo` output rather than a silently-dropped event (the previous behaviour fell
/// through to `no_pins` which masked the real bug). The opening `"` is at byte
/// `qstart - 1` (one byte before the payload start `qstart = from + idx + plen`); the
/// error envelope carries that byte as `offset`, matching the byte-tracking convention
/// `parse_sexpr` set in Phase 1c polish.
fn quoted_events(
    text: &str,
    pat: &str,
    tag: bool,
) -> Result<Vec<(usize, bool, String)>, crate::Error> {
    let plen = pat.len();
    let mut from = 0;
    let mut out: Vec<(usize, bool, String)> = Vec::new();
    while let Some(idx) = text[from..].find(pat) {
        let qstart = from + idx + plen; // byte just past the opening `"`
        let Some(rel_end) = text[qstart..].find('"') else {
            // Open quote at byte `qstart - 1` (the `"` just past the pat's last char).
            return Err(crate::error::manifest::parse_err(
                qstart - 1,
                "unclosed quoted token",
            ));
        };
        let end = qstart + rel_end;
        out.push((from + idx, tag, text[qstart..end].to_string()));
        from = end + 1;
    }
    Ok(out)
}

/// Import the pin name↔number map from a `.kicad_sym` file.
///
/// `path` accepts any `AsRef<Path>` (typically `&Path`, `PathBuf`, or `&str`). All call
/// sites funnel through one concrete `PathBuf` so downstream `.to_path_buf()` /
/// `Display` / `read_to_string()` calls resolve to a known type.
pub fn import_symbol_pinmap(path: impl AsRef<std::path::Path>) -> Result<PinMap, crate::Error> {
    let path_buf: std::path::PathBuf = path.as_ref().to_path_buf();
    // Use the cross-file SSOT constructor at `crate::error::manifest::io_at` so the
    // failing case reads as one line and the inner `io::Error` joins a `#[source]`
    // chain via the same path every other import uses (Phase 1b dedup).
    let text = std::fs::read_to_string(&path_buf)
        .map_err(|source| crate::error::manifest::io_at(path_buf.clone(), source))?;
    // Within each (pin …), the name token precedes the number token; both carry a quoted value.
    // Each `quoted_events` call can fail with `Manifest::Parse` if a `(name "` or `(number "`
    // token is opened but never closed (half-finished vendor symbol, copy-paste artefact,
    // etc.); the partial event stream is discarded because the typed diagnostic supersedes
    // it — the user wants to know which token broke, not collect events that pair against
    // undefined payloads. The Phase 1c polish byte offset `qstart - 1` points at the
    // opening `"`; `parse_err` is the SSOT constructor that already carries the byte.
    let mut events: Vec<(usize, bool, String)> = quoted_events(&text, "(name \"", true)?;
    events.extend(quoted_events(&text, "(number \"", false)?);
    events.sort_by_key(|e: &(usize, bool, String)| e.0);

    let mut pins: Vec<(String, String)> = Vec::new();
    let mut pending: Option<String> = None;
    for (_, is_name, val) in events {
        if is_name {
            pending = Some(val);
        } else if let Some(name) = pending.take() {
            pins.push((name, val));
        }
    }
    if pins.is_empty() {
        // Routes through the cross-file SSOT helper at `crate::error::manifest::no_pins`.
        // Identical envelope shape to the inline literal otherwise; the hoist mirrors
        // the Phase 1b follow-up SSOT pattern.
        return Err(crate::error::manifest::no_pins(path_buf));
    }
    Ok(PinMap { pins })
}
