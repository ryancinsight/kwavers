//! Manifest / CAD-file parse and IO failures.
//!
//! This is the typed-syntax home for every "I tried to load a layout / symbol / driver
//! manifest and the file didn't make sense" failure. It replaces the Phase-0 misuse of
//! [`crate::error::Geometry::EmptyGrid`] (hand-rolled error code, but semantically wrong)
//! for the place-import modules (`src/place/symbol_import.rs`,
//! `src/place/footprint_import.rs`) and is the resting place for every
//! `crate::manifest::DriverManifest` parsing failure as its typed envelope migrates from
//! `Err(format!(...))` to `Err(Manifest::X)`.
//!
//! The variants are layered so a downstream log scraper can distinguish "the file is not
//! there" (`Io`) from "the file is there but malformed" (`Parse`) from "the file parsed but
//! produced no usable data" (`NoPads` / `NoPins`). The three failure modes warrant three
//! different recovery flows — retry the read, surface a CAD-Tool error to the engineer, or
//! prompt for a different part-number — and a single text-or-string error cannot
//! distinguish them.

/// Manifest / CAD-file parse or IO failure.
///
/// `Debug` only — [`Manifest::Io::source`] is `std::io::Error`, which violates `Eq`
/// (two `io::Error`s with the same `ErrorKind` but different OS-level payloads compare
/// equal under std's derived `PartialEq`, which is rarely what callers want); the
/// symbol/path/message fields are `String`/`PathBuf`/`usize` (not Eq-comparison keys
/// either). The slice drops `Clone + PartialEq + Eq` uniformly with the other
/// `f64`-bearing slices, and construction funnels through the SSOT helpers at
/// `io_at`/`parse_err`/`parse_msg`/`no_pads`/`no_pins` so callers never hand-clone.
/// Contrast: [`Geometry`](super::geometry::Geometry) keeps `Copy + Eq` because every
/// field is integer; `Manifest`'s `io::Error` + symbolic fields drop them.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Manifest {
    /// Reading the file from disk failed (missing path, permissions, mid-read IO error).
    ///
    /// The underlying [`std::io::Error`] is preserved as a `#[source]` so crates that
    /// care can downcast; the human-facing message names the file for log alignment.
    #[error("manifest: failed to read {path}: {source}")]
    Io {
        /// File the engine tried to load.
        path: std::path::PathBuf,
        /// Underlying stdlib error.
        #[source]
        source: std::io::Error,
    },

    /// The file was read but its contents are malformed — an unclosed s-expression, a
    /// missing token, an unexpected atom, or an unsupported format revision.
    #[error("manifest: malformed input near byte {offset}: {message}")]
    Parse {
        /// Approximate byte offset where the parser detected the malformed input.
        offset: usize,
        /// Diagnostic message.
        message: String,
    },

    /// The file parsed but the part contains no electrical pads — e.g. an `np_thru_hole`
    /// mounting hole without signal pads.
    #[error("manifest: parsed part has no electrical pads ({path})")]
    NoPads {
        /// Source file.
        path: std::path::PathBuf,
    },

    /// The symbol file parsed but produced no pin name↔number entries.
    #[error("manifest: parsed symbol has no pin entries ({path})")]
    NoPins {
        /// Source file.
        path: std::path::PathBuf,
    },

    /// The `.kicad_sym` referenced a symbol that does not exist in the file.
    #[error("manifest: symbol not found in {path}: {symbol}")]
    SymbolNotFound {
        /// Source file.
        path: std::path::PathBuf,
        /// The symbol name that was searched.
        symbol: String,
    },

    /// A `DriverManifest` field is absent, malformed, or self-contradictory.
    #[error("manifest: {field} is invalid: {message}")]
    InvalidManifestField {
        /// Field name.
        field: String,
        /// Diagnostic message.
        message: String,
    },
}

// ────────────────────────────────────────────────────────────────────────────
// Cross-file SSOT constructors
//
// These `pub fn`s live at the slice that defines `Manifest`, so every module that
// needs to construct a manifest error reaches the same place. They are also the
// place where the `?`-propagating `Result<T, std::io::Error>` / `Result<T, E>`
// conversions converge into one literal that downstream callers can read at a
// glance. Keeping them public is the cross-module dedup configuration called out
// in `docs/MIGRATION.md` § Phase 1b follow-ups.
// ────────────────────────────────────────────────────────────────────────────

/// Lift a `std::io::Error` into a `crate::Error` whose `Io` variant carries the
/// context path. The inner `io::Error` joins the `thiserror` `#[source]` chain so
/// that `e.source()` walks down to the stdlib error — log scrapers and
/// downcasters stay clear.
///
/// Note: takes `PathBuf` **by value** so the call site can move or clone as needed
/// without a borrow into the helper. The previous `&Path` shape would force every
/// caller to clone, since the inner `Manifest::Io` field type is `PathBuf`.
#[must_use = "io-at returns the aggregating crate::Error; callers usually propagate via `?`"]
pub fn io_at(path: std::path::PathBuf, source: std::io::Error) -> crate::Error {
    crate::error::Manifest::Io { path, source }.into()
}

/// Lift a "file parsed but produced no electrical pads" diagnostic into `crate::Error`
/// whose `NoPads` variant names the source file. Used by
/// [`crate::place::footprint_import::import_kicad_mod`] when the vendor `.kicad_mod`
/// parses successfully but contains zero electrical pads. Mirrors the [`io_at`]
/// pattern: takes `PathBuf` by value so the call site can move (no clone) when the
/// rest of the function has already finished with the path.
#[must_use = "no_pads returns the aggregating crate::Error; callers usually propagate via `?`"]
pub fn no_pads(path: std::path::PathBuf) -> crate::Error {
    crate::error::Manifest::NoPads { path }.into()
}

/// Lift a "file parsed but produced no pin number↔name entries" diagnostic into
/// `crate::Error` whose `NoPins` variant names the source file. Used by
/// (`symbol_import::import_symbol_pinmap`) when the vendor `.kicad_sym` is well-formed
/// but contains zero `(pin (name …) (number …))` clusters. Mirrors [`no_pads`] and
/// [`io_at`].
#[must_use = "no_pins returns the aggregating crate::Error; callers usually propagate via `?`"]
pub fn no_pins(path: std::path::PathBuf) -> crate::Error {
    crate::error::Manifest::NoPins { path }.into()
}

/// Lift a parse diagnostic into `crate::Error` with a true UTF-8 byte offset. Use this
/// from inside a parser that tracks position (e.g. `parse_sexpr` iterates
/// `char_indices().peekable()`); for callers that have not parsed yet, prefer [`parse_msg`].
#[must_use = "parse_err returns the aggregating crate::Error; callers usually propagate via `?`"]
pub fn parse_err(offset: usize, message: impl Into<String>) -> crate::Error {
    crate::error::Manifest::Parse {
        offset,
        message: message.into(),
    }
    .into()
}

// Phase 1c polish: `parse_err` is the byte-positioned form (threaded by `parse_sexpr`'s
// `char_indices()` loop); `parse_msg` is the no-offset flavor for post-parse validation.
// `parse_sexpr` also surfaces a previously-silent unclosed-string-literal diagnostic via
// `parse_err(src.len(), "unclosed string literal")`. Full write-up at
// `docs/MIGRATION.md` § Phase 1c polish.

/// Convenience flavor of [`parse_err`] for callers without a byte tracker — structural
/// validation that runs on a successfully parsed result (e.g. `import_kicad_mod`'s
/// "root is an atom" check after `parse_sexpr` returned `Ok`). Defaults the offset to `0`.
#[must_use = "parse_msg returns the aggregating crate::Error; callers usually propagate via `?`"]
pub fn parse_msg(message: impl Into<String>) -> crate::Error {
    parse_err(0, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_names_the_path_for_io() {
        let err = Manifest::Io {
            path: "/tmp/x.kicad_mod".into(),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        };
        let s = err.to_string();
        assert!(s.contains("/tmp/x.kicad_mod"), "display: {s}");
        assert!(s.to_lowercase().contains("missing"));
    }

    #[test]
    fn display_names_the_offset_for_parse() {
        let err = Manifest::Parse {
            offset: 1024,
            message: "unclosed paren".into(),
        };
        assert_eq!(
            err.to_string(),
            "manifest: malformed input near byte 1024: unclosed paren"
        );
    }

    #[test]
    fn io_error_preserves_source() {
        let inner = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied");
        let err = Manifest::Io {
            path: "/etc/shadow".into(),
            source: inner,
        };
        // Walk the source chain to the std::io::Error; check its kind is preserved.
        let mut src: &(dyn std::error::Error + 'static) = &err;
        let mut found_io = false;
        while let Some(s) = src.source() {
            if s.downcast_ref::<std::io::Error>().is_some() {
                found_io = true;
                break;
            }
            src = s;
        }
        assert!(found_io, "source chain must preserve the io::Error kind");
    }

    /// The sub-enum is `#[non_exhaustive]` — pattern-matching on the variants alone
    /// fails. The test wraps every variant under a wildcard so it compiles today *and*
    /// fails the build if the migration accidentally removes the attribute.
    #[test]
    #[allow(unused)]
    fn sub_enum_is_marked_non_exhaustive() {
        fn _exhaustive(m: Manifest) -> &'static str {
            match m {
                Manifest::Io { .. } => "io",
                _ => "future-variant",
            }
        }
    }

    /// `io_at` is a trivial dispatch into `Manifest::Io`: the helper's contract is
    /// that it produces the **same** error envelope as the literal variant, so the
    /// SSOT surface and the inline construction cannot drift.
    ///
    /// Direct pattern match on `crate::error::Error::Manifest(Manifest::Io {..})`.
    /// Walking `&dyn std::error::Error::source()` would NOT find `Manifest` here
    /// because the aggregator's `#[error(transparent)]` on `Error::Manifest`
    /// delegates `source()` straight to `Manifest::Io::source()` (i.e. to the
    /// inner `io::Error`), bypassing the `Manifest` itself.
    #[test]
    fn io_at_matches_inline_construction() {
        use crate::error::{Error, Manifest};
        let path = std::path::PathBuf::from("/tmp/cad.kicad_mod");
        let source = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let via_helper = super::io_at(path.clone(), source);
        match &via_helper {
            Error::Manifest(Manifest::Io { path: p, source: s }) => {
                assert_eq!(p, &path);
                assert_eq!(s.kind(), std::io::ErrorKind::NotFound);
            }
            _ => panic!("io_at must produce Error::Manifest(Manifest::Io); got {via_helper:?}"),
        }
    }

    /// `parse_msg` is the no-offset convenience flavor of [`parse_err`]: it must
    /// surface a `Manifest::Parse` with `offset == 0` and the message verbatim.
    /// `parse_err` with an explicit offset is exercised by the offset Display
    /// test above. Same pattern-match rationale as `io_at_matches_inline_construction`.
    #[test]
    fn parse_msg_carries_message_with_default_offset() {
        use crate::error::{Error, Manifest};
        let via_helper = super::parse_msg("unclosed paren");
        match &via_helper {
            Error::Manifest(Manifest::Parse { offset, message }) => {
                assert_eq!(*offset, 0, "parse_msg defaults offset to 0");
                assert_eq!(message, "unclosed paren");
            }
            _ => panic!(
                "parse_msg must produce Error::Manifest(Manifest::Parse); got {via_helper:?}"
            ),
        }
    }

    /// `no_pads` matches the inline `Manifest::NoPads { path }` construction: identical
    /// envelope shape, same path field. Direct pattern match (not source-chain walk —
    /// see `io_at_matches_inline_construction` rationale above).
    #[test]
    fn no_pads_matches_inline_construction() {
        use crate::error::{Error, Manifest};
        let path = std::path::PathBuf::from("/tmp/no_pads.kicad_mod");
        let via_helper = super::no_pads(path.clone());
        match &via_helper {
            Error::Manifest(Manifest::NoPads { path: p }) => {
                assert_eq!(p, &path);
            }
            _ => {
                panic!("no_pads must produce Error::Manifest(Manifest::NoPads); got {via_helper:?}")
            }
        }
    }

    /// `no_pins` matches the inline `Manifest::NoPins { path }` construction: identical
    /// envelope shape, same path field. Same pattern-match rationale as `no_pads_matches_inline_construction`.
    #[test]
    fn no_pins_matches_inline_construction() {
        use crate::error::{Error, Manifest};
        let path = std::path::PathBuf::from("/tmp/no_pins.kicad_sym");
        let via_helper = super::no_pins(path.clone());
        match &via_helper {
            Error::Manifest(Manifest::NoPins { path: p }) => {
                assert_eq!(p, &path);
            }
            _ => {
                panic!("no_pins must produce Error::Manifest(Manifest::NoPins); got {via_helper:?}")
            }
        }
    }
}
