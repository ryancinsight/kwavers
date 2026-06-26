//! Minimal KiCad S-expression parser kernel shared by the `.kicad_mod` footprint importer
//! ([`crate::place::footprint_import`]) and the `.kicad_pcb` board reader
//! ([`crate::io::pcb_parse`]).
//!
//! Second-occurrence consolidation trigger: `io::pcb_parse` is the second consumer of the
//! S-expression type system, so the parser migrated here from `footprint_import.rs`
//! (where it lived as `pub(super)` from Phase 2c).
//!
//! # Evidence tier
//!
//! Value-semantic unit tests in `crate::place::tests` pin the byte-offset contract
//! (`parse_sexpr_unclosed_*`, `parse_sexpr_unicode_byte_offset_differs_from_char_offset`).

/// A parsed s-expression node: an atom (bare or quoted token) or a parenthesised list.
///
/// Visible `pub(crate)` so both `place::footprint_import` and `io::pcb_parse` share the
/// single definition without external leakage.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Sexpr {
    /// A bare word or quoted string token.
    Atom(String),
    /// A parenthesised `( … )` list.
    List(Vec<Sexpr>),
}

impl Sexpr {
    /// The list items if `self` is a `List`, otherwise `None`.
    pub(crate) fn as_list(&self) -> Option<&[Sexpr]> {
        match self {
            Sexpr::List(v) => Some(v),
            Sexpr::Atom(_) => None,
        }
    }

    /// The atom string if `self` is an `Atom`, otherwise `None`.
    pub(crate) fn as_atom(&self) -> Option<&str> {
        match self {
            Sexpr::Atom(s) => Some(s),
            Sexpr::List(_) => None,
        }
    }

    /// The head keyword of a list (`(head …)`), if this is a list whose first element is an atom.
    pub(crate) fn head(&self) -> Option<&str> {
        self.as_list()?.first()?.as_atom()
    }
}

/// Tokenise and parse one top-level s-expression.
///
/// KiCad `.kicad_mod` files are a single `(footprint …)` form; `.kicad_pcb` files are a single
/// `(kicad_pcb …)` form.
///
/// # Byte-position tracking
///
/// Every [`crate::error::Manifest::Parse`] surfaced from this function carries the true UTF-8
/// byte offset of the offending token — the byte at which a debugger can plant a cursor. The
/// state machine iterates with `char_indices().peekable()` — NOT `chars().enumerate()` — so
/// multi-byte UTF-8 sequences point the offset at the actual byte, not the Unicode-scalar ordinal.
///
/// # Errors
///
/// Returns [`crate::Error::Manifest`] with a `Parse` variant when:
/// - an unexpected `)` appears with no matching `(` (byte offset points at the `)`),
/// - a string literal is not closed before end-of-input (offset = `src.len()`),
/// - the input ends before the top-level list is closed (offset = `src.len()`).
pub(crate) fn parse_sexpr(src: &str) -> crate::Result<Sexpr> {
    let mut chars = src.char_indices().peekable();
    let mut stack: Vec<Vec<Sexpr>> = Vec::new();
    let mut cur: Option<Vec<Sexpr>> = None;
    while let Some(&(_pos, c)) = chars.peek() {
        match c {
            '(' => {
                chars.next();
                if let Some(c) = cur.take() {
                    stack.push(c);
                }
                cur = Some(Vec::new());
            }
            ')' => {
                let (pos, _) = chars.next().expect("invariant: peek() returned Some");
                let done = cur.take().ok_or_else(|| {
                    crate::error::manifest::parse_err(
                        pos,
                        "unexpected closing paren — no open list",
                    )
                })?;
                let node = Sexpr::List(done);
                match stack.pop() {
                    Some(mut parent) => {
                        parent.push(node);
                        cur = Some(parent);
                    }
                    None => return Ok(node),
                }
            }
            '"' => {
                chars.next();
                let mut s = String::new();
                let mut closed = false;
                while let Some(&(_pos, c)) = chars.peek() {
                    chars.next();
                    if c == '\\' {
                        if let Some(&(_, e)) = chars.peek() {
                            chars.next();
                            s.push(e);
                        }
                    } else if c == '"' {
                        closed = true;
                        break;
                    } else {
                        s.push(c);
                    }
                }
                if !closed {
                    return Err(crate::error::manifest::parse_err(
                        src.len(),
                        "unclosed string literal",
                    ));
                }
                if let Some(list) = cur.as_mut() {
                    list.push(Sexpr::Atom(s));
                }
            }
            c if c.is_whitespace() => {
                chars.next();
            }
            _ => {
                let mut s = String::new();
                while let Some(&(_pos, c)) = chars.peek() {
                    if c == '(' || c == ')' || c == '"' || c.is_whitespace() {
                        break;
                    }
                    s.push(c);
                    chars.next();
                }
                if let Some(list) = cur.as_mut() {
                    list.push(Sexpr::Atom(s));
                }
            }
        }
    }
    Err(crate::error::manifest::parse_err(
        src.len(),
        "input ended before top-level s-expression closed",
    ))
}

/// Find the first direct child list of `node` whose head is `key`.
pub(crate) fn child<'a>(node: &'a Sexpr, key: &str) -> Option<&'a Sexpr> {
    node.as_list()?.iter().find(|c| c.head() == Some(key))
}

/// Parse `f64` from an atom at position `i` of a list.
pub(crate) fn num(list: &[Sexpr], i: usize) -> Option<f64> {
    list.get(i)?.as_atom()?.parse::<f64>().ok()
}

/// Parse a KiCad nested vector such as `(offset (xyz x y z))` or `(rotate (xyz rx ry rz))`.
pub(crate) fn xyz_child(node: &Sexpr, key: &str) -> Option<(f64, f64, f64)> {
    let xyz = child(child(node, key)?, "xyz")?;
    let list = xyz.as_list()?;
    Some((num(list, 1)?, num(list, 2)?, num(list, 3)?))
}
