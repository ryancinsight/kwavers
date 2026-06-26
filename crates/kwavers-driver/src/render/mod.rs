//! Native SVG renderer for the routed board.
//!
//! Produces a self-contained vector image from the in-memory board model without any external
//! tool dependency. The output opens in any web browser, Inkscape, or vector graphics editor.
//!
//! # Layer colour scheme
//!
//! | Layer | Colour |
//! |-------|--------|
//! | F.Cu (0) | `#CC3333` red |
//! | In1.Cu | `#CC8833` orange |
//! | In2.Cu | `#33BB33` green |
//! | In3.Cu | `#5555CC` blue |
//! | In4.Cu | `#AA33AA` purple |
//! | B.Cu (last) | `#3388CC` cyan-blue |
//! | Inner > In4 | cycle the inner palette |
//!
//! Evidence tier: visual/empirical — the geometry is derived from the same board model used by
//! the LVS and DRC, so the render faithfully represents the routed copper.
//!
//! # Module layout (Phase 4d carve)
//!
//! The slice contains a single `board_svg` sub-axis because the two public entry points are
//! tightly coupled (`save_board_svg` is a thin I/O wrapper that calls `render_board_svg` and
//! writes the returned string to disk). If future render functionality grows past a single
//! concern (e.g. per-layer render, raw-placement render, layer-scheme variants), promote that
//! into a sibling sub-module and re-export through this facade.

pub mod board_svg;

pub use board_svg::{render_board_svg, save_board_svg};
