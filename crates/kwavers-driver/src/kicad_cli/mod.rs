//! External **kicad-cli** wrapper — turns the previously-manual DRC/Gerber/drill/BOM/render
//! invocations into something the example [`cargo run`] drives, and feeds the DRC verdict back
//! into the verification suite. Closes the residual-risk item *"External KiCad DRC is currently a
//! manual empirical gate invoked through kicad-cli.exe; it is not yet automated in CI"*.
//!
//! Anything that shells out returns [`Result<T, String>`] with the full `STDOUT`+`STDERR` of a
//! failing kicad-cli call so a vendor-process crash is debuggable from the example log. The wrapper
//! therefore *cannot* be a hidden soft-dependency: if kicad-cli is missing it surfaces a single clear
//! error and the caller decides whether to fail the build.
//!
//! # Slice layout
//!
//! Carved by **role** (Phase 4a output-slice migration). Plain backticks name the slice-private
//! submodules (keeps `rustdoc`'s `private_intra_doc_links` lint quiet); the public types each hosts
//! stay clickable.
//! * `cli` — the [`KiCadCli`] process wrapper + [`DrcOptions`]: locate/spawn the external binary and
//!   drive `pcb drc` / `pcb render` / fab export.
//! * `drc` — the [`DrcReport`] / [`DrcDefectCount`] model and the permissive, version-tolerant
//!   KiCad-DRC JSON parser.
//! * `fab` — the [`FabBundle`] artifact set produced by [`KiCadCli::export_fab`].

mod cli;
mod drc;
mod fab;

#[cfg(test)]
mod tests;

pub use cli::{DrcOptions, KiCadCli};
pub use drc::{DrcDefectCount, DrcReport};
pub use fab::FabBundle;
