//! Spectral element geometry: hexahedral element and mesh collection.
//!
//! ## Module layout
//!
//! | Sub-module   | Responsibility                                         |
//! |--------------|--------------------------------------------------------|
//! | `element`    | [`SemElement`] — per-element Jacobian geometry         |
//! | `jacobian`   | 3×3 Jacobian computation (pure math, no struct state)  |
//! | `collection` | [`SemMesh`]   — mesh topology and construction         |

mod collection;
mod element;
mod jacobian;

pub use collection::SemMesh;
pub use element::SemElement;
