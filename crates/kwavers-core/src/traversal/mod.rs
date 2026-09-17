//! Parallel lockstep traversal of equally shaped fields.
//!
//! A kernel that writes one to three fields from zero to five others, element
//! by element, calls [`zip_mut`], [`zip_mut_pair`] or [`zip_mut_triple`] — or
//! the `_indexed` form of each when it needs the element's index — with the read-only views as a [`ZipInputs`] tuple:
//!
//! ```
//! use kwavers_core::traversal::zip_mut;
//! use leto::Array3;
//!
//! let pressure = Array3::from_elem([2, 3, 4], 2.0_f64);
//! let density = Array3::from_elem([2, 3, 4], 4.0_f64);
//! let mut ratio = Array3::zeros([2, 3, 4]);
//! zip_mut(ratio.view_mut(), (pressure.view(), density.view()), |r, (p, rho)| {
//!     *r = p / rho;
//! });
//! assert!(ratio.iter().all(|&r| r == 0.5));
//! ```
//!
//! Elements pair by logical row-major position, whatever each field's
//! storage order. When every field is C-contiguous the traversal runs as
//! moirai unit tasks whose width follows the bytes one element moves;
//! otherwise it walks the logical order on the calling thread.

mod inputs;
mod zip;

#[cfg(test)]
mod tests;

pub use inputs::ZipInputs;
pub use zip::{
    zip_mut, zip_mut_indexed, zip_mut_pair, zip_mut_pair_indexed, zip_mut_triple,
    zip_mut_triple_indexed,
};
