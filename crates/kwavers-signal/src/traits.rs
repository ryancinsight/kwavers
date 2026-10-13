/// Base bound shared by the atomic signal component traits.
///
/// `Amplitude`, `Frequency`, and `Phase` each represent a single aspect of a
/// time-varying signal.  All three require thread safety and debug formatting;
/// declaring the common bounds here once prevents the `Debug + Send + Sync`
/// triple from being copy-pasted into each component trait.
///
/// See [`kwavers_core::ComponentBase`] for the cross-crate root of this hierarchy.
pub trait SignalPrimitive: kwavers_core::ComponentBase {}

/// Blanket implementation: any `ComponentBase` type satisfies `SignalPrimitive`.
impl<T: kwavers_core::ComponentBase> SignalPrimitive for T {}

/// Expand the one-line `clone_box` implementation shared by every
/// concrete signal type.
///
/// Every `clone_box` body is `Box::new(self.clone())` but the return type
/// differs per trait (`Box<dyn Signal>`, `Box<dyn Amplitude>`, etc.), so
/// the trait name must be passed as an argument.
///
/// # Usage
///
/// ```rust,ignore
/// impl Signal for MySig {
///     signal_clone_box!(Signal);
///     // … other required methods …
/// }
/// ```
#[macro_export]
macro_rules! signal_clone_box {
    ($trait:ident) => {
        fn clone_box(&self) -> ::std::boxed::Box<dyn $trait> {
            ::std::boxed::Box::new(self.clone())
        }
    };
}

/// Core Signal trait representing time-varying amplitude.
///
// dyn: used as `dyn Signal`/`Arc<dyn Signal>` (open, cross-crate-extensible
// implementor set — 22 impls across 6 crates: kwavers-signal, -source,
// -transducer, -solver, -python, facade). Per ADR 012 this is a sanctioned
// dynamic-dispatch boundary; a closed `SignalKind` enum is infeasible without
// circular crate dependencies. Dispatch is O(num_sources)/step (sampled once per
// source per step), never per-cell, so it is not throughput-critical.
pub trait Signal: kwavers_core::ComponentBase {
    /// Get the signal amplitude at time t
    fn amplitude(&self, t: f64) -> f64;

    /// Get the signal duration (if finite)
    fn duration(&self) -> Option<f64> {
        None
    }

    /// Get the instantaneous frequency at time t
    fn frequency(&self, t: f64) -> f64;

    /// Get the instantaneous phase at time t
    fn phase(&self, t: f64) -> f64;

    /// Clone the signal into a boxed trait object
    fn clone_box(&self) -> Box<dyn Signal>;
}

impl Clone for Box<dyn Signal> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
