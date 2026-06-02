/// Histotripsy exposure regime.
///
/// Each regime is defined by the dominant nucleation mechanism, not by the
/// pulse duration alone. Sub-threshold millisecond cavitation does not
/// require bulk boiling; boiling histotripsy seeds a vapor bubble through
/// shock-rich absorption heating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistotripsyRegime {
    /// Intrinsic-threshold (classical "microsecond") histotripsy.
    /// 1–20 cycles, `|p^-_min| > p_t(f)`. Single-pulse nucleation from
    /// pre-existing nuclei (Maxwell 2013, Vlaisavljevich 2015).
    IntrinsicThreshold,
    /// Shock-scattering histotripsy. 3–20 cycles, sub-threshold rarefactional
    /// drive but pre-focal positive shock backscatters off a seed bubble to
    /// produce locally super-threshold tension that grows the cloud
    /// (Maxwell 2011).
    ShockScattering,
    /// Boiling histotripsy. ms-duration shock-rich pulses; rapid absorption
    /// of the shock energy produces a millimetre-scale vapor bubble in
    /// ~5–20 ms which drives the cavitation cloud (Khokhlova 2014, 2019).
    Boiling,
    /// Sub-threshold millisecond cavitation. Long pulses below `p_t(f)`
    /// rely on many-cycle bubble growth and inertial collapse of stable
    /// nuclei; mechanism is purely cavitation, not bulk boiling
    /// (Vlaisavljevich 2018).
    MillisecondCavitation,
}

impl HistotripsyRegime {
    /// Whether the regime relies on a single-pulse intrinsic-threshold
    /// nucleation event.
    #[must_use]
    pub const fn is_intrinsic_threshold(self) -> bool {
        matches!(self, Self::IntrinsicThreshold)
    }

    /// Whether the regime is fundamentally a millisecond exposure.
    #[must_use]
    pub const fn is_millisecond(self) -> bool {
        matches!(self, Self::Boiling | Self::MillisecondCavitation)
    }
}
