//! Reactive Oxygen Species (ROS) types

/// Enumeration of reactive oxygen species
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ROSSpecies {
    /// Hydroxyl radical (•OH) - most reactive
    HydroxylRadical,
    /// Hydrogen peroxide (H₂O₂)
    HydrogenPeroxide,
    /// Superoxide anion (O₂•⁻)
    Superoxide,
    /// Singlet oxygen (¹O₂)
    SingletOxygen,
    /// Ozone (O₃)
    Ozone,
    /// Hydroperoxyl radical (HO₂•)
    HydroperoxylRadical,
    /// Atomic oxygen (O)
    AtomicOxygen,
    /// Atomic hydrogen (H)
    AtomicHydrogen,
    /// Peroxynitrite (ONOO⁻)
    Peroxynitrite,
    /// Nitric oxide (NO)
    NitricOxide,
    /// Nitrogen dioxide (NO₂)
    NitrogenDioxide,
}
