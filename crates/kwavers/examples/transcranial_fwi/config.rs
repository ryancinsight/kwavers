//! Compile-time configuration for the transcranial FWI demonstration.

/// Zero-sized owner for the fixed demonstration grid and phantom constants.
pub(crate) struct GridSpec;

impl GridSpec {
    /// Grid spacing in metres.
    pub(crate) const DX: f64 = 3.0e-3;
    /// Lateral voxel count.
    pub(crate) const NX: usize = 64;
    /// Embedded depth-plane count required by the 3-D FDTD stencil.
    pub(crate) const NY: usize = 2;
    /// Axial voxel count.
    pub(crate) const NZ: usize = 64;

    /// Outer head radius in voxels.
    pub(crate) const R_HEAD: f64 = 26.0;
    /// Outer cortical boundary in voxels.
    pub(crate) const R_SKULL_OUT: f64 = 24.0;
    /// Diploe boundary in voxels.
    pub(crate) const R_DIPLOE: f64 = 21.0;
    /// Inner cortical boundary in voxels.
    pub(crate) const R_SKULL_IN: f64 = 18.0;
    /// Brain boundary in voxels.
    pub(crate) const R_BRAIN: f64 = 17.5;

    /// Hounsfield units for the water coupling bath.
    pub(crate) const HU_WATER: f64 = 0.0;
    /// Hounsfield units for scalp and dura.
    pub(crate) const HU_SCALP: f64 = 40.0;
    /// Hounsfield units for outer cortical bone.
    pub(crate) const HU_CORTICAL_OUT: f64 = 720.0;
    /// Hounsfield units for diploe.
    pub(crate) const HU_DIPLOE: f64 = 380.0;
    /// Hounsfield units for inner cortical bone.
    pub(crate) const HU_CORTICAL_IN: f64 = 660.0;
    /// Hounsfield units for brain parenchyma.
    pub(crate) const HU_BRAIN: f64 = 35.0;
}
