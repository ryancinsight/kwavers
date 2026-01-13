use super::builder::PhantomBuilder;
use crate::domain::grid::GridDimensions;
use crate::domain::medium::properties::OpticalPropertyData;
use crate::physics::optics::map_builder::OpticalPropertyMap;

/// Predefined clinical phantoms
#[derive(Debug)]
pub struct ClinicalPhantoms;

impl ClinicalPhantoms {
    /// Standard blood oxygenation phantom
    ///
    /// Contains arterial vessel (sO₂=98%), venous vessel (sO₂=65%),
    /// and hypoxic tumor (sO₂=55%).
    pub fn standard_blood_oxygenation(dims: GridDimensions) -> OpticalPropertyMap {
        let cx = dims.dx * (dims.nx as f64) / 2.0;
        let cy = dims.dy * (dims.ny as f64) / 2.0;
        let cz = dims.dz * (dims.nz as f64) / 2.0;

        PhantomBuilder::blood_oxygenation()
            .dimensions(dims)
            .wavelength(800.0)
            .add_artery([cx - 0.005, cy, cz - 0.005], 0.002, 0.98)
            .add_vein([cx + 0.005, cy, cz - 0.005], 0.003, 0.65)
            .add_tumor([cx, cy, cz + 0.003], 0.004, 0.55)
            .build()
    }

    /// Skin tissue phantom (epidermis/dermis/fat/muscle)
    pub fn skin_tissue(dims: GridDimensions) -> OpticalPropertyMap {
        PhantomBuilder::layered_tissue()
            .dimensions(dims)
            .wavelength(800.0)
            .add_skin_layer(0.0, 0.001) // 1 mm epidermis
            .add_dermis_layer(0.001, 0.003) // 2 mm dermis
            .add_fat_layer(0.003, 0.010) // 7 mm fat
            .add_muscle_layer(0.010, dims.dz * dims.nz as f64) // Muscle below
            .build()
    }

    /// Breast tissue phantom with tumor
    pub fn breast_tumor(dims: GridDimensions, tumor_center: [f64; 3]) -> OpticalPropertyMap {
        PhantomBuilder::tumor_detection()
            .dimensions(dims)
            .wavelength(800.0)
            .background(OpticalPropertyData::fat()) // Breast is primarily fat
            .add_tumor(tumor_center, 0.008, 0.60) // 8 mm hypoxic tumor
            .build()
    }

    /// Vascular network phantom
    pub fn vascular_network(dims: GridDimensions) -> OpticalPropertyMap {
        let cx = dims.dx * (dims.nx as f64) / 2.0;
        let cy = dims.dy * (dims.ny as f64) / 2.0;
        let z_min = 0.0;
        let z_max = dims.dz * dims.nz as f64;

        PhantomBuilder::vascular()
            .dimensions(dims)
            .wavelength(800.0)
            .add_vessel(
                [cx - 0.003, cy, z_min],
                [cx - 0.003, cy, z_max],
                0.001,
                0.97,
            ) // Artery
            .add_vessel(
                [cx, cy - 0.003, z_min],
                [cx, cy - 0.003, z_max],
                0.0008,
                0.97,
            ) // Artery
            .add_vessel(
                [cx + 0.003, cy, z_min],
                [cx + 0.003, cy, z_max],
                0.0015,
                0.68,
            ) // Vein
            .add_vessel(
                [cx, cy + 0.003, z_min],
                [cx, cy + 0.003, z_max],
                0.0012,
                0.68,
            ) // Vein
            .build()
    }
}
