use super::types::*;
use super::utils::*;
use crate::clinical::imaging::chromophores::HemoglobinDatabase;
use crate::domain::grid::GridDimensions;
use crate::domain::medium::optical_map::{
    Layer, OpticalPropertyMap, OpticalPropertyMapBuilder, Region,
};
use crate::domain::medium::properties::OpticalPropertyData;

/// Clinical phantom builder with domain-specific presets
#[derive(Debug)]
pub struct PhantomBuilder {
    pub builder: OpticalPropertyMapBuilder,
    pub phantom_type: PhantomType,
    pub wavelength_nm: f64,
}

impl PhantomBuilder {
    /// Create blood oxygenation phantom builder
    ///
    /// Designed for spectroscopic photoacoustic imaging validation.
    /// Default wavelength: 800 nm (near-infrared, good tissue penetration)
    pub fn blood_oxygenation() -> BloodOxygenationPhantomBuilder {
        BloodOxygenationPhantomBuilder {
            dimensions: None,
            background: OpticalPropertyData::soft_tissue(),
            vessels: Vec::new(),
            tumors: Vec::new(),
            wavelength_nm: 800.0,
        }
    }

    /// Create layered tissue phantom builder
    ///
    /// Models stratified media (e.g., skin/fat/muscle).
    pub fn layered_tissue() -> LayeredTissuePhantomBuilder {
        LayeredTissuePhantomBuilder {
            dimensions: None,
            layers: Vec::new(),
            wavelength_nm: 800.0,
        }
    }

    /// Create tumor detection phantom builder
    ///
    /// Background tissue with embedded lesions for detection algorithm validation.
    pub fn tumor_detection() -> TumorDetectionPhantomBuilder {
        TumorDetectionPhantomBuilder {
            dimensions: None,
            background: OpticalPropertyData::soft_tissue(),
            tumors: Vec::new(),
            wavelength_nm: 800.0,
        }
    }

    /// Create vascular phantom builder
    ///
    /// Models vessel networks for angiogenesis and perfusion studies.
    pub fn vascular() -> VascularPhantomBuilder {
        VascularPhantomBuilder {
            dimensions: None,
            background: OpticalPropertyData::soft_tissue(),
            vessels: Vec::new(),
            wavelength_nm: 800.0,
        }
    }
}

/// Blood oxygenation phantom builder
///
/// Models arterial/venous vessels and tumors with specified oxygenation levels.
#[derive(Debug)]
pub struct BloodOxygenationPhantomBuilder {
    dimensions: Option<GridDimensions>,
    background: OpticalPropertyData,
    vessels: Vec<VesselSpec>,
    tumors: Vec<TumorSpec>,
    wavelength_nm: f64,
}

impl BloodOxygenationPhantomBuilder {
    /// Set grid dimensions
    pub fn dimensions(mut self, dims: GridDimensions) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Set wavelength (nm)
    pub fn wavelength(mut self, wavelength_nm: f64) -> Self {
        self.wavelength_nm = wavelength_nm;
        self
    }

    /// Set background tissue properties
    pub fn background(mut self, props: OpticalPropertyData) -> Self {
        self.background = props;
        self
    }

    /// Add arterial vessel (high oxygenation: sO₂ ≈ 0.95-0.99)
    pub fn add_artery(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.vessels.push(VesselSpec {
            center,
            radius,
            so2,
            vessel_type: VesselType::Artery,
        });
        self
    }

    /// Add venous vessel (lower oxygenation: sO₂ ≈ 0.60-0.75)
    pub fn add_vein(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.vessels.push(VesselSpec {
            center,
            radius,
            so2,
            vessel_type: VesselType::Vein,
        });
        self
    }

    /// Add capillary (intermediate oxygenation: sO₂ ≈ 0.70-0.85)
    pub fn add_capillary(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.vessels.push(VesselSpec {
            center,
            radius,
            so2,
            vessel_type: VesselType::Capillary,
        });
        self
    }

    /// Add tumor region (typically hypoxic: sO₂ ≈ 0.50-0.70)
    pub fn add_tumor(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.tumors.push(TumorSpec {
            center,
            radius,
            so2,
        });
        self
    }

    /// Build phantom
    pub fn build(self) -> OpticalPropertyMap {
        let dims = self
            .dimensions
            .expect("Dimensions must be set before building");

        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(self.background);

        let hb_db = HemoglobinDatabase::default();

        // Add vessels
        for vessel in &self.vessels {
            let props = compute_blood_properties(&hb_db, self.wavelength_nm, vessel.so2);
            builder.add_region(Region::sphere(vessel.center, vessel.radius), props);
        }

        // Add tumors
        for tumor in &self.tumors {
            // Tumors have enhanced vascularity but are typically hypoxic
            let props = compute_tumor_properties(&hb_db, self.wavelength_nm, tumor.so2);
            builder.add_region(Region::sphere(tumor.center, tumor.radius), props);
        }

        builder.build()
    }
}

/// Layered tissue phantom builder
///
/// Models stratified media with horizontal layers (e.g., skin/fat/muscle).
#[derive(Debug)]
pub struct LayeredTissuePhantomBuilder {
    dimensions: Option<GridDimensions>,
    layers: Vec<LayerSpec>,
    wavelength_nm: f64,
}

impl LayeredTissuePhantomBuilder {
    /// Set grid dimensions
    pub fn dimensions(mut self, dims: GridDimensions) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Set wavelength (nm)
    pub fn wavelength(mut self, wavelength_nm: f64) -> Self {
        self.wavelength_nm = wavelength_nm;
        self
    }

    /// Add skin epidermis layer
    pub fn add_skin_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::SkinEpidermis,
        });
        self
    }

    /// Add dermis layer
    pub fn add_dermis_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::SkinDermis,
        });
        self
    }

    /// Add fat layer
    pub fn add_fat_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::Fat,
        });
        self
    }

    /// Add muscle layer
    pub fn add_muscle_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::Muscle,
        });
        self
    }

    /// Add liver layer
    pub fn add_liver_layer(mut self, z_min: f64, z_max: f64) -> Self {
        self.layers.push(LayerSpec {
            z_min,
            z_max,
            tissue_type: TissueType::Liver,
        });
        self
    }

    /// Build phantom
    pub fn build(self) -> OpticalPropertyMap {
        let dims = self
            .dimensions
            .expect("Dimensions must be set before building");

        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(OpticalPropertyData::soft_tissue());

        for layer_spec in &self.layers {
            let props = get_tissue_properties(layer_spec.tissue_type);
            builder.add_layer(Layer::new(layer_spec.z_min, layer_spec.z_max, props));
        }

        builder.build()
    }
}

/// Tumor detection phantom builder
#[derive(Debug)]
pub struct TumorDetectionPhantomBuilder {
    dimensions: Option<GridDimensions>,
    background: OpticalPropertyData,
    tumors: Vec<TumorSpec>,
    wavelength_nm: f64,
}

impl TumorDetectionPhantomBuilder {
    /// Set grid dimensions
    pub fn dimensions(mut self, dims: GridDimensions) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Set wavelength (nm)
    pub fn wavelength(mut self, wavelength_nm: f64) -> Self {
        self.wavelength_nm = wavelength_nm;
        self
    }

    /// Set background tissue properties
    pub fn background(mut self, props: OpticalPropertyData) -> Self {
        self.background = props;
        self
    }

    /// Add tumor lesion
    pub fn add_tumor(mut self, center: [f64; 3], radius: f64, so2: f64) -> Self {
        self.tumors.push(TumorSpec {
            center,
            radius,
            so2,
        });
        self
    }

    /// Build phantom
    pub fn build(self) -> OpticalPropertyMap {
        let dims = self
            .dimensions
            .expect("Dimensions must be set before building");

        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(self.background);

        let hb_db = HemoglobinDatabase::default();

        for tumor in &self.tumors {
            let props = compute_tumor_properties(&hb_db, self.wavelength_nm, tumor.so2);
            builder.add_region(Region::sphere(tumor.center, tumor.radius), props);
        }

        builder.build()
    }
}

/// Vascular phantom builder
#[derive(Debug)]
pub struct VascularPhantomBuilder {
    dimensions: Option<GridDimensions>,
    background: OpticalPropertyData,
    vessels: Vec<VesselGeometry>,
    wavelength_nm: f64,
}

impl VascularPhantomBuilder {
    /// Set grid dimensions
    pub fn dimensions(mut self, dims: GridDimensions) -> Self {
        self.dimensions = Some(dims);
        self
    }

    /// Set wavelength (nm)
    pub fn wavelength(mut self, wavelength_nm: f64) -> Self {
        self.wavelength_nm = wavelength_nm;
        self
    }

    /// Set background tissue properties
    pub fn background(mut self, props: OpticalPropertyData) -> Self {
        self.background = props;
        self
    }

    /// Add vessel segment (cylinder)
    pub fn add_vessel(mut self, start: [f64; 3], end: [f64; 3], radius: f64, so2: f64) -> Self {
        self.vessels.push(VesselGeometry {
            start,
            end,
            radius,
            so2,
        });
        self
    }

    /// Build phantom
    pub fn build(self) -> OpticalPropertyMap {
        let dims = self
            .dimensions
            .expect("Dimensions must be set before building");

        let mut builder = OpticalPropertyMapBuilder::new(dims);
        builder.set_background(self.background);

        let hb_db = HemoglobinDatabase::default();

        for vessel in &self.vessels {
            let props = compute_blood_properties(&hb_db, self.wavelength_nm, vessel.so2);
            builder.add_region(
                Region::cylinder(vessel.start, vessel.end, vessel.radius),
                props,
            );
        }

        builder.build()
    }
}
