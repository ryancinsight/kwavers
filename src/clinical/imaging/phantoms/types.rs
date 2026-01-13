/// Phantom type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhantomType {
    BloodOxygenation,
    LayeredTissue,
    TumorDetection,
    Vascular,
    Calibration,
    Custom,
}

#[derive(Clone, Debug)]
pub struct VesselSpec {
    pub center: [f64; 3],
    pub radius: f64,
    pub so2: f64,
    pub vessel_type: VesselType,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum VesselType {
    Artery,
    Vein,
    Capillary,
}

#[derive(Clone, Debug)]
pub struct TumorSpec {
    pub center: [f64; 3],
    pub radius: f64,
    pub so2: f64,
}

#[derive(Clone, Debug)]
pub struct LayerSpec {
    pub z_min: f64,
    pub z_max: f64,
    pub tissue_type: TissueType,
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum TissueType {
    SkinEpidermis,
    SkinDermis,
    Fat,
    Muscle,
    Liver,
    Brain,
    Bone,
    Custom(usize),
}

#[derive(Clone, Debug)]
pub struct VesselGeometry {
    pub start: [f64; 3],
    pub end: [f64; 3],
    pub radius: f64,
    pub so2: f64,
}
