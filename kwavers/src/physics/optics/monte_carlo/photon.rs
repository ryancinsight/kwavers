/// Photon state vector
#[derive(Clone, Debug)]
pub(crate) struct Photon {
    pub position: [f64; 3],
    pub direction: [f64; 3],
    pub weight: f64,
    /// Propagation flag; set at launch, termination governed by solver loop logic
    #[allow(dead_code)]
    pub alive: bool,
}
