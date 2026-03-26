/// Photon state vector
#[derive(Clone, Debug)]
pub(crate) struct Photon {
    pub position: [f64; 3],
    pub direction: [f64; 3],
    pub weight: f64,
    pub alive: bool,
}
