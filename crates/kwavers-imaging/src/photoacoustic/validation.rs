#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceptanceStatus {
    Pass,
    Fail,
}

#[derive(Debug, Clone)]
pub struct AcceptanceCheck {
    pub name: String,
    pub status: AcceptanceStatus,
    pub detail: String,
}
