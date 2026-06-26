//! The role a board plays in the shield stack — the top controller versus a high-voltage driver —
//! plus its stable manifest spelling and the reverse parse.

/// A board role in the shield stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackBoardRole {
    /// Top controller shield carrying the FPGA/programming interface.
    Controller,
    /// A high-voltage driver shield.
    Driver,
}

impl StackBoardRole {
    /// Stable manifest spelling.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            StackBoardRole::Controller => "controller",
            StackBoardRole::Driver => "driver",
        }
    }
}

impl TryFrom<&str> for StackBoardRole {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "controller" => Ok(StackBoardRole::Controller),
            "driver" => Ok(StackBoardRole::Driver),
            other => Err(format!("unknown stack board role {other}")),
        }
    }
}
