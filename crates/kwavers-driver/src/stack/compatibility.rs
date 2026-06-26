//! The controller↔driver stack-connector mating check: identical board geometry and connector
//! placement, plus pin-by-pin net compatibility, between two [`StackBoardManifest`]s.

use super::manifest::StackBoardManifest;
use super::role::StackBoardRole;
use super::util::check_close;

/// Result of comparing the controller shield stack connector to one driver shield connector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackCompatibility {
    /// Whether the shields can mate electrically and mechanically.
    pub pass: bool,
    /// Human-readable mismatch descriptions.
    pub mismatches: Vec<String>,
}

/// Verify that a controller shield and driver shield stack connector mate with identical geometry
/// and compatible pin semantics.
#[must_use]
pub fn verify_stack_pair(
    controller: &StackBoardManifest,
    driver: &StackBoardManifest,
) -> StackCompatibility {
    let mut mismatches = Vec::new();
    if controller.role != StackBoardRole::Controller {
        mismatches.push(format!(
            "controller manifest role is {}",
            controller.role.as_str()
        ));
    }
    if driver.role != StackBoardRole::Driver {
        mismatches.push(format!("driver manifest role is {}", driver.role.as_str()));
    }
    check_close(
        "board width",
        controller.board_w_mm,
        driver.board_w_mm,
        &mut mismatches,
    );
    check_close(
        "board height",
        controller.board_h_mm,
        driver.board_h_mm,
        &mut mismatches,
    );
    check_close(
        "connector x",
        controller.connector_x_mm,
        driver.connector_x_mm,
        &mut mismatches,
    );
    check_close(
        "connector y",
        controller.connector_y_mm,
        driver.connector_y_mm,
        &mut mismatches,
    );
    check_close(
        "connector rotation",
        controller.connector_rot_deg,
        driver.connector_rot_deg,
        &mut mismatches,
    );
    if controller.pin_nets.len() != driver.pin_nets.len() {
        mismatches.push(format!(
            "pin count differs: controller {} driver {}",
            controller.pin_nets.len(),
            driver.pin_nets.len()
        ));
    }
    for (idx, (a, b)) in controller
        .pin_nets
        .iter()
        .zip(driver.pin_nets.iter())
        .enumerate()
    {
        if a != b {
            mismatches.push(format!("pin {} net differs: {a} != {b}", idx + 1));
        }
    }
    StackCompatibility {
        pass: mismatches.is_empty(),
        mismatches,
    }
}
