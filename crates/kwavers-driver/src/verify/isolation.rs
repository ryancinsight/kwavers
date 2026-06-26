//! Isolation — graph-based BFS isolation path check.
use crate::board::{Board, NetClassKind, NetId};
use crate::place::component::Component;
use crate::place::footprint::FootprintDef;
use std::collections::{HashMap, HashSet, VecDeque};

/// Schematic isolation violation: a path from control logic to high-voltage pulser domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IsolationViolation {
    /// Net name on the control domain.
    pub control_net: String,
    /// Net name on the high-voltage/pulser domain.
    pub hv_net: String,
    /// Net/component path showing the leakage.
    pub path: Vec<String>,
}

/// Results of schematic isolation BFS.
#[derive(Debug, Clone, Default)]
pub struct IsolationReport {
    /// List of violations.
    pub violations: Vec<IsolationViolation>,
    /// True if no violations are found.
    pub pass: bool,
}

/// Traverses the schematic connection graph to verify that low-voltage control nets
/// are isolated from high-voltage pulser nets by digital isolator components.
#[must_use]
pub fn schematic_isolation_bfs(
    board: &Board,
    comps: &[Component],
    lib: &[FootprintDef],
) -> IsolationReport {
    let mut hv_nets = HashSet::new();
    for net in &board.nets {
        if net.class == NetClassKind::Hv {
            hv_nets.insert(net.id);
        }
    }
    for c in comps {
        let name = c.refdes.to_uppercase();
        let fp_name = lib[c.fp].name.to_uppercase();
        if name.starts_with("U1")
            || name.starts_with("U2")
            || name.starts_with("U3")
            || fp_name.contains("HV7355")
        {
            for &net in c.nets.iter().flatten() {
                hv_nets.insert(net);
            }
        }
    }

    let mut control_nets = HashSet::new();
    for c in comps {
        let name = c.refdes.to_uppercase();
        let fp_name = lib[c.fp].name.to_uppercase();
        if name.starts_with("J_STACK")
            || name == "J2"
            || name.starts_with("U4")
            || fp_name.contains("CPLD")
            || fp_name.contains("FPGA")
        {
            for &net in c.nets.iter().flatten() {
                if board.class_of(net).is_low_voltage() {
                    control_nets.insert(net);
                }
            }
        }
    }

    let is_traversable = |n: NetId| {
        let class = board.class_of(n);
        matches!(class, NetClassKind::Signal | NetClassKind::Hv)
    };

    let mut adj: HashMap<NetId, Vec<(NetId, String)>> = HashMap::new();

    for c in comps {
        let refdes = &c.refdes;
        let name_upper = refdes.to_uppercase();
        if name_upper.starts_with("ISO") || name_upper.starts_with("J") {
            continue;
        }

        let connected_nets: Vec<NetId> = c
            .nets
            .iter()
            .filter_map(|&n| n)
            .filter(|&n| is_traversable(n))
            .collect();

        for i in 0..connected_nets.len() {
            for j in (i + 1)..connected_nets.len() {
                let n1 = connected_nets[i];
                let n2 = connected_nets[j];
                if n1 != n2 {
                    adj.entry(n1).or_default().push((n2, refdes.clone()));
                    adj.entry(n2).or_default().push((n1, refdes.clone()));
                }
            }
        }
    }

    let mut violations = Vec::new();

    for &start in &control_nets {
        if !is_traversable(start) {
            continue;
        }
        let mut visited = HashSet::new();
        visited.insert(start);
        let mut queue = VecDeque::new();
        let start_name = board.nets[start.0 as usize].name.clone();
        queue.push_back((start, vec![start_name]));

        while let Some((curr, path)) = queue.pop_front() {
            if hv_nets.contains(&curr) {
                let end_name = board.nets[curr.0 as usize].name.clone();
                violations.push(IsolationViolation {
                    control_net: board.nets[start.0 as usize].name.clone(),
                    hv_net: end_name,
                    path,
                });
                break;
            }

            if let Some(neighbors) = adj.get(&curr) {
                for &(next, ref component) in neighbors {
                    if visited.insert(next) {
                        let next_name = board.nets[next.0 as usize].name.clone();
                        let mut new_path = path.clone();
                        new_path.push(component.clone());
                        new_path.push(next_name);
                        queue.push_back((next, new_path));
                    }
                }
            }
        }
    }

    violations.sort_by(|a, b| a.control_net.cmp(&b.control_net));
    let pass = violations.is_empty();
    IsolationReport { violations, pass }
}
