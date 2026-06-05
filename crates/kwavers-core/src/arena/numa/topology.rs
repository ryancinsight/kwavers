/// NUMA topology information.
#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub node_count: usize,
    pub total_cpus: usize,
    pub cpus_per_node: usize,
    pub distance_matrix: Vec<Vec<u32>>,
    pub has_numa: bool,
}

impl Default for NumaTopology {
    fn default() -> Self {
        Self::single_node()
    }
}

impl NumaTopology {
    #[must_use]
    pub fn single_node() -> Self {
        let cpus = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self {
            node_count: 1,
            total_cpus: cpus,
            cpus_per_node: cpus,
            distance_matrix: vec![vec![10]],
            has_numa: false,
        }
    }

    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_os = "linux")]
        return Self::detect_linux();

        #[cfg(target_os = "windows")]
        return Self::detect_windows();

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        Self::single_node()
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Self {
        use std::fs;

        let node_count = fs::read_dir("/sys/devices/system/node/")
            .ok()
            .map(|dir| {
                dir.filter_map(|entry| {
                    let path = entry.ok()?.path();
                    let name = path.file_name()?.to_string_lossy().into_owned();
                    if name.starts_with("node") && name.len() > 4 {
                        name[4..].parse::<usize>().ok()
                    } else {
                        None
                    }
                })
                .count()
            })
            .filter(|c| *c > 0)
            .unwrap_or(1);

        if node_count <= 1 {
            return Self::single_node();
        }

        let mut distance_matrix = Vec::with_capacity(node_count);
        for node in 0..node_count {
            let mut row = Vec::with_capacity(node_count);
            let dist_str =
                fs::read_to_string(format!("/sys/devices/system/node/node{}/distance", node))
                    .unwrap_or_default();
            for target in 0..node_count {
                let dist = dist_str
                    .split_whitespace()
                    .nth(target)
                    .and_then(|n| n.parse::<u32>().ok())
                    .unwrap_or(if node == target { 10 } else { 20 });
                row.push(dist);
            }
            distance_matrix.push(row);
        }

        let total_cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        Self {
            node_count,
            total_cpus,
            cpus_per_node: total_cpus / node_count,
            distance_matrix,
            has_numa: true,
        }
    }

    #[cfg(target_os = "windows")]
    fn detect_windows() -> Self {
        Self::single_node()
    }

    #[must_use]
    pub fn distance(&self, from: usize, to: usize) -> u32 {
        if from >= self.node_count || to >= self.node_count {
            return 20;
        }
        self.distance_matrix[from][to]
    }

    #[must_use]
    pub fn nearest_node(&self, node: usize) -> Option<usize> {
        if !self.has_numa || node >= self.node_count {
            return Some(0);
        }
        let mut min_dist = u32::MAX;
        let mut nearest = None;
        for other in 0..self.node_count {
            if other == node {
                continue;
            }
            let d = self.distance(node, other);
            if d < min_dist {
                min_dist = d;
                nearest = Some(other);
            }
        }
        nearest
    }

    #[must_use]
    pub fn nodes_by_distance(&self, from: usize) -> Vec<(usize, u32)> {
        if !self.has_numa || from >= self.node_count {
            return vec![(0, 10)];
        }
        let mut nodes: Vec<(usize, u32)> = (0..self.node_count)
            .map(|n| (n, self.distance(from, n)))
            .collect();
        nodes.sort_by_key(|(_, d)| *d);
        nodes
    }
}
