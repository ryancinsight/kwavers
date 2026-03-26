/// Size distribution types
#[derive(Debug, Clone)]
pub enum SizeDistribution {
    Uniform { min: f64, max: f64 },
    LogNormal { mean: f64, std_dev: f64 },
    PowerLaw { min: f64, max: f64, exponent: f64 },
}

/// Spatial distribution types
#[derive(Debug, Clone)]
pub enum SpatialDistribution {
    Uniform,
    Gaussian {
        center: (f64, f64, f64),
        std_dev: f64,
    },
    Cluster {
        centers: Vec<(f64, f64, f64)>,
        radius: f64,
    },
}
