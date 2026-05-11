//! IMEXRK struct definition and coefficient construction.

use super::types::{IMEXRKConfig, IMEXRKType};

/// IMEX Runge-Kutta scheme
#[derive(Debug)]
pub struct IMEXRK {
    pub(super) config: IMEXRKConfig,
    /// Explicit RK coefficients (`a_ij`)
    pub(super) a_explicit: Vec<Vec<f64>>,
    /// Implicit RK coefficients (`a_ij`)
    pub(super) a_implicit: Vec<Vec<f64>>,
    /// RK weights (`b_i`)
    pub(super) b: Vec<f64>,
    /// Number of stages
    pub(super) s: usize,
    /// Order of the method
    pub(super) p: usize,
    /// Stiffness adjustment factor
    pub(super) stiffness_factor: f64,
}

impl IMEXRK {
    /// Create a new IMEX-RK scheme
    #[must_use]
    pub fn new(config: IMEXRKConfig) -> Self {
        let (a_explicit, a_implicit, b, _c, s, p) = match config.scheme_type {
            IMEXRKType::SSP2_222 => Self::ssp2_222_coefficients(),
            IMEXRKType::SSP3_333 => Self::ssp3_333_coefficients(),
            IMEXRKType::ARK3 => Self::ark3_coefficients(),
            IMEXRKType::ARK4 => Self::ark4_coefficients(),
        };

        Self {
            config,
            a_explicit,
            a_implicit,
            b,
            s,
            p,
            stiffness_factor: 1.0,
        }
    }

    /// SSP2(2,2,2) coefficients
    fn ssp2_222_coefficients() -> (
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        usize,
        usize,
    ) {
        // Explicit tableau
        let a_explicit = vec![vec![0.0, 0.0], vec![1.0, 0.0]];

        // Implicit tableau (L-stable)
        let gamma = 1.0 - 1.0 / 2.0_f64.sqrt();
        let a_implicit = vec![vec![gamma, 0.0], vec![2.0f64.mul_add(-gamma, 1.0), gamma]];

        let b = vec![0.5, 0.5];
        let c = vec![gamma, 1.0 - gamma];

        (a_explicit, a_implicit, b, c, 2, 2)
    }

    /// SSP3(3,3,3) coefficients
    fn ssp3_333_coefficients() -> (
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        usize,
        usize,
    ) {
        // Explicit tableau
        let a_explicit = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];

        // Implicit tableau
        let gamma = 0.4358665215;
        let a_implicit = vec![
            vec![gamma, 0.0, 0.0],
            vec![0.3212788860, gamma, 0.0],
            vec![0.1058582961, 0.3586522499, gamma],
        ];

        let b = vec![1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0];
        let c = vec![gamma, 0.7571053801, 1.0];

        (a_explicit, a_implicit, b, c, 3, 3)
    }

    /// ARK3 coefficients (3rd order)
    fn ark3_coefficients() -> (
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        usize,
        usize,
    ) {
        let a_explicit = Self::ark3_explicit_coefficients();
        let a_implicit = Self::ark3_implicit_coefficients();
        let b = Self::ark3_b_coefficients();
        let c = Self::ark3_c_coefficients();

        (a_explicit, a_implicit, b, c, 4, 3)
    }

    /// Explicit coefficient matrix for ARK3
    fn ark3_explicit_coefficients() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![1767732205903.0 / 2027836641118.0, 0.0, 0.0, 0.0],
            vec![
                5535828885825.0 / 10492691773637.0,
                788022342437.0 / 10882634858940.0,
                0.0,
                0.0,
            ],
            vec![
                6485989280629.0 / 16251701735622.0,
                -4246266847089.0 / 9704473918619.0,
                10755448449292.0 / 10357097424841.0,
                0.0,
            ],
        ]
    }

    /// Implicit coefficient matrix for ARK3
    fn ark3_implicit_coefficients() -> Vec<Vec<f64>> {
        vec![
            vec![0.0, 0.0, 0.0, 0.0],
            vec![
                1767732205903.0 / 4055673282236.0,
                1767732205903.0 / 4055673282236.0,
                0.0,
                0.0,
            ],
            vec![
                2746238789719.0 / 10658868560708.0,
                -640167445237.0 / 6845629431997.0,
                1767732205903.0 / 4055673282236.0,
                0.0,
            ],
            vec![
                1471266399579.0 / 7840856788654.0,
                -4482444167858.0 / 7529755066697.0,
                11266239266428.0 / 11593286722821.0,
                1767732205903.0 / 4055673282236.0,
            ],
        ]
    }

    /// B coefficients for ARK3
    fn ark3_b_coefficients() -> Vec<f64> {
        vec![
            1471266399579.0 / 7840856788654.0,
            -4482444167858.0 / 7529755066697.0,
            11266239266428.0 / 11593286722821.0,
            1767732205903.0 / 4055673282236.0,
        ]
    }

    /// C coefficients for ARK3
    fn ark3_c_coefficients() -> Vec<f64> {
        vec![0.0, 1767732205903.0 / 2027836641118.0, 3.0 / 5.0, 1.0]
    }

    /// ARK4 coefficients (4th order)
    fn ark4_coefficients() -> (
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        usize,
        usize,
    ) {
        // Using Kennedy-Carpenter ARK4(3)6L[2]SA coefficients
        // This is a 6-stage, 4th order, L-stable scheme
        let a_explicit = Self::ark4_explicit_coefficients();
        let a_implicit = Self::ark4_implicit_coefficients();
        let b = Self::ark4_b_coefficients();
        let c = Self::ark4_c_coefficients();

        (a_explicit, a_implicit, b, c, 6, 4)
    }

    /// Explicit coefficient matrix for ARK4
    fn ark4_explicit_coefficients() -> Vec<Vec<f64>> {
        vec![
            vec![0.0; 6],
            vec![1.0 / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![13861.0 / 62500.0, 6889.0 / 62500.0, 0.0, 0.0, 0.0, 0.0],
            vec![
                -116923316275.0 / 2393684061468.0,
                -2731218467317.0 / 15368042101831.0,
                9408046702089.0 / 11113171139209.0,
                0.0,
                0.0,
                0.0,
            ],
            vec![
                -451086348788.0 / 2902428689909.0,
                -2682348792572.0 / 7519795681897.0,
                12662868775082.0 / 11960479115383.0,
                3355817975965.0 / 11060851509271.0,
                0.0,
                0.0,
            ],
            vec![
                647845179188.0 / 3216320057751.0,
                73281519250.0 / 8382639484533.0,
                552539513391.0 / 3454668386233.0,
                3354512671639.0 / 8306763924573.0,
                4040.0 / 17871.0,
                0.0,
            ],
        ]
    }

    /// Implicit coefficient matrix for ARK4
    fn ark4_implicit_coefficients() -> Vec<Vec<f64>> {
        vec![
            vec![0.0; 6],
            vec![1.0 / 4.0, 1.0 / 4.0, 0.0, 0.0, 0.0, 0.0],
            vec![
                8611.0 / 62500.0,
                -1743.0 / 31250.0,
                1.0 / 4.0,
                0.0,
                0.0,
                0.0,
            ],
            vec![
                5012029.0 / 34652500.0,
                -654441.0 / 2922500.0,
                174375.0 / 388108.0,
                1.0 / 4.0,
                0.0,
                0.0,
            ],
            vec![
                15267082809.0 / 155376265600.0,
                -71443401.0 / 120774400.0,
                730878875.0 / 902184768.0,
                2285395.0 / 8070912.0,
                1.0 / 4.0,
                0.0,
            ],
            vec![
                82889.0 / 524892.0,
                0.0,
                15625.0 / 83664.0,
                69875.0 / 102672.0,
                -2260.0 / 8211.0,
                1.0 / 4.0,
            ],
        ]
    }

    /// B coefficients for ARK4
    fn ark4_b_coefficients() -> Vec<f64> {
        vec![
            82889.0 / 524892.0,
            0.0,
            15625.0 / 83664.0,
            69875.0 / 102672.0,
            -2260.0 / 8211.0,
            1.0 / 4.0,
        ]
    }

    /// C coefficients for ARK4
    fn ark4_c_coefficients() -> Vec<f64> {
        vec![0.0, 1.0 / 2.0, 83.0 / 250.0, 31.0 / 50.0, 17.0 / 20.0, 1.0]
    }
}
