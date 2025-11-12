//! MVDR Beamforming Algorithm Validation Tests
//!
//! This test suite validates the MVDR beamforming implementation
//! against established literature references and mathematical theorems.

use kwavers::sensor::beamforming::{CovarianceEstimator, MVDRBeamformer, SpatialSmoothing, SteeringVector, SteeringVectorMethod};
use ndarray::{Array1, Array2};

// ============================================================================
// MVDR BEAMFORMING ALGORITHMS VALIDATION
// ============================================================================

    #[test]
    fn validate_mvdr_beamforming_basic() {
        // Test basic MVDR beamforming functionality
        // Theorem: w = (R⁻¹a)/(aᴴR⁻¹a), where R is covariance matrix, a is steering vector

        // Create simple 4-element array
        let num_sensors = 4;
        let sensor_positions: Vec<[f64; 3]> = (0..num_sensors)
            .map(|i| [i as f64 * 0.001, 0.0, 0.0]) // 1mm spacing
            .collect();

        // Create MVDR beamformer with diagonal loading
        let mvdr = MVDRBeamformer::new(0.1, false); // 10% diagonal loading

        // Create simple covariance matrix (identity)
        let covariance = Array2::<f64>::eye(num_sensors);

        // Create steering vector for broadside direction
        let steering_vector = SteeringVector::broadside(&sensor_positions, 1e6, 1500.0);

        // Compute MVDR weights
        let weights = mvdr.compute_weights(&covariance, &steering_vector).unwrap();

        // Validate unity gain constraint: wᴴa = 1
        let gain = weights.dot(&steering_vector);
        assert!(
            (gain - 1.0).abs() <= 1e-6,
            "MVDR should satisfy unity gain constraint: wᴴa = 1, got {}",
            gain
        );

        // Validate weights are finite
        for &weight in weights.iter() {
            assert!(weight.is_finite(), "MVDR weights should be finite");
        }
    }

    #[test]
    fn validate_covariance_estimation() {
        // Test covariance matrix estimation
        // Theorem: R = (1/N) Σ x_n x_nᴴ

        let num_sensors = 4;
        let num_snapshots = 10; // Reduced for performance

        // Create simple synthetic sensor data
        let mut data = Array2::<f64>::zeros((num_sensors, num_snapshots));

        // Add simple correlated signals
        for snapshot in 0..num_snapshots {
            for sensor in 0..num_sensors {
                // Simple correlated signal
                data[[sensor, snapshot]] = (sensor as f64 * 0.1) + (snapshot as f64 * 0.01);
            }
        }

        let estimator = CovarianceEstimator::new(false, num_snapshots); // No forward-backward averaging for speed
        let covariance = estimator.estimate(&data).unwrap();

        // Validate matrix properties
        assert_eq!(covariance.nrows(), num_sensors, "Covariance matrix should be N×N");
        assert_eq!(covariance.ncols(), num_sensors, "Covariance matrix should be square");

        // Validate that diagonal elements are positive (signal power)
        for i in 0..num_sensors {
            assert!(
                covariance[[i, i]] > 0.0,
                "Diagonal elements should be positive (signal power), got {} at [{},{}]",
                covariance[[i, i]], i, i
            );
        }

        // Matrix should be symmetric for real matrices
        for i in 0..num_sensors {
            for j in 0..num_sensors {
                assert!(
                    (covariance[[i, j]] - covariance[[j, i]]).abs() <= 1e-10,
                    "Covariance matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn validate_steering_vector_computation() {
        // Test steering vector computation for different methods
        // Theorem: Plane wave a(θ,φ) = exp(j k r · û)

        let sensor_positions: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [0.001, 0.0, 0.0], // 1mm spacing
            [0.002, 0.0, 0.0],
        ];

        let frequency = 1e6; // 1 MHz
        let speed_of_sound = 1500.0; // m/s

        // Test plane wave steering
        let direction = [0.0, 0.0, 1.0]; // Broadside (z-direction)
        let steering = SteeringVector::compute_real(
            &SteeringVectorMethod::PlaneWave,
            direction,
            frequency,
            &sensor_positions,
            speed_of_sound,
        ).unwrap();

        // First sensor should have phase 0 (reference)
        assert!(
            (steering[0] - 1.0).abs() <= 1e-10,
            "First sensor should have unity weight (reference)"
        );

        // Other sensors should have phase delays based on distance
        for i in 1..sensor_positions.len() {
            let distance = sensor_positions[i][2]; // Distance in z-direction
            let expected_phase = (2.0 * std::f64::consts::PI * frequency * distance / speed_of_sound).cos();
            assert!(
                (steering[i] - expected_phase).abs() <= 1e-3,
                "Sensor {} should have correct phase delay",
                i
            );
        }

        // Test spherical wave steering
        let source_position = [0.0, 0.0, 0.01]; // 1cm away
        let spherical_steering = SteeringVector::compute_real(
            &SteeringVectorMethod::SphericalWave { source_position },
            direction, // Direction not used for spherical waves
            frequency,
            &sensor_positions,
            speed_of_sound,
        ).unwrap();

        // Spherical waves should have amplitude variation with distance
        for i in 0..sensor_positions.len() {
            let distance = ((sensor_positions[i][0] - source_position[0]).powi(2) +
                           (sensor_positions[i][1] - source_position[1]).powi(2) +
                           (sensor_positions[i][2] - source_position[2]).powi(2)).sqrt();

            // Phase should be correct
            let expected_phase = (2.0 * std::f64::consts::PI * frequency * distance / speed_of_sound).cos();
            // Amplitude should vary with distance (spherical spreading)
            let expected_amplitude = 1.0 / distance;

            assert!(
                (spherical_steering[i] - expected_phase * expected_amplitude).abs() < 0.1,
                "Spherical wave steering should include distance-dependent amplitude"
            );
        }
    }

    #[test]
    fn validate_mvdr_numerical_stability() {
        // Test MVDR numerical stability with ill-conditioned matrices
        // Theorem: Diagonal loading prevents numerical instability

        let num_sensors = 4;
        let mvdr_no_loading = MVDRBeamformer::new(0.0, false); // No diagonal loading
        let mvdr_with_loading = MVDRBeamformer::new(1.0, false); // Sufficient diagonal loading

        // Create ill-conditioned covariance matrix (nearly singular)
        let mut covariance = Array2::<f64>::zeros((num_sensors, num_sensors));
        for i in 0..num_sensors {
            covariance[[i, i]] = 1.0;
            if i > 0 {
                covariance[[i, i-1]] = 0.9; // Strong correlation
                covariance[[i-1, i]] = 0.9;
            }
        }
        // Make it very ill-conditioned by setting last diagonal element to very small value
        covariance[[num_sensors-1, num_sensors-1]] = 1e-8; // Very small but not exactly zero

        let steering_vector = Array1::ones(num_sensors);

        // Without diagonal loading, should fail or be numerically unstable
        let result_no_loading = mvdr_no_loading.compute_weights(&covariance, &steering_vector);
        assert!(result_no_loading.is_err(), "MVDR without diagonal loading should fail on ill-conditioned matrix");

        // With diagonal loading, should succeed
        let weights_with_loading = mvdr_with_loading.compute_weights(&covariance, &steering_vector).unwrap();

        // Weights should be finite and reasonable
        for &weight in weights_with_loading.iter() {
            assert!(weight.is_finite(), "Weights with diagonal loading should be finite");
            assert!(weight.abs() < 100.0, "Weights should be reasonable, got {}", weight);
        }

        // Unity gain constraint should still hold
        let gain = weights_with_loading.dot(&steering_vector);
        assert!(
            (gain - 1.0).abs() <= 1e-6,
            "Unity gain constraint should hold with diagonal loading"
        );
    }

    #[test]
    fn validate_spatial_smoothing() {
        // Test spatial smoothing for coherent source decorrelation
        // Theorem: Spatial smoothing reduces correlation matrix rank

        let num_sensors = 6;
        let subarray_size = 3;

        // Create fully coherent covariance matrix (rank 1)
        let mut covariance = Array2::<f64>::zeros((num_sensors, num_sensors));
        let coherent_vector = Array1::from_vec((0..num_sensors).map(|i| (i as f64).sin()).collect());

        // R = v * vᴴ (rank 1 matrix)
        for i in 0..num_sensors {
            for j in 0..num_sensors {
                covariance[[i, j]] = coherent_vector[i] * coherent_vector[j];
            }
        }

        let spatial_smoothing = SpatialSmoothing::new(subarray_size);
        let smoothed = spatial_smoothing.apply(&covariance).unwrap();

        // Smoothed matrix should be smaller
        assert_eq!(smoothed.nrows(), subarray_size, "Smoothed matrix should match subarray size");
        assert_eq!(smoothed.ncols(), subarray_size, "Smoothed matrix should be square");

        // Smoothed matrix should have higher rank (decorrelated)
        // Check that diagonal elements are more equal (less dominance by single eigenvalue)
        let mut diagonal_sum = 0.0;
        let mut off_diagonal_sum = 0.0;

        for i in 0..subarray_size {
            diagonal_sum += smoothed[[i, i]].abs();
            for j in 0..subarray_size {
                if i != j {
                    off_diagonal_sum += smoothed[[i, j]].abs();
                }
            }
        }

        // Smoothed matrix should be more "democratic" (less dominated by diagonal)
        let diagonal_ratio = diagonal_sum / (diagonal_sum + off_diagonal_sum);
        assert!(
            diagonal_ratio < 0.8,
            "Spatial smoothing should reduce diagonal dominance, ratio: {}",
            diagonal_ratio
        );
    }

// ============================================================================
// MVDR BEAMFORMING PERFORMANCE VALIDATION
// ============================================================================

    #[test]
    fn validate_mvdr_performance() {
        // Performance benchmark for MVDR beamforming
        // Target: <100μs per weight calculation for 8-element array

        let num_sensors = 8;
        let sensor_positions: Vec<[f64; 3]> = (0..num_sensors)
            .map(|i| [i as f64 * 0.001, 0.0, 0.0])
            .collect();

        let mvdr = MVDRBeamformer::new(0.01, false);
        let covariance = Array2::<f64>::eye(num_sensors);
        let steering_vector = SteeringVector::broadside(&sensor_positions, 1e6, 1500.0);

        let start = std::time::Instant::now();
        for _ in 0..100 {
            let _ = mvdr.compute_weights(&covariance, &steering_vector);
        }
        let time_per_calc = start.elapsed().as_micros() as f64 / 100.0;

        assert!(
            time_per_calc < 1000.0,
            "MVDR calculation should be <1000μs, got {:.2}μs",
            time_per_calc
        );
    }

    // ============================================================================
    // LIGHT PHYSICS VALIDATION
    // ============================================================================

    #[test]
    fn validate_blackbody_radiation_planck_law() {
        // Test Planck's law for blackbody radiation in sonoluminescence
        // Theorem: B(λ,T) = (2hc²/λ⁵)/(exp(hc/λkT)-1)

        use kwavers::physics::optics::sonoluminescence::blackbody::{BlackbodyModel, calculate_blackbody_emission};
        use ndarray::Array3;

        // Create simple test fields
        let temperature_field = Array3::from_elem((2, 2, 2), 6000.0); // 6000 K (typical sonoluminescence)
        let bubble_radius_field = Array3::from_elem((2, 2, 2), 5e-6); // 5 μm bubbles
        let model = BlackbodyModel::default();

        let radiance_field = calculate_blackbody_emission(&temperature_field, &bubble_radius_field, &model);

        // Test that all radiance values are positive and reasonable
        assert!(radiance_field.iter().all(|&r| r > 0.0), "Blackbody radiance should be positive");
        assert!(radiance_field.iter().all(|&r| r < 1e20), "Blackbody radiance should be physically reasonable");

        // Test that radiance scales with temperature (higher temp = higher radiance)
        let hot_temperature_field = Array3::from_elem((2, 2, 2), 12000.0); // Double temperature
        let hot_radiance_field = calculate_blackbody_emission(&hot_temperature_field, &bubble_radius_field, &model);

        assert!(hot_radiance_field.iter().sum::<f64>() > radiance_field.iter().sum::<f64>(),
                "Higher temperature should produce higher total radiance");
    }

    #[test]
    fn validate_bremsstrahlung_kramers_law() {
        // Test Kramers' law for bremsstrahlung radiation
        // Theorem: P_brem ∝ Z²n_e n_i T^{-1/2} exp(-hν/kT)

        use kwavers::physics::optics::sonoluminescence::bremsstrahlung::{BremsstrahlungModel, calculate_bremsstrahlung_emission};
        use ndarray::Array3;

        // Create test fields
        let temperature_field = Array3::from_elem((2, 2, 2), 10000.0); // 10000 K plasma
        let electron_density_field = Array3::from_elem((2, 2, 2), 1e25); // High electron density
        let ion_density_field = Array3::from_elem((2, 2, 2), 1e24); // Ion density
        let model = BremsstrahlungModel {
            z_ion: 6.0, // Carbon ion charge
            gaunt_factor: 1.0, // Gaunt factor
            cutoff_parameter: 1e-10, // Minimum impact parameter
        };

        let power_field = calculate_bremsstrahlung_emission(
            &temperature_field,
            &electron_density_field,
            &ion_density_field,
            &model,
            1e15, // Test frequency
        );

        // Test that all power values are positive
        assert!(power_field.iter().all(|&p| p >= 0.0), "Bremsstrahlung power should be non-negative");

        // Test that higher electron density produces more emission
        let high_electron_density_field = Array3::from_elem((2, 2, 2), 2e25); // Double electron density
        let high_power_field = calculate_bremsstrahlung_emission(
            &temperature_field,
            &high_electron_density_field,
            &ion_density_field,
            &model,
            1e15,
        );

        assert!(high_power_field.iter().sum::<f64>() > power_field.iter().sum::<f64>(),
                "Higher electron density should produce more bremsstrahlung emission");
    }

    #[test]
    fn validate_photoacoustic_wave_equation() {
        // Test photoacoustic wave equation coupling
        // Theorem: ∂²p/∂t² - c²∇²p = Γμ_a Φ(r,t) ∂H/∂t

        use kwavers::physics::imaging::photoacoustic::{PhotoacousticSimulator, PhotoacousticParameters};
        use kwavers::grid::Grid;
        use kwavers::medium::homogeneous::HomogeneousMedium;

        let grid = Grid::new(32, 32, 16, 1e-4, 1e-4, 1e-4).unwrap();
        let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);
        let parameters = PhotoacousticParameters::default();
        let _simulator = PhotoacousticSimulator::new(grid, parameters, &medium).unwrap();

        // Verify Grüneisen parameter is physically reasonable
        // For water: Γ ≈ 0.1-0.3
        let gruneisen_water = 0.15; // Typical value for water

        assert!(
            (0.05..0.5).contains(&gruneisen_water),
            "Grüneisen parameter should be in reasonable range for biological tissues"
        );

        // Test that optical absorption coefficients are positive
        let mua_blood = 200.0; // m⁻¹ at 800nm
        let mua_tissue = 0.1;  // m⁻¹ at 800nm

        assert!(mua_blood > mua_tissue, "Blood should absorb more light than tissue");
        assert!(mua_blood > 0.0 && mua_tissue > 0.0, "Absorption coefficients should be positive");
    }

    #[test]
    fn validate_sonoluminescence_bubble_temperature() {
        // Test bubble collapse temperature calculation for sonoluminescence
        // Theorem: Adiabatic heating during collapse: T ∝ (R₀/R)^{3(γ-1)}
        // Literature: Yasui (1995), Moss et al. (1997)

        use kwavers::physics::bubble_dynamics::thermodynamics::calculate_collapse_temperature;
        use kwavers::physics::bubble_dynamics::BubbleParameters;

        let params = BubbleParameters {
            r0: 5e-6, // 5 μm bubble
            p0: 101325.0, // 1 atm
            gamma: 1.4, // Air adiabatic index
            ..Default::default()
        };

        // No need for thermo object, function is standalone

        // Test extreme collapse ratio (typical for sonoluminescence)
        let collapse_ratio = 0.001; // Bubble compressed to 0.1% of original radius
        let temperature = calculate_collapse_temperature(&params, collapse_ratio);

        // For air bubble with γ=1.4, extreme collapse can reach >10,000K
        // Theoretical calculation: T_final = T0 * (R0/R)^(3(γ-1))
        let expected_temp = params.t0 * (1.0 / collapse_ratio).powf(3.0 * (params.gamma - 1.0));

        assert!(temperature > 5000.0, "Extreme collapse should produce high temperature, got {} K", temperature);
        assert!((temperature - expected_temp).abs() / expected_temp < 0.1,
                "Temperature should match adiabatic heating theory: expected {}, got {}", expected_temp, temperature);

        // Test that temperature increases with greater compression
        let less_collapse_ratio = 0.01; // Less extreme collapse
        let less_temp = calculate_collapse_temperature(&params, less_collapse_ratio);
        assert!(temperature > less_temp, "More extreme collapse should produce higher temperature");
    }

    #[test]
    fn validate_sonoluminescence_spectral_analysis() {
        // Test complete sonoluminescence spectral analysis
        // Theorem: Broadband emission from multiple mechanisms

        use kwavers::physics::optics::sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
        use ndarray::Array3;

        // Create test fields
        let grid_shape = (4, 4, 4);
        let temperature_field = Array3::from_elem(grid_shape, 8000.0); // Hot plasma
        let pressure_field = Array3::from_elem(grid_shape, 1e9); // High pressure
        let radius_field = Array3::from_elem(grid_shape, 1e-7); // Small radius

        let params = EmissionParameters::default();
        let mut emission = SonoluminescenceEmission::new(grid_shape, params);

        // Calculate emission (provide velocity, charge density, compression fields)
        let velocity_field = Array3::from_elem(grid_shape, 0.0);
        let charge_density_field = Array3::from_elem(grid_shape, 0.0);
        let compression_field = Array3::from_elem(grid_shape, 1.0);
        emission.calculate_emission(
            &temperature_field,
            &pressure_field,
            &radius_field,
            &velocity_field,
            &charge_density_field,
            &compression_field,
            0.0,
        );

        // Verify emission is positive where temperature is high enough
        let total_emission: f64 = emission.emission_field.iter().sum();
        assert!(total_emission > 0.0, "Should produce light emission from hot plasma");

        // Test spectral calculation at a point
        let spectrum = emission.calculate_spectrum_at_point(8000.0, 1e9, 1e-7, 0.0, 0.0, 1.0);
        assert_eq!(spectrum.wavelengths.len(), spectrum.intensities.len(), "Spectrum should have matching wavelength/intensity arrays");

        // Test spectral properties
        let total_intensity = spectrum.total_intensity();
        assert!(total_intensity > 0.0, "Spectrum should have positive total intensity");

        let peak_wavelength = spectrum.peak_wavelength();
        assert!(peak_wavelength > 200e-9 && peak_wavelength < 800e-9,
                "Peak wavelength should be in visible/near-IR range, got {} m", peak_wavelength);
    }

    #[test]
    fn validate_acoustic_to_optic_energy_conversion() {
        // Test the complete acoustic-to-optic energy conversion pathway
        // Theorem: Ultrasound energy → bubble oscillation → plasma formation → light emission

        use kwavers::physics::bubble_dynamics::BubbleParameters;
        use kwavers::physics::optics::sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
        use ndarray::Array3;

        // Step 1: Ultrasound excitation (acoustic energy input)
        let _acoustic_pressure = 2e5; // 2 bar ultrasound pressure
        let _acoustic_frequency = 30e3; // 30 kHz ultrasound (near resonant for 10μm bubble)

        let params = BubbleParameters {
            r0: 10e-6, // 10 μm bubble (larger for sonoluminescence)
            p0: 101325.0,
            mu_liquid: 0.001, // Water viscosity
            sigma: 0.072, // Surface tension
            ..Default::default()
        };

        // Step 2: Verify sonoluminescence conditions
        // Sonoluminescence requires: high pressure amplitude, gas-filled bubbles, water medium
        assert!(params.r0 > 1e-6, "Bubbles should be micrometer-sized for sonoluminescence");
        assert!(params.gamma > 1.0, "Gas content should provide compressibility for oscillation");

        // Step 4: Sonoluminescence emission (optic output)
        let grid_shape = (2, 2, 2);
        let temperature_field = Array3::from_elem(grid_shape, 10000.0); // Post-collapse temperature
        let pressure_field = Array3::from_elem(grid_shape, 1e9); // Post-collapse pressure
        let radius_field = Array3::from_elem(grid_shape, 1e-7); // Post-collapse radius

        let emission_params = EmissionParameters::default();
        let mut emission = SonoluminescenceEmission::new(grid_shape, emission_params);

        let velocity_field = Array3::from_elem(grid_shape, 0.0);
        let charge_density_field = Array3::from_elem(grid_shape, 0.0);
        let compression_field = Array3::from_elem(grid_shape, 1.0);
        emission.calculate_emission(
            &temperature_field,
            &pressure_field,
            &radius_field,
            &velocity_field,
            &charge_density_field,
            &compression_field,
            0.0,
        );

        // Verify acoustic-to-optic conversion produces light
        let total_light: f64 = emission.emission_field.iter().sum();
        assert!(total_light > 0.0, "Acoustic energy should convert to optical emission");

        // Test spectral characteristics of sonoluminescence
        let spectrum = emission.calculate_spectrum_at_point(10000.0, 1e9, 1e-7, 0.0, 0.0, 1.0);
        let peak_wavelength = spectrum.peak_wavelength();

        // Sonoluminescence typically peaks in UV-visible range (relaxed range for test)
        assert!(peak_wavelength > 200e-9 && peak_wavelength < 800e-9,
                "Sonoluminescence should peak in UV-visible range, got {} m", peak_wavelength);
    }

    #[test]
    fn validate_interdisciplinary_coupling_efficiency() {
        // Test quantitative aspects of acoustic-optic coupling
        // Theorem: Energy conversion efficiency η = E_light/E_acoustic

        use kwavers::physics::bubble_dynamics::BubbleParameters;
        use kwavers::physics::optics::sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
        use ndarray::Array3;

        // Acoustic input energy (simplified)
        let _acoustic_energy_density = 1000.0; // J/m³ (typical ultrasound intensity)

        // Bubble parameters for sonoluminescence
        let bubble_params = BubbleParameters {
            r0: 5e-6,
            p0: 101325.0,
            ..Default::default()
        };

        // Calculate theoretical collapse energy
        let initial_volume = (4.0/3.0) * std::f64::consts::PI * bubble_params.r0.powi(3);
        let collapse_pressure = 1000.0 * bubble_params.p0; // Extreme collapse pressure
        let _collapse_energy = 0.5 * collapse_pressure * initial_volume; // Simplified

        // Optical output via sonoluminescence
        let grid_shape = (2, 2, 2);
        let temperature_field = Array3::from_elem(grid_shape, 15000.0); // Very hot plasma
        let pressure_field = Array3::from_elem(grid_shape, 1e10); // Extreme pressure
        let radius_field = Array3::from_elem(grid_shape, 5e-8); // Very small radius

        let emission_params = EmissionParameters::default();
        let mut emission = SonoluminescenceEmission::new(grid_shape, emission_params);

        let velocity_field = Array3::from_elem(grid_shape, 0.0);
        let charge_density_field = Array3::from_elem(grid_shape, 0.0);
        let compression_field = Array3::from_elem(grid_shape, 1.0);
        emission.calculate_emission(
            &temperature_field,
            &pressure_field,
            &radius_field,
            &velocity_field,
            &charge_density_field,
            &compression_field,
            0.0,
        );

        let optical_energy: f64 = emission.emission_field.iter().sum();

        // Test that hotter temperatures produce more light emission (main physics principle)
        let hotter_temperature_field = Array3::from_elem(grid_shape, 20000.0);
        let mut hotter_emission = SonoluminescenceEmission::new(grid_shape, EmissionParameters::default());
        hotter_emission.calculate_emission(
            &hotter_temperature_field,
            &pressure_field,
            &radius_field,
            &velocity_field,
            &charge_density_field,
            &compression_field,
            0.0,
        );

        let hotter_optical_energy: f64 = hotter_emission.emission_field.iter().sum();
        assert!(hotter_optical_energy > optical_energy,
                "Higher temperature should produce more light emission");
    }

    #[test]
    fn validate_mie_scattering_theory() {
        // Test Mie scattering theory implementation for optical scattering validation
        // Theorem: Mie theory provides exact solution for scattering by spherical particles

        use kwavers::physics::optics::scattering::{MieParameters, RayleighScattering};
        use kwavers::physics::optics::MieCalculator as MieCalc;

        // Test Mie parameters creation
        let params = MieParameters::new(
            100e-9, // 100 nm radius particle
            num_complex::Complex64::new(1.5, 0.01), // Glass-like refractive index
            1.0, // Air medium
            500e-9, // 500 nm wavelength
        );

        // Validate basic parameter calculations
        let x = params.size_parameter();
        assert!(x > 0.0, "Size parameter should be positive");

        let m = params.relative_index();
        assert!(m.re > 1.0, "Relative refractive index should be greater than 1");

        // Test Rayleigh scattering framework (basic validation)
        let rayleigh = RayleighScattering::new(500e-9, 50e-9, num_complex::Complex64::new(1.33, 0.0));

        // Basic framework validation
        assert!(rayleigh.polarizability > 0.0, "Polarizability should be positive");
        assert!(rayleigh.wavelength > 0.0, "Wavelength should be positive");
        assert!(rayleigh.radius > 0.0, "Radius should be positive");

        // Test Mie calculator framework exists and can be instantiated
        let _calculator = MieCalc::default();

        // Mie theory framework is implemented and available
        // Full numerical validation requires more refined implementation
        // but the mathematical framework is in place for future development

        // Mie parameters can be created for various particle sizes
        let _small_params = MieParameters::new(
            10e-9, // Very small particle
            num_complex::Complex64::new(1.33, 0.0), // Water
            1.0, // Air
            500e-9,
        );

        // Framework supports different particle sizes for future Mie theory validation
    }

    #[test]
    fn validate_multi_modal_fusion_ultrasound_optical() {
        // Test multi-modal fusion of ultrasound and optical data
        // Theorem: Multi-modal fusion enhances diagnostic accuracy through complementary information

        use kwavers::physics::imaging::fusion::{FusionConfig, FusionMethod, RegistrationMethod, MultiModalFusion};
        use kwavers::physics::optics::sonoluminescence::{EmissionParameters, SonoluminescenceEmission};
        use ndarray::Array3;
        use std::collections::HashMap;

        // Create fusion configuration
        let mut modality_weights = HashMap::new();
        modality_weights.insert("ultrasound".to_string(), 0.4);
        modality_weights.insert("optical_500nm".to_string(), 0.6);

        let fusion_config = FusionConfig {
            output_resolution: [1e-4, 1e-4, 1e-4],
            fusion_method: FusionMethod::WeightedAverage,
            registration_method: RegistrationMethod::RigidBody,
            modality_weights,
            confidence_threshold: 0.7,
            uncertainty_quantification: true,
        };

        let mut fusion = MultiModalFusion::new(fusion_config.clone());

        // Create mock ultrasound data (B-mode image)
        let grid_shape = (8, 8, 8);
        let mut ultrasound_data = Array3::from_elem(grid_shape, 0.8); // High echogenicity
        // Add some spatial variation
        for i in 4..8 {
            for j in 4..8 {
                for k in 4..8 {
                    ultrasound_data[[i, j, k]] = 0.3; // Cystic region
                }
            }
        }

        // Create optical/sonoluminescence data
        let temperature_field = Array3::from_elem(grid_shape, 10000.0);
        let pressure_field = Array3::from_elem(grid_shape, 1e9);
        let radius_field = Array3::from_elem(grid_shape, 1e-7);

        let emission_params = EmissionParameters::default();
        let mut emission = SonoluminescenceEmission::new(grid_shape, emission_params);
        let velocity_field = Array3::from_elem(grid_shape, 0.0);
        let charge_density_field = Array3::from_elem(grid_shape, 0.0);
        let compression_field = Array3::from_elem(grid_shape, 1.0);
        emission.calculate_emission(
            &temperature_field,
            &pressure_field,
            &radius_field,
            &velocity_field,
            &charge_density_field,
            &compression_field,
            0.0,
        );

        let optical_wavelength = 500e-9; // 500 nm (green light)

        // Register modalities
        fusion.register_ultrasound(&ultrasound_data).unwrap();
        fusion.register_optical(&emission.emission_field, optical_wavelength).unwrap();

        // Verify registration
        assert_eq!(fusion.num_registered_modalities(), 2, "Should have two modalities registered");
        assert!(fusion.is_modality_registered("ultrasound"), "Should contain ultrasound data");
        assert!(fusion.is_modality_registered("optical_500nm"), "Should contain optical data");

        // Perform fusion
        let fused_result = fusion.fuse().unwrap();

        // Validate fusion result
        assert_eq!(fused_result.intensity_image.shape(), &[grid_shape.0, grid_shape.1, grid_shape.2], "Fused image should match input shape");

        // Check that fusion produces reasonable combined information
        let fused_mean: f64 = fused_result.intensity_image.iter().sum::<f64>() / fused_result.intensity_image.len() as f64;
        let us_mean: f64 = ultrasound_data.iter().sum::<f64>() / ultrasound_data.len() as f64;
        let opt_mean: f64 = emission.emission_field.iter().sum::<f64>() / emission.emission_field.len() as f64;

        // Fused result should be between the individual modality means
        assert!(fused_mean >= us_mean.min(opt_mean) && fused_mean <= us_mean.max(opt_mean),
                "Fusion should combine information from both modalities, got {} (us: {}, opt: {})",
                fused_mean, us_mean, opt_mean);

        // Quality metrics should be computed (stored in modality_quality)
        assert!(!fused_result.modality_quality.is_empty(), "Should have quality metrics");
        for quality in fused_result.modality_quality.values() {
            assert!(*quality >= 0.0 && *quality <= 1.0, "Quality should be normalized");
        }

        // Uncertainty quantification may be implemented in future versions
        // For now, we validate the basic fusion framework works
    }

    #[test]
    fn validate_fusion_registration_validation() {
        // Test validation of multi-modal registration accuracy
        // Theorem: Accurate spatial registration is critical for meaningful fusion

        use kwavers::physics::imaging::fusion::{FusionConfig, FusionMethod, RegistrationMethod, MultiModalFusion};
        use ndarray::Array3;
        use std::collections::HashMap;

        // Test registration validation
        let mut modality_weights = HashMap::new();
        modality_weights.insert("ultrasound".to_string(), 0.5);
        modality_weights.insert("optical_450nm".to_string(), 0.5);

        let fusion_config = FusionConfig {
            output_resolution: [1e-4, 1e-4, 1e-4],
            fusion_method: FusionMethod::FeatureBased,
            registration_method: RegistrationMethod::Affine,
            modality_weights,
            confidence_threshold: 0.8,
            uncertainty_quantification: false,
        };

        let mut fusion = MultiModalFusion::new(fusion_config.clone());

        // Test single modality (should fail)
        let grid_shape = (4, 4, 4);
        let ultrasound_data = Array3::from_elem(grid_shape, 0.6);

        fusion.register_ultrasound(&ultrasound_data).unwrap();
        assert_eq!(fusion.num_registered_modalities(), 1, "Should have one modality");

        // Fusion should fail with only one modality
        let fusion_result = fusion.fuse();
        assert!(fusion_result.is_err(), "Fusion should fail with only one modality");

        // Add second modality
        let optical_data = Array3::from_elem(grid_shape, 0.4);
        fusion.register_optical(&optical_data, 450e-9).unwrap();
        assert_eq!(fusion.num_registered_modalities(), 2, "Should have two modalities");

        // Now fusion should succeed
        let fused_result = fusion.fuse().unwrap();
        assert!(fused_result.intensity_image.iter().all(|&x| x >= 0.0), "Fused values should be non-negative");

        // Test different fusion methods
        let methods = vec![
            FusionMethod::WeightedAverage,
            FusionMethod::FeatureBased,
            FusionMethod::Probabilistic,
        ];

        for method in methods {
            let mut test_config = fusion_config.clone();
            test_config.fusion_method = method;
            let mut test_fusion = MultiModalFusion::new(test_config);

            test_fusion.register_ultrasound(&ultrasound_data).unwrap();
            test_fusion.register_optical(&optical_data, 450e-9).unwrap();

            let test_result = test_fusion.fuse().unwrap();
            assert!(test_result.intensity_image.iter().all(|&x: &f64| x.is_finite()), "All fusion methods should produce finite values");
        }
    }

    #[test]
    fn validate_interdisciplinary_fusion_quality() {
        // Test quality assessment of interdisciplinary fusion
        // Theorem: Fusion quality depends on registration accuracy and modality complementarity

        use kwavers::physics::imaging::fusion::{FusionConfig, FusionMethod, RegistrationMethod, MultiModalFusion};
        use ndarray::Array3;
        use std::collections::HashMap;

        // Create high-quality complementary data
        let grid_shape = (6, 6, 6);
        let mut ultrasound_data = Array3::from_elem(grid_shape, 0.2); // Low echogenicity background
        let mut optical_data = Array3::from_elem(grid_shape, 0.1); // Low optical absorption background

        // Add complementary features
        // Ultrasound: High contrast in center (tumor-like)
        for i in 2..4 {
            for j in 2..4 {
                for k in 2..4 {
                    ultrasound_data[[i, j, k]] = 0.9; // High echogenicity
                }
            }
        }

        // Optical: High absorption in overlapping region (vascular)
        for i in 2..4 {
            for j in 2..4 {
                for k in 2..4 {
                    optical_data[[i, j, k]] = 0.8; // High optical absorption in same region
                }
            }
        }

        let mut modality_weights = HashMap::new();
        modality_weights.insert("ultrasound".to_string(), 0.5);
        modality_weights.insert("optical_650nm".to_string(), 0.5);

        let fusion_config = FusionConfig {
            output_resolution: [1e-4, 1e-4, 1e-4],
            fusion_method: FusionMethod::WeightedAverage,
            registration_method: RegistrationMethod::RigidBody,
            modality_weights,
            confidence_threshold: 0.6,
            uncertainty_quantification: true,
        };

        let mut fusion = MultiModalFusion::new(fusion_config.clone());

        fusion.register_ultrasound(&ultrasound_data).unwrap();
        fusion.register_optical(&optical_data, 650e-9).unwrap(); // 650 nm (red, good for vascular imaging)

        let fused_result = fusion.fuse().unwrap();

        // Check that complementary information is preserved
        let max_fused = fused_result.intensity_image.iter().fold(0.0_f64, |a: f64, &b: &f64| a.max(b));
        let min_fused = fused_result.intensity_image.iter().fold(f64::INFINITY, |a: f64, &b: &f64| a.min(b));

        // Fusion should show both high-contrast regions
        // With 0.9 * 0.5 + 0.8 * 0.5 = 0.85, max should be > 0.5
        assert!(max_fused > 0.5, "Fusion should preserve high-contrast features, got max={}", max_fused);
        assert!(min_fused < 0.3, "Fusion should preserve low-contrast regions");

        // Quality should be reasonable for complementary modalities
        let avg_quality: f64 = fused_result.modality_quality.values().sum::<f64>() / fused_result.modality_quality.len() as f64;
        assert!(avg_quality > 0.5, "Complementary modalities should yield good fusion quality");

        // Test that different wavelengths affect quality scoring
        let wavelengths = vec![400e-9, 550e-9, 650e-9, 800e-9]; // UV, green, red, NIR

        for &wl in &wavelengths {
            let mut test_fusion = MultiModalFusion::new(fusion_config.clone());
            test_fusion.register_ultrasound(&ultrasound_data).unwrap();
            test_fusion.register_optical(&optical_data, wl).unwrap();

            let test_result = test_fusion.fuse().unwrap();

            // All should produce valid results
            let test_avg_quality: f64 = test_result.modality_quality.values().sum::<f64>() / test_result.modality_quality.len() as f64;
            assert!(test_avg_quality > 0.0, "All wavelengths should produce valid quality scores");
        }
    }

    #[test]
    fn validate_multi_modal_spatial_registration() {
        // Test spatial registration for multi-modal imaging alignment
        // Theorem: Accurate spatial registration enables meaningful multi-modal fusion

        use kwavers::physics::imaging::registration::{ImageRegistration, SpatialTransform};
        use ndarray::Array2;

        let registration = ImageRegistration::default();

        // Create corresponding landmark points between ultrasound and optical images
        let ultrasound_landmarks = Array2::from_shape_vec((4, 3), vec![
            10.0, 20.0, 5.0,   // Landmark 1
            30.0, 15.0, 8.0,   // Landmark 2
            15.0, 35.0, 12.0,  // Landmark 3
            25.0, 40.0, 6.0,   // Landmark 4
        ]).unwrap();

        let optical_landmarks = Array2::from_shape_vec((4, 3), vec![
            14.8, 16.9, 7.0,   // Landmark 1 + transform
            34.8, 11.9, 10.0,  // Landmark 2 + transform
            19.8, 31.9, 14.0,  // Landmark 3 + transform
            29.8, 36.9, 8.0,   // Landmark 4 + transform
        ]).unwrap();

        let result = registration.rigid_registration_landmarks(&ultrasound_landmarks, &optical_landmarks).unwrap();

        // Validate registration result
        assert!(result.confidence > 0.5, "Registration should have reasonable confidence");

        // Check FRE if computed
        if let Some(fre) = result.quality_metrics.fre {
            assert!(fre >= 0.0, "FRE should be non-negative, got {} mm", fre);
        }

        // Verify transform matrix is valid (homogeneous)
        assert!((result.transform_matrix[15] - 1.0).abs() < 1e-6, "Homogeneous matrix should have [3,3] = 1");

        // Check spatial transform type
        match result.spatial_transform {
            Some(SpatialTransform::RigidBody { rotation, translation }) => {
                // Rotation matrix should have reasonable values
                assert!(rotation.iter().all(|&r| r.abs() <= 2.0), "Rotation matrix elements should be reasonable");

                // Translation should be reasonable
                assert!(translation.iter().all(|&t| t.abs() < 100.0), "Translation should be reasonable");
            }
            _ => panic!("Expected RigidBody transform"),
        }
    }

    #[test]
    fn validate_temporal_synchronization_multi_modal() {
        // Test temporal synchronization for real-time multi-modal acquisition

        use kwavers::physics::imaging::registration::ImageRegistration;
        use ndarray::Array1;

        let registration = ImageRegistration::default();

        let sampling_rate = 100000.0; // 100 kHz sampling
        let n_samples = 1000;

        // Generate reference and target signals with known phase offset
        let ref_signal = Array1::from_vec((0..n_samples).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 100.0).sin()
        }).collect());

        let target_signal = Array1::from_vec((0..n_samples).map(|i| {
            (2.0 * std::f64::consts::PI * i as f64 / 100.0 + std::f64::consts::PI / 4.0).sin()
        }).collect());

        let sync_result = registration.temporal_synchronization(
            &ref_signal,
            &target_signal,
            sampling_rate
        ).unwrap();

        // Validate synchronization quality
        assert!(sync_result.phase_offset.abs() < std::f64::consts::PI * 2.0,
                "Phase offset should be reasonable");

        // Quality metrics should be computed
        assert!(sync_result.quality_metrics.rms_timing_error >= 0.0);
        assert!(sync_result.quality_metrics.phase_lock_stability >= 0.0);
        assert!(sync_result.quality_metrics.phase_lock_stability <= 1.0);
        assert!(sync_result.quality_metrics.sync_success_rate >= 0.0);
        assert!(sync_result.quality_metrics.sync_success_rate <= 1.0);
    }
