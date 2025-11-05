%% k-Wave validation script for: plane_wave_homogeneous
%% Plane wave propagation in homogeneous medium

clear all;
close all;

%% Grid setup
Nx = 256;
Ny = 64;
Nz = 1;
dx = 0.100000;     %% mm
dy = 0.100000;     %% mm
dz = 0.100000;     %% mm
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

%% Medium setup
medium.sound_speed = 1482.0;      %% m/s
medium.density = 998.2;           %% kg/m³
medium.alpha_coeff = 0.75;         %% dB/(MHz^y cm)
medium.alpha_power = 1.5;          %% y

%% Source setup
source.p0 = 100000.0;              %% Pa
source_freq = 1000000.0;            %% Hz
source_cycles = 3;
source.p = makeTimeVaryingSource(kgrid, source, source_freq, source_cycles);
source.p_mask = zeros(Nx, Ny, Nz);
source.p_mask(1, :, :) = 1;

%% Sensor setup
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(129, 1, 1) = 1;
sensor.mask(129, 2, 1) = 1;
sensor.mask(129, 3, 1) = 1;
sensor.mask(129, 4, 1) = 1;
sensor.mask(129, 5, 1) = 1;
sensor.mask(129, 6, 1) = 1;
sensor.mask(129, 7, 1) = 1;
sensor.mask(129, 8, 1) = 1;
sensor.mask(129, 9, 1) = 1;
sensor.mask(129, 10, 1) = 1;
sensor.mask(129, 11, 1) = 1;
sensor.mask(129, 12, 1) = 1;
sensor.mask(129, 13, 1) = 1;
sensor.mask(129, 14, 1) = 1;
sensor.mask(129, 15, 1) = 1;
sensor.mask(129, 16, 1) = 1;
sensor.mask(129, 17, 1) = 1;
sensor.mask(129, 18, 1) = 1;
sensor.mask(129, 19, 1) = 1;
sensor.mask(129, 20, 1) = 1;
sensor.mask(129, 21, 1) = 1;
sensor.mask(129, 22, 1) = 1;
sensor.mask(129, 23, 1) = 1;
sensor.mask(129, 24, 1) = 1;
sensor.mask(129, 25, 1) = 1;
sensor.mask(129, 26, 1) = 1;
sensor.mask(129, 27, 1) = 1;
sensor.mask(129, 28, 1) = 1;
sensor.mask(129, 29, 1) = 1;
sensor.mask(129, 30, 1) = 1;
sensor.mask(129, 31, 1) = 1;
sensor.mask(129, 32, 1) = 1;
sensor.mask(129, 33, 1) = 1;
sensor.mask(129, 34, 1) = 1;
sensor.mask(129, 35, 1) = 1;
sensor.mask(129, 36, 1) = 1;
sensor.mask(129, 37, 1) = 1;
sensor.mask(129, 38, 1) = 1;
sensor.mask(129, 39, 1) = 1;
sensor.mask(129, 40, 1) = 1;
sensor.mask(129, 41, 1) = 1;
sensor.mask(129, 42, 1) = 1;
sensor.mask(129, 43, 1) = 1;
sensor.mask(129, 44, 1) = 1;
sensor.mask(129, 45, 1) = 1;
sensor.mask(129, 46, 1) = 1;
sensor.mask(129, 47, 1) = 1;
sensor.mask(129, 48, 1) = 1;
sensor.mask(129, 49, 1) = 1;
sensor.mask(129, 50, 1) = 1;
sensor.mask(129, 51, 1) = 1;
sensor.mask(129, 52, 1) = 1;
sensor.mask(129, 53, 1) = 1;
sensor.mask(129, 54, 1) = 1;
sensor.mask(129, 55, 1) = 1;
sensor.mask(129, 56, 1) = 1;
sensor.mask(129, 57, 1) = 1;
sensor.mask(129, 58, 1) = 1;
sensor.mask(129, 59, 1) = 1;
sensor.mask(129, 60, 1) = 1;
sensor.mask(129, 61, 1) = 1;
sensor.mask(129, 62, 1) = 1;
sensor.mask(129, 63, 1) = 1;
sensor.mask(129, 64, 1) = 1;

%% Simulation setup
input_args = {'PMLSize', 20, 'PMLInside', false, 'PlotPML', false, 'Smooth', false};
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});

%% Save results
save('plane_wave_homogeneous_kwave_output.mat', 'sensor_data', 'kgrid', 'medium', 'source', 'sensor');
fprintf('k-Wave simulation completed for: plane_wave_homogeneous\n');
