%% k-Wave validation script for: point_source_radiation
%% Point source radiation pattern

clear all;
close all;

%% Grid setup
Nx = 128;
Ny = 128;
Nz = 1;
dx = 0.050000;     %% mm
dy = 0.050000;     %% mm
dz = 0.050000;     %% mm
kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz);

%% Medium setup
medium.sound_speed = 1482.0;      %% m/s
medium.density = 998.2;           %% kg/m³
medium.alpha_coeff = 0.75;         %% dB/(MHz^y cm)
medium.alpha_power = 1.5;          %% y

%% Source setup
source.p0 = 1000000.0;              %% Pa
source_freq = 5000000.0;            %% Hz
source_cycles = 3;
source.p = makeTimeVaryingSource(kgrid, source, source_freq, source_cycles);
source.p_mask = zeros(Nx, Ny, Nz);
source.p_mask(128, 128, 1) = 1;

%% Sensor setup
sensor.mask = zeros(Nx, Ny, Nz);
sensor.mask(33, 33, 1) = 1;
sensor.mask(65, 33, 1) = 1;
sensor.mask(97, 33, 1) = 1;
sensor.mask(65, 33, 1) = 1;
sensor.mask(65, 65, 1) = 1;
sensor.mask(65, 97, 1) = 1;

%% Simulation setup
input_args = {'PMLSize', 10, 'PMLInside', false, 'PlotPML', false, 'Smooth', false};
sensor_data = kspaceFirstOrder3D(kgrid, medium, source, sensor, input_args{:});

%% Save results
save('point_source_radiation_kwave_output.mat', 'sensor_data', 'kgrid', 'medium', 'source', 'sensor');
fprintf('k-Wave simulation completed for: point_source_radiation\n');
