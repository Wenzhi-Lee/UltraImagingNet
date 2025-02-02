clearvars;

% =========================================================================
% SIMULATION
% =========================================================================

% Create the computational grid
DATA_CAST = 'gpuArray-single'; 
CFL = 0.25; 
sigma = 2.5; 
alpha_0 = 0.05 * 1;
p0 = 10e5; 
c0 = 5918; % speed of sound in the medium
rho0 = 8000;
f0 = 1e6;  % frequency [Hz]
PML_size = 8; % Size of the PML in grid points
PML_alpha = 1.5;
N = 7;
% Update Nx and Ny to 512
Nx = 512;  % number of grid points in the x direction
Ny = 512;  % number of grid points in the y direction

% Set the physical area dimensions to 51.2 mm x 51.2 mm
area_size = 51.2e-3;  % Area size in meters (51.2 mm)

% Calculate grid spacing based on the area size and number of grid points
dx = area_size / Nx;  % Grid point spacing in the x direction [m]
dy = area_size / Ny;  % Grid point spacing in the y direction [m]

kgrid = kWaveGrid(Nx, dx, Ny, dy);

% Other parameters
mach_num = p0 / (rho0 * c0.^2);


% define the properties of the propagation medium
%medium.sound_speed = 1500;	% [m/s]
medium.alpha_coeff = alpha_0*(1 + 0.2 * (rand(Nx, Ny) - 0.5));      % 衰减系数 [dB/(MHz^y cm)]
medium.alpha_power = 2;       % 衰减的幂次关系
medium.BonA = 1e-6;   

% Create the medium properties (sound speed and density maps)
sound_speed_map = c0 * ones([Nx, Ny]);   % Background sound speed
density_map = rho0 * ones([Nx, Ny]);       % Background density

% Create the medium properties (sound speed and density maps)
sound_speed_map = c0 * (1 + 0.2 * (rand([Nx, Ny]) - 0.5));   % Randomized fluctuation of ±10% for sound speed
density_map = rho0 * (1 + 0.2 * (rand([Nx, Ny]) - 0.5));      % Randomized fluctuation of ±10% for density

%%成像目标
%4target
for i = 1:4
    i = 4
    disc_x_pos = 6+100*i;            % [grid points]
    disc_y_pos = Ny/2+225-90*i; 
    disc_radius = 40;            % [grid points]
    disc_2 = makeDisc(Nx, Ny, disc_x_pos, disc_y_pos, disc_radius);

    scattering_c0 = 340 + 50 * randn(sum(disc_2(:)), 1);
    scattering_c0(scattering_c0 < 290) = 290;
    scattering_c0(scattering_c0 > 390) = 390;

    scattering_rho0 = 13 + 0.3 * randn(sum(disc_2(:)), 1);
    scattering_rho0(scattering_rho0 < 10) = 10;
    scattering_rho0(scattering_rho0 > 16) = 16;


    sound_speed_map(disc_2 == 1) = scattering_c0;
    density_map(disc_2 == 1) = scattering_rho0;
end

% Set the medium properties
medium.sound_speed = sound_speed_map;
medium.density = density_map;

% Plot the sound speed map
figure;
imagesc(sound_speed_map); % Plot the heatmap
colorbar; % Display color bar
colormap('hot'); % Use 'hot' colormap
axis image; % Maintain aspect ratio

% Set the number of grid points per wavelength and wavelength separation
points_per_wavelength = 10;  % Grid points per wavelength at f0
wavelength_separation = 600;  % Separation between source and detector

% Calculate points per period and the duration of the signal
points_per_period = round(points_per_wavelength / CFL);
dt = 1 / (points_per_period * f0);  % Duration for 5 cycles of the source frequency
c_max = max(sound_speed_map(:));  % 获取最大声速
dt = CFL * dx / c_max;          % 基于最大声速计算时间步长

% Compute the time steps
Nt = round(4e-5 / dt);
t_end = Nt * dt ;


% Set the time array for the grid
kgrid.setTime(Nt, dt);

% Signal source (e.g., square wave)
f_center1 = 2e6;  % 2 MHz
t = kgrid.t_array;

% Generate the square wave signal
square_wave1 = square(2 * pi * f_center1 * t);  % 2 MHz square wave

% Adjust the amplitude as needed
amplitude1 = p0;  % Amplitude for 2 MHz

% Apply the source signal to the pressure source with no windowing
% Adjust this part for the new signal (e.g., square wave)
source.p(1 + 40 * 6 : points_per_period * 6 + 1 + 40 * 6) = amplitude1 * square_wave1(1:points_per_period * 6 + 1);

% Input arguments for simulation
input_args = {'PlotFreq', 20 , 'PlotScale', [-1, 1] * p0 * 1.05, ...
    'PMLInside', false, 'PMLSize', PML_size, 'PMLAlpha', PML_alpha, 'DataCast', DATA_CAST, 'DataRecast', true};

% Assign the source signal
source.p_mask = zeros(Nx, Ny);
num_source = 32;
sensor.mask = zeros(kgrid.Nx, kgrid.Ny);
%sensor.mask(1,:) = 1;
sensor.mask(end, :) = 1;

data = zeros(num_source, Ny, Nt);

disp(['Time step: ', num2str(kgrid.dt)]);

% Run the simulation for 32 sources
for i = 1:32
    source.p_mask(end, Ny/2 - 16 + i) = 1; % Line source along y-axis center
    source.p_mask(end, Ny/2 - 16 + i - 1) = 0; % Line source along y-axis center

    % Run the simulation
    sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});
    sensor_data_directional = sensor_data;
    data(i,:,:) = sensor_data;
    
    % 确保 y 轴索引是整数
    y1 = round(Ny / 4);      % y = Ny / 4
    y2 = round(Ny / 2);      % y = Ny / 2
    y3 = round(Ny * 3 / 4);  % y = Ny * 3 / 4

    index = 1;
    % 提取传感器数据
    sensor_data_y1 = sensor_data_directional(y1, index:end);  % Ny/4 的时间数据
    sensor_data_y2 = sensor_data_directional(y2, index:end);  % Ny/2 的时间数据
    sensor_data_y3 = sensor_data_directional(y3, index:end);  % Ny*3/4 的时间数据

    t1 = (index:(size(sensor_data_directional, 2))) * dt; % 时间向量，单位为秒

    % Visualize time-domain signals for different sensors
    figure(2);
    % Time domain signal at Ny/4
    subplot(3, 1, 1);
    plot(t1 * 1e6, sensor_data_y1, 'b');
    title('Time Domain Signal at Ny/4');
    xlabel('Time [\mu s]');
    ylabel('Pressure [Pa]');
    grid on;

    % Time domain signal at Ny/2
    subplot(3, 1, 2);
    plot(t1 * 1e6, sensor_data_y2, 'r');
    title('Time Domain Signal at Ny/2');
    xlabel('Time [\mu s]');
    ylabel('Pressure [Pa]');
    grid on;

    % Time domain signal at Ny*3/4
    subplot(3, 1, 3);
    plot(t1 * 1e6, sensor_data_y3, 'g');
    title('Time Domain Signal at Ny*3/4');
    xlabel('Time [\mu s]');
    ylabel('Pressure [Pa]');
    grid on;
end


