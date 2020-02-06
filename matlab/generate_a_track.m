function [car_traj,bearings_aci, bearings_obj] = generate_a_track(measurement_sigma, process_noise, ts, sensor_positions)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
A = @(Ts)[1 0 Ts 0;0 1 0 Ts; 0 0 1 0; 0 0 0 1];
B =@(Ts)[0.5*Ts^2*eye(2);Ts*eye(2)];

initial_positions = rand(1,2) .* sensor_positions(2,:);
initial_velocities = rand(1,1) * 10; %m/s
initial_orientation = rand(1,1) * 360;
dx = cosd(90 - initial_orientation) * initial_velocities;
dy = sind(90 - initial_orientation) * initial_velocities;
a = -1 * ts/2;
b = ts/2;
ts_noise = (b-a).*rand(1,500) + a;
temp_time_array = linspace(0,500,500/ts) + ts_noise;
%% Hedef Trajectory
car_traj = [initial_positions dx dy];
for i = 1:length(temp_time_array)-1
    ts = temp_time_array(i+1) - temp_time_array(i);
    car_traj = [car_traj; (A(ts) * car_traj(end,:)')' + (B(ts) * randn(2,1) * process_noise)']; 
end
%% Hedef İcin Acilar
bearings_obj = cell(length(temp_time_array),1);
bearings_aci = zeros(length(temp_time_array),1);
for i = 1:length(temp_time_array)
    sensor_no = randsample( [1 2], 1, true, [0.5 0.5] );
    if sensor_no == 1
        sensor_location = sensor_positions(1,:);
        angle = atan2d(car_traj(i,1) - sensor_location(1), car_traj(i,2) - sensor_location(2)) + measurement_sigma * randn(1,1);
        bearings_obj{i} = bearing(sensor_location,angle, temp_time_array(i));
        bearings_aci(i) = angle;
    else
        sensor_location = sensor_positions(2,:);
        angle = atan2d(car_traj(i,1) - sensor_location(1), car_traj(i,2) - sensor_location(2)) + measurement_sigma * randn(1,1);
        bearings_obj{i} = bearing(sensor_location,angle, temp_time_array(i));
        bearings_aci(i) = angle;
    end
    
    
end

