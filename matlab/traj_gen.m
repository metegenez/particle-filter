function [car_traj,deg_traj,noisy_deg_traj,noisy_car_traj,time_array,track_ids] = traj_gen(int_coord,sen_coord,dur_time,Ts)
%TRAJ_GEN Generates a trajection

%INPUTS
% int_coord: inital position and states of the target
%            respectively x coordinates, y coordinates
%            initial velocity and initial direction of movement
% sen_coord: Coordinates of the sensors in cartesian
% dur_time:  Duration of time with starting and end point
% Ts:        Sampling Time

%OUTPUTS
% car_traj:       Trajectory points in cartesian plane
% deg_traj:       Trajectory points in theta plane (in degree)
% noisy_deg_traj: Received bearings for both sensors
global sigma
global process_noise
sigma_r=sqrt(sigma);
q=sqrt(process_noise);
d=sen_coord;

inf=@(x)[(d(2,1)*tand(90-x(2))-d(1,1)*tand(90-x(1))+d(1,2)-d(2,2))/(tand(90-x(2))-tand(90-x(1)));
    (tand(90-x(1))-d(1,1))*(d(2,1)*tand(90-x(2))-d(1,1)*tand(90-x(1))+d(1,2)-d(2,2))/(tand(90-x(2))-tand(90-x(1)))+d(1,2)];

x_int=int_coord(1);
y_int=int_coord(2);
v_int=int_coord(3); %initial velocity
a_int=int_coord(4); %initial direction of movement



A=[1 0 Ts 0;0 1 0 Ts; 0 0 1 0; 0 0 0 1];
B=[0.5*Ts^2*eye(2);Ts*eye(2)];
dx=v_int*cosd(90 - a_int);
dy=v_int*sind(90 - a_int);
car_traj(:,1)=[x_int;y_int;dx;dy];
deg_traj(:,1)=[atan2d(car_traj(1,1)-sen_coord(1,1),car_traj(2,1)-sen_coord(1,2)); ...
    atan2d(car_traj(1,1)-sen_coord(2,1),car_traj(2,1)-sen_coord(2,2))];
noisy_deg_traj(:,1)=deg_traj(:,1)+sigma_r*randn(2,1);


L=floor((dur_time(2)-dur_time(1))/Ts);
for i=2:L
    car_traj(:,i)=A*car_traj(:,i-1)+B*q*randn(2,1);
    deg_traj(:,i)=[atan2d(car_traj(1,i)-sen_coord(1,1),car_traj(2,i)-sen_coord(1,2)); ...
        atan2d(car_traj(1,i)-sen_coord(2,1),car_traj(2,i)-sen_coord(2,2))];
    noisy_deg_traj(:,i)=deg_traj(:,i)+sigma_r*randn(2,1);
    noisy_car_traj(:,i)=inf(noisy_deg_traj(:,i));
    test_car_traj(:,i)=inf(deg_traj(:,i));
end

time_array=[dur_time(1):Ts:dur_time(2)];
track_ids=100+randi(899,2,1)';
end

