function [bearings] = h_bearing_only(particles, sensor_positions)
%h_bearing_only State'i olcume ceviren fonksiyon, vektorize.
% Hesaplanan, atan2d(x,y) oluyor derece referansından dolayı.

bearings = atan2d(particles(:,1) - sensor_positions(1), particles(:,2)-sensor_positions(2)); 
end

