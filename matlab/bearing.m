classdef bearing < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        sensor_no
        angle
        time
        sensor_location
    end
    
    methods
        function obj = bearing(sensor_location,angle, time)
            %UNTITLED2 Construct an instance of this class
            %   Detailed explanation goes here
            obj.sensor_location = sensor_location;
            obj.angle = angle;
            obj.time = time;
        end
    end
end

