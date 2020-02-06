classdef particle_filter < handle
    %particle_filter Monte Carlo Filtering
    
    properties
        particle_number
        h
        covariance
        particles
        weights
    end
    
    methods
        function obj = particle_filter(particle_number,h)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.particle_number = particle_number;
            obj.h = h;
        end
        
        function obj = initiate(obj, initial_state, initial_covariance)
            obj.covariance = initial_covariance;
            obj.particles = mvnrnd(initial_state,initial_covariance,obj.particle_number);
            obj.weights = ones(2000,1) * (1/obj.particle_number);
            
        end
        
        function obj = predict(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
        function obj = update(obj)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            
        end
        function obj = estimate(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
        function obj = resample(obj)
            
        end
        
        
    end
end

