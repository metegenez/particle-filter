classdef senaryo < handle
    %SENARYO Senaryolarin degiskenleri burada tutulacak
    %   Bir ortam� olusturuken,
    
    % initial_position
    % initial_velocity
    % initial_orientation
    
    % verileri kullanilmaktadir. Bunlarla beraber
    
    % max_initial_velocity verisi de statik belirlenmektedir.
    
    % �stege gore random veya secerek senaryo olusturabilir. Eger random
    % bolmesine hicbi sey girilmezse, default olarak random hedefler
    % olusturuyor. Hedefleri secebilmek icin ozellikle random degiskeni "0" girilmeli.
    
    properties
 
        boundaries
        max_initial_velocity
        time
        Ts
        track_time_bounds
        sensor_positions
        number_of_targets
        pts
        data
        track_id_all
        ground_truth
        sorted_data
        velocities
        orientations
        target_positions
        senaryo_cesit
    end
    
    methods
        function obj = senaryo(no_of_targets, Ts,time, sensors, sigma_r, q, random, boundaries, senaryo_cesit)
            %SENARYO Senaryo olusturken, nokta secilebilmeli
            %   Ginput kullan�lacak
            if nargin > 6
                obj.randomness = random;
            else
                obj.randomness = true;
            end
            global sigma
            global process_noise
            sigma = sigma_r;
            process_noise = q;
            obj.senaryo_cesit = senaryo_cesit;
            obj.boundaries = boundaries;
            obj.Ts = Ts;
            obj.data = [];
            obj.track_id_all = [];
            obj.ground_truth = [];
            obj.time = 1:Ts:time;
            obj.max_initial_velocity = 10;
            obj.track_time_bounds = [obj.time(1) obj.time(end)];
            obj.sensor_positions = sensors;
            obj.number_of_targets = no_of_targets;
            if obj.randomness
                obj.random_target_generate_data();
            else
                obj.select_targets_generate_data();
            end
            [~, order] = sort(obj.data(:,1));
            obj.sorted_data = array2table(obj.data(order,:),'VariableNames',{char("zaman"),char("BR"),char("STRN"),char("sensor"),char("BR_real")});
            
            
        end
        
        
        
        function obj = random_target_generate_data(obj)
            % Senaryo sinirlari
            errs = 1;
            while errs
                
                if obj.senaryo_cesit(1) == 1
                    obj.velocities = ones(obj.number_of_targets,1) * randi([3 obj.max_initial_velocity],1,1);
                    
                else
                    obj.velocities = randi([3 obj.max_initial_velocity],obj.number_of_targets,1);
                end
                
                
                if obj.senaryo_cesit(2) == 1
                    obj.orientations = ones(obj.number_of_targets,1) * randi(360,1,1);
                else
                    obj.orientations = randi(360,obj.number_of_targets,1);
                end
                
                if obj.senaryo_cesit(3) == 1
                    % 20 km x 20 km lik bir alanda yogun hedefler
                    center = [randi([obj.boundaries(1) obj.boundaries(2)],1,1) randi([obj.boundaries(3) obj.boundaries(4)],1,1)];
                    obj.target_positions = [randi([center(1) - 20000 center(1)],obj.number_of_targets,1) randi([center(2) - 20000 center(2)],obj.number_of_targets,1)];
                else
                    obj.target_positions = [randi([obj.boundaries(1) obj.boundaries(2)],obj.number_of_targets,1) randi([obj.boundaries(3) obj.boundaries(4)],obj.number_of_targets,1)];
                end
                
                
                obj.track_id_all = [];
                obj.data = [];
                obj.ground_truth = [];
                for i = 1:obj.number_of_targets
                    initial_position = obj.target_positions(i,:);
                    initial_velocity = obj.velocities(i);
                    initial_orientation = obj.orientations(i);
                    [car_traj,deg_traj,noisy_deg_traj,~,time_array,track_ids] = traj_gen([initial_position initial_velocity initial_orientation],obj.sensor_positions,obj.track_time_bounds,obj.Ts);
                    obj.data = [obj.data; [time_array(1:end-1)' noisy_deg_traj(1,:)' ones(length(time_array)-1,1).*track_ids(1) ones(length(time_array)-1,1) deg_traj(1,:)'; time_array(1:end-1)' noisy_deg_traj(2,:)' ones(length(time_array)-1,1).*track_ids(2) ones(length(time_array)-1,1)*2 deg_traj(2,:)']];
                    obj.track_id_all = [obj.track_id_all track_ids]; %track_ids 2x1
                    obj.ground_truth=[obj.ground_truth; car_traj];
                end
                errs = 0;
                if ~isequal(sort(obj.track_id_all),unique(obj.track_id_all))
                    errs=1;
                    obj.data = [];
                    obj.track_id_all = []; %track_ids 2x1
                    obj.ground_truth=[];
                    obj.velocities = [];
                end
            end
            
        end
    end
end
