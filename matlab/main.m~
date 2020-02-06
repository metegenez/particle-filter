%% Parameters
% Senaryo sinirlari(random hedefler icin)
x_left = -77000;
x_right = 30810;
y_bottom = -169100;
y_top = -10000;
boundaries = [x_left x_right y_bottom y_top];
% Takip parametreleri
Ts = 1;
time = 100;
sensor_positions = [0 0; 67810 -179100];
number_of_targets = 1;
process_noise = 10^(-2);
sigma_r = 10^(-6); %standart sapma, varyans degil.
% % % Senaryo Cesitleri
% Farkli tipte senaryolarin denenmesi icin, kombine edilebilecek sekilde
% senaryo olusturulmasi adina eklendi. 
% % Genel - tum parametreler random
% senaryo_cesit = [0 0 0];
% % Ayný hiz - hizlar random ama esit, geri kalaný random
% senaryo_cesit = [1 0 0]; 
% % Ayný aci - acilar random ama esit, geri kalaný random
% senaryo_cesit = [0 1 0];
% % Yakin hedefler - pozisyonlar yakin, geri kalan random
% senaryo_cesit = [0 0 1];

%% Senaryo
[car_traj,bearings_aci, bearings_obj] = generate_a_track(sigma_r, process_noise, Ts, sensor_positions);
%% Particle

N = 2000;
particle_f = particle_filter(N, @h_bearing_only);
initial_state = olusturulan_senaryo.ground_truth(:,1)';
initial_covarience_matrix = diag([10^2 10^2 10 10]);
particle_f.initiate(initial_state, initial_covarience_matrix);


for i = 1:length(bearings_obj)
    
end

