import pickle
import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from particler_filter_pytorch import stratified_resample, residual_resample, systematic_resample

measurement_sigmas = list(np.sqrt(np.linspace(0.02, 5, 10)))
Ns = [50, 100,500,1000, 5000, 10000]
resample_funcs = [stratified_resample, residual_resample, systematic_resample]
resample_funcs_names = ["stratified_resample", "residual_resample", "systematic_resample"]
process_noise = [np.sqrt(0.75)]
trials = list(product(measurement_sigmas, process_noise, Ns, resample_funcs, list(range(1000))))
performance_to_sigma = pickle.load(open(f'results_low_particle_resamplings.pickle', "rb"))

rmse_dict = {}
ind_mse_dict = {}
for func in resample_funcs_names:
    plt.figure()
    for particle_s in Ns:
        rmse_dict = {}
        ind_mse_dict = {}
        for sigma in measurement_sigmas:
            rmse_dict["{0:.2f}".format(sigma)] = []
            ind_mse_dict["{0:.2f}".format(sigma)] = []


        for i,each_try in enumerate(performance_to_sigma):
            if each_try != [] and particle_s == trials[i][2]:
                if func == trials[i][3].__name__:
                # temp_particle = particle[trials[i][2]]
                    rmse_dict["{0:.2f}".format(trials[i][0])].append(performance_to_sigma[i][0][0])
                    ind_mse_dict["{0:.2f}".format(trials[i][0])].append(performance_to_sigma[i][0][1])
            # particle[trials[i][2]] = temp_particle

        y = []
        x = []
        for sigma in rmse_dict.keys():
            y.append(np.nanmean(np.array(rmse_dict[sigma])))
            x.append(sigma)
        x.reverse()
        y.reverse()
        plt.plot(x,y, label=f"{particle_s} particle")
        plt.title(f"RMSE Performance by Measurement Error for {func}")
        plt.xlabel("Measurement Error")
        plt.ylabel("RMSE")
    plt.show()
    plt.pause(0.0001)
