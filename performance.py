import pickle
import numpy as np
from itertools import product
from matplotlib import pyplot as plt

measurement_sigmas = list(np.sqrt(np.linspace(0.02, 5, 10)))
Ns = [1000,5000, 10000, 15000, 20000]
process_noise = [np.sqrt(0.75)]
trials = list(product(measurement_sigmas, process_noise, Ns, list(range(100))))
performance_to_sigma = pickle.load(open(f'results_for_sigmas.pickle', "rb"))

particle = {}
rmse_dict = {}
ind_mse_dict = {}

for particle_s in Ns:
    particle[particle_s] = []
    for sigma in measurement_sigmas:
        rmse_dict["{0:.2f}".format(sigma)] = []
        ind_mse_dict["{0:.2f}".format(sigma)] = []
    particle[particle_s].append(rmse_dict)
    particle[particle_s].append(ind_mse_dict)


for i,each_try in enumerate(performance_to_sigma):
    if performance_to_sigma[i] != []:
        temp_particle = particle[trials[i][2]]
        temp_particle[0]["{0:.2f}".format(trials[i][0])].append(performance_to_sigma[i][0][0])
        temp_particle[1]["{0:.2f}".format(trials[i][0])].append(performance_to_sigma[i][0][1])

del performance_to_sigma
for particle_s in Ns:
    y = []
    x = []
    for sigma in particle[particle_s][0].keys():
        y.append(np.nanmean(np.array(particle[particle_s][0][sigma])))
        x.append(sigma)
    x.reverse()
    y.reverse()
    plt.plot(x,y, label=f"{particle_s} particle")
    plt.title("RMSE Performance by Measurement Error")
    plt.xlabel("Measurement Error")
    plt.ylabel("RMSE")
plt.show()