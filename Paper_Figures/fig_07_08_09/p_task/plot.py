import csv
import json
import math
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import SmoothBivariateSpline

base_paths = ["ports_connected_service_used","ports_connected_service_disabled","ports_disconnected_service_disabled"]

def measurement_uncertainty(std_dev_real_power):
    measurement_uncertainty = (((0.001 * 300) * (0.001 * 50) * 2) ** 2)
    measurement_uncertainty = math.sqrt(measurement_uncertainty + std_dev_real_power ** 2)
    return measurement_uncertainty
def measurement_uncertainty2(std_dev_real_power,std_dev_real_power2):
    measurement_uncertainty = (((0.001 * 300) * (0.001 * 50) * 2) ** 2) * 2
    measurement_uncertainty = math.sqrt(measurement_uncertainty + std_dev_real_power ** 2++ std_dev_real_power2 ** 2)
    return measurement_uncertainty

all_res = [[] for i in range(len(base_paths))]

for ind, currfile in enumerate(base_paths):
    for currun in range(5):
        with open(str(currfile) + "/results_0_" + str(currun) + ".json") as jsonfile:
            data = dict(json.load(jsonfile))
        for curr in range(len(data["Time"])):
            ts = data["Time"][curr] - data["Start_Time"][0]
            if ts > 10:
                try:
                    all_res[ind].append(float((data["Real_Power"][curr])))
                except:
                    pass


plt.rcParams.update({'font.size': 24})
plt.rcParams['figure.figsize'] = [10, 6]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0,1,5))

for ind,curr in enumerate(all_res):
    plt.boxplot(curr,positions=[ind])
    print(np.mean(curr))
    print(measurement_uncertainty(np.std(curr)))
print(measurement_uncertainty2(np.std(all_res[0]),np.std(all_res[1])))
plt.xlabel('scenario')
plt.ylabel('real power [W]')
plt.legend()
plt.grid(True)
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig("power_diff.pdf")
plt.close()