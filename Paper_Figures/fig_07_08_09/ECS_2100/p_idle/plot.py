import json
import math

import numpy as np
import itertools

base_paths=["idle_ohne_ssh","idle_ohne_kabel","idle_mit_ssh"]
scenarios=[[1,0,64]]
all_res = [[[] for i in range(len(scenarios))] for x in base_paths]

def measurement_uncertainty(std_dev_real_power):
    measurement_uncertainty = (((0.001 * 300) * (0.001 * 50) * 2) ** 2) * 2
    measurement_uncertainty = math.sqrt(measurement_uncertainty + std_dev_real_power ** 2)
    return measurement_uncertainty

for p_ind, base_path in enumerate(base_paths):
    for ind, currfile in enumerate(scenarios):
        for currun in range(5):
            print(str(base_path) + "/results_" + str(ind) + "_" + str(currun) + ".json")
            with open(str(base_path) + "/results_" + str(ind) + "_" + str(currun) + ".json") as jsonfile:
                data = dict(json.load(jsonfile))
            for curr in range(len(data["Time"])):
                ts = data["Time"][curr] - data["Start_Time"][0]
                if ts > 10:
                    try:
                        all_res[p_ind][ind].append(float((data["Real_Power"][curr])))
                    except:
                        pass
    all_res[p_ind]=list(itertools.chain(all_res[p_ind]))
print(np.nanmean(all_res[1]))
print(measurement_uncertainty(np.nanstd(all_res[1])))

diff=np.nanmean([all_res[0][0][x]-all_res[1][0][x] for x in range(len(all_res[0]))])
print(diff/8)
print(measurement_uncertainty(np.nanstd(all_res[0]))/8)
diff=np.nanmean([all_res[2][0][x]-all_res[0][0][x] for x in range(len(all_res[0]))])
print(diff)
print(measurement_uncertainty(np.nanstd(all_res[2])))