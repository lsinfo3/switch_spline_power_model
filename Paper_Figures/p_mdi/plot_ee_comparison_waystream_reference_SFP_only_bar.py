import csv
import json
import math

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rcParams['figure.figsize'] = [10, 6]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0,1,9))
max_uncertainty = ((0.001 * 300) * (0.001 * 50) * 2)**2


def get_results(path,runs,baseline):
    all_results=[]
    for currfile in runs:
        current_results=[]
        with open(str(path)+str(currfile) + ".json") as jsonfile:
            data = dict(json.load(jsonfile))
        for curr in range(len(data["Voltage"])):
            if(data["Time"][curr]-data["Start_Time"][0])>130:
                all_results.append((data["Real_Power"][curr]-baseline)/2)
                current_results.append(data["Real_Power"][curr])
        #print(str(currfile)+": "+str(np.mean(current_results)))
    return all_results

def confidence_interval_multiple_variables(sample1, sample2, alpha):
    e1 = np.mean(sample1)
    e2 = np.mean(sample2)
    v1 = np.var(sample1)
    v2 = np.var(sample2)
    e12 = e1 * e2
    v12 = (e1 ** 2) * v2 + (e2 ** 2) * v1 + v1 * v2
    n = max(len(sample1), len(sample2))
    d = st.t.interval(1 - (alpha / 2), n - 1)[1] * math.sqrt(v12 / n)
    l = e12 - d
    u = e12 + d
    return ([l, u, d])


configs = ["0 MBit/s", "125 MBit/s", "500 MBit/s", "1000 MBit/s"]
results = []
baseline = []
toexport = {}
confidence_intervals = {}
all_res = {}

# Baseline for DAC Cable based on specs: 30AWG, 0.5m -> 0.1647993 Ohm resistance -> 0.014832W assuming 300mA (maximum receive current for SFP)

# results.append(get_results(["results_idle_0","results_idle_1","results_idle_2","results_idle_3","results_idle_4"]))
# results.append(get_results(["results_0_0","results_0_1","results_0_2","results_0_3","results_0_4"]))
# results.append(get_results(["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"]))
# results.append(get_results(["results_2_0","results_2_1","results_2_2","results_2_3","results_2_4"]))
baseline.append(get_results("./results_DAC/", ["results_idle_0", "results_idle_4"], 0.018399607072691553))
baseline.append(
    get_results("./results_DAC/", ["results_0_0", "results_0_1", "results_0_3", "results_0_4"], 0.018399607072691553))
baseline.append(
    get_results("./results_DAC/", ["results_1_0", "results_1_1", "results_1_2", "results_1_3", "results_1_4"],
                0.018399607072691553))
baseline.append(get_results("./results_DAC/", ["results_2_0", "results_2_3"], 0.018399607072691553))

baseline_means = []
for curr in baseline:
    baseline_means.append(np.mean(curr)*2)

results.append(get_results("./results_LR/", ["results_idle_0", "results_idle_2", "results_idle_4"], baseline_means[0]))
results.append(get_results("./results_LR/", ["results_0_0", "results_0_3"], baseline_means[1]))
results.append(
    get_results("./results_LR/", ["results_1_1", "results_1_2", "results_1_3", "results_1_4"], baseline_means[2]))
results.append(get_results("./results_LR/", ["results_2_0", "results_2_2"], baseline_means[3]))

toexport["DAC"] = [float(np.mean(x)) for x in baseline_means]
toexport["LR"] = [float(np.mean(x)) for x in results]
all_res["DAC"] = baseline
all_res["LR"] = results

results = []
results.append(
    get_results("./results_LR_Waystream/", ["results_idle_0", "results_idle_1", "results_idle_2", "results_idle_4"],
                baseline_means[0]))
results.append(get_results("./results_LR_Waystream/", ["results_0_0", "results_0_1", "results_0_3", "results_0_4"],
                           baseline_means[1]))
results.append(
    get_results("./results_LR_Waystream/", ["results_1_0", "results_1_1", "results_1_2", "results_1_3", "results_1_4"],
                baseline_means[2]))
results.append(get_results("./results_LR_Waystream/", ["results_2_0", "results_2_1", "results_2_2"], baseline_means[3]))

toexport["LR_waystream"] = [float(np.mean(x)) for x in results]
all_res["LR_waystream"] = results

results = []
results.append(get_results("./results_BX/", ["results_idle_1", "results_idle_4"], baseline_means[0]))
results.append(get_results("./results_BX/", ["results_0_0", "results_0_1", "results_0_2", "results_0_3", "results_0_4"],
                           baseline_means[1]))
results.append(get_results("./results_BX/", ["results_1_0", "results_1_1", "results_1_2", "results_1_3", "results_1_4"],
                           baseline_means[2]))
results.append(get_results("./results_BX/", ["results_2_1", "results_2_4"], baseline_means[3]))

toexport["BX"] = [float(np.mean(x)) for x in results]
all_res["BX"] = results

results = []
results.append(
    get_results("./results_BX_Waystream/", ["results_idle_0", "results_idle_1", "results_idle_2", "results_idle_3"],
                baseline_means[0]))
results.append(get_results("./results_BX_Waystream/", ["results_0_1", "results_0_3"], baseline_means[1]))
results.append(get_results("./results_BX_Waystream/", ["results_1_1", "results_1_2", "results_1_3"], baseline_means[2]))
results.append(get_results("./results_BX_Waystream/", ["results_2_0", "results_2_1", "results_2_2"], baseline_means[3]))

toexport["BX_waystream"] = [float(np.mean(x)) for x in results]
all_res["BX_waystream"] = results

print(toexport)

toplot = []
all_res_sfp=[]
all_res_ref=[]
all_res_dac=[]
for curr in range(4):
    # toplot.append(1-(toexport["LR_waystream"][curr]/toexport["LR"][curr]))
    raw_values=[]
    raw_values_sfp=[]
    raw_values_dac=[]
    raw_values_ref=[]
    for currval in range(len(all_res["BX_waystream"][curr])):
        try:
            raw_values.append(all_res["BX_waystream"][curr][currval]/all_res["LR_waystream"][curr][currval])
            raw_values_sfp.append(all_res["LR_waystream"][curr][currval])
            raw_values_ref.append(all_res["BX_waystream"][curr][currval])
            raw_values_dac.append(all_res["DAC"][curr][currval]-baseline_means[curr])
        except:
            pass # No more measurement points available
    toplot.append(raw_values)
    all_res_sfp.append(raw_values_sfp)
    all_res_dac.append(raw_values_dac)
    all_res_ref.append(raw_values_ref)
plt.bar([0, 4, 8, 12], [np.mean(x)*100 for x in toplot], color=colors[2], label="Waystream LR")

max_values = []
min_values = []
for curr in range(len(toplot)):
    #max_values.append(np.mean(toplot[curr])*100 + (math.sqrt(np.std(toplot[curr]) ** 2 + max_uncertainty**2) / 2)*100)
    #min_values.append(np.mean(toplot[curr])*100 - (math.sqrt(np.std(toplot[curr]) ** 2 + max_uncertainty**2) / 2)*100)
    max_values.append(((math.sqrt(
        (math.sqrt(np.std(all_res_sfp[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(all_res_sfp[curr])) ** 2 + (
                math.sqrt(np.std(all_res_ref[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(
            all_res_ref[curr])) ** 2)) * np.mean(toplot[curr])) * 100)
    min_values.append(((math.sqrt(
        (math.sqrt(np.std(all_res_sfp[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(all_res_sfp[curr])) ** 2 + (
                math.sqrt(np.std(all_res_ref[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(
            all_res_ref[curr])) ** 2)) * np.mean(toplot[curr])) * 100)
plt.errorbar([0,4,8,12],[np.mean(x)*100 for x in toplot],yerr=[min_values,max_values],fmt='none',ecolor="black",elinewidth=10)

toplot = []
all_res_sfp=[]
all_res_ref=[]
all_res_dac=[]
for curr in range(4):
    # toplot.append(1-(toexport["LR_waystream"][curr]/toexport["LR"][curr]))
    raw_values=[]
    raw_values_sfp=[]
    raw_values_dac=[]
    raw_values_ref=[]
    for currval in range(len(all_res["BX_waystream"][curr])):
        try:
            raw_values.append(all_res["BX_waystream"][curr][currval]/all_res["LR"][curr][currval])
            raw_values_sfp.append(all_res["LR"][curr][currval])
            raw_values_ref.append(all_res["BX_waystream"][curr][currval])
            raw_values_dac.append(all_res["DAC"][curr][currval]-baseline_means[curr])
        except:
            pass # No more measurement points available
    toplot.append(raw_values)
    all_res_sfp.append(raw_values_sfp)
    all_res_dac.append(raw_values_dac)
    all_res_ref.append(raw_values_ref)
plt.bar([1, 5, 9, 13], [np.mean(x)*100 for x in toplot], color=colors[4], label="FS.com LR")

max_values = []
min_values = []
for curr in range(len(toplot)):
    #max_values.append(np.mean(toplot[curr])*100 + (math.sqrt(np.std(toplot[curr]) ** 2 + max_uncertainty**2) / 2)*100)
    #min_values.append(np.mean(toplot[curr])*100 - (math.sqrt(np.std(toplot[curr]) ** 2 + max_uncertainty**2) / 2)*100)
    max_values.append(((math.sqrt(
        (math.sqrt(np.std(all_res_sfp[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(all_res_sfp[curr])) ** 2 + (
                math.sqrt(np.std(all_res_ref[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(
            all_res_ref[curr])) ** 2)) * np.mean(toplot[curr])) * 100)
    min_values.append(((math.sqrt(
        (math.sqrt(np.std(all_res_sfp[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(all_res_sfp[curr])) ** 2 + (
                math.sqrt(np.std(all_res_ref[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(
            all_res_ref[curr])) ** 2)) * np.mean(toplot[curr])) * 100)
plt.errorbar([1,5,9,13],[np.mean(x)*100 for x in toplot],yerr=[min_values,max_values],fmt='none',ecolor="black",elinewidth=10)

toplot = []
all_res_sfp=[]
all_res_ref=[]
all_res_dac=[]
for curr in range(4):
    # toplot.append(1-(toexport["LR_waystream"][curr]/toexport["LR"][curr]))
    raw_values=[]
    raw_values_sfp=[]
    raw_values_dac=[]
    raw_values_ref=[]
    for currval in range(len(all_res["BX_waystream"][curr])):
        try:
            raw_values.append(all_res["BX_waystream"][curr][currval]/all_res["BX"][curr][currval])
            raw_values_sfp.append(all_res["BX"][curr][currval])
            raw_values_ref.append(all_res["BX_waystream"][curr][currval])
            raw_values_dac.append(all_res["DAC"][curr][currval]-baseline_means[curr])
        except:
            pass # No more measurement points available
    toplot.append(raw_values)
    all_res_sfp.append(raw_values_sfp)
    all_res_dac.append(raw_values_dac)
    all_res_ref.append(raw_values_ref)
plt.bar([2, 6, 10, 14], [np.mean(x)*100 for x in toplot], color=colors[6], label="FS.com BX")

max_values = []
min_values = []
for curr in range(len(toplot)):
    #max_values.append(np.mean(toplot[curr])*100 + (math.sqrt(np.std(toplot[curr]) ** 2 + max_uncertainty**2) / 2)*100)
    #min_values.append(np.mean(toplot[curr])*100 - (math.sqrt(np.std(toplot[curr]) ** 2 + max_uncertainty**2) / 2)*100)
    max_values.append(((math.sqrt(
        (math.sqrt(np.std(all_res_sfp[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(all_res_sfp[curr])) ** 2 + (
                math.sqrt(np.std(all_res_ref[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(
            all_res_ref[curr])) ** 2)) * np.mean(toplot[curr])) * 100)
    min_values.append(((math.sqrt(
        (math.sqrt(np.std(all_res_sfp[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(all_res_sfp[curr])) ** 2 + (
                math.sqrt(np.std(all_res_ref[curr]) ** 2 + max_uncertainty+np.std(all_res_dac[curr])**2) / np.mean(
            all_res_ref[curr])) ** 2)) * np.mean(toplot[curr])) * 100)
plt.errorbar([2,6,10,14],[np.mean(x)*100 for x in toplot],yerr=[min_values,max_values],fmt='none',ecolor="black",elinewidth=10)


plt.xticks([0,5,9,13],[0,125,500,1000])
plt.legend(loc="lower right",ncol=2,bbox_to_anchor=(1,1))
#plt.ylim(0,1)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.ylabel(r"$CrEE (\eta)$ [%]")
plt.xlabel("bit rate [MBit/s]")
ax.grid(which='major', alpha=0.2)
#plt.show()
plt.savefig('ee_comparison_SFP_only.png')
plt.close()