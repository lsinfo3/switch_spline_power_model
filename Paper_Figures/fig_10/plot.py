import json

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import SmoothBivariateSpline
import pickle
import random
import math
from matplotlib.patches import Patch

random.seed(42)
# P_env Basisparameter
a_env = -0.00011215736513100496
b_env = 0.0002996332246959438
c_env = -0.00017340552076487947
t_env = 0

p_mdi_dac = 0.0001526627219
p_mdi_sfp = 0.2405337005325426
# Idle (0C) - Offset
p_idle = 24.5973191097385-p_mdi_dac*48-0.6649290744404355

mean_packet_size = 1125.5
# median 1125.5, varianz 605.8795823674454

scenarios = []
for br in range(0, 1001, 80):
    scenarios.append([1, br, 64])
scenarios.append([1,1000,64])
base_paths = ["12_ports", "24_ports", "48_ports", "12_ports_sfp", "24_ports_sfp", "48_ports_sfp"]
labels = ["12 DAC", "24 DAC", "48 DAC", "12 BX", "24 BX", "48 BX"]
label_colors = [0, 1, 2, 3, 4, 5]


def transform_points(transformation_matrix, point):
    res = list(np.array(transformation_matrix).dot(np.array([point[0], point[1], 1])))
    return [res[0] / res[2], res[1] / res[2]]


def pps(rb, ps):
    return ((rb * 1000000) / (ps * 8)) / 1000000


def p_env(x, a, b, c, t):
    return (a * x) ** 3 + (b * x) ** 2 + c * x + t


def model_results(br, pps, RPM, p_mdi, p_idle, M, spline):
    transformed_coords = transform_points(M, [br, pps])
    p_load = spline.ev(transformed_coords[0], transformed_coords[1])
    return (p_idle + p_load + p_mdi + p_env(RPM, a_env, b_env, c_env, t_env))


def measurement_uncertainty(std_dev_real_power):
    measurement_uncertainty = (((0.001 * 300) * (0.001 * 50) * 2) ** 2) * 2 + 0.16883344575946913 ** 2
    measurement_uncertainty = math.sqrt(measurement_uncertainty + std_dev_real_power ** 2)
    return measurement_uncertainty


all_res = [[[] for i in range(len(scenarios))] for x in base_paths]
all_res_dev = [[[] for i in range(len(scenarios))] for x in base_paths]

for p_ind, base_path in enumerate(base_paths):
    for ind, currfile in enumerate(scenarios):
        for currun in range(5):
            print(str(base_path) + "/results_" + str(ind) + "_" + str(currun) + "dev_metrics.json")
            with open(str(base_path) + "/results_" + str(ind) + "_" + str(currun) + "dev_metrics.json") as jsonfile:
                strfile = str(jsonfile.read())
                strfile = strfile.split("}]}")
                try:
                    data_dev = json.loads(str(strfile[-2] + "}]}"))
                except:
                    data_dev = {}
            print(str(base_path) + "/results_" + str(ind) + "_" + str(currun) + ".json")
            with open(str(base_path) + "/results_" + str(ind) + "_" + str(currun) + ".json") as jsonfile:
                data = dict(json.load(jsonfile))
            for curr in range(len(data["Time"])):
                ts = data["Time"][curr] - data["Start_Time"][0]
                if ts > 10:
                    try:
                        all_res[p_ind][ind].append(float((data["Real_Power"][curr])))
                        all_res_dev[p_ind][ind].append(float((data_dev["RPM"][curr]["1"])))
                    except:
                        pass

# Bitrate: x-axis, Packet size: y-axis
pts1 = np.float32([[50, pps(50, 1500)], [50, pps(50, 64)], [1000, pps(1000, 1500)], [1000, pps(1000, 64)]])
pts2 = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])

M = cv.getPerspectiveTransform(pts1, pts2)
M2 = cv.getPerspectiveTransform(pts2, pts1)

with open('spline.pkl', 'rb') as f:
    spline = pickle.load(f)

rpm_values = []
rpm_diffs = []
for ports, x in enumerate(all_res_dev):
    rpms = []
    rpm_diff = []
    for br, x2 in enumerate(x):
        try:
            rpm_diff.append(np.max(x2) - np.min(x2))
            rpms.append(np.mean(x2))
        except:
            pass
    rpm_values.append(rpms)
    rpm_diffs.append(rpm_diff)
print(min([min(x)for x in rpm_values]))
print(max([max(x)for x in rpm_values]))
print(p_env(max([max(x)for x in rpm_values]),a_env, b_env, c_env, t_env)-p_env(min(min(x)for x in rpm_values),a_env, b_env, c_env, t_env))
print()
print(rpm_diffs)

real_power_means = []
real_power_measurement_uncertainties = []
for sc, x in enumerate(all_res):
    real_power_mean = []
    real_power_measurement_uncertainty = []
    for br, x2 in enumerate(x):
        real_power_mean.append(np.mean(x2))
        real_power_measurement_uncertainty.append(measurement_uncertainty(np.std(x2)))
    real_power_means.append(real_power_mean)
    real_power_measurement_uncertainties.append(real_power_measurement_uncertainty)

model_power = []
for ports, label in enumerate(base_paths):
    model_res = []
    for sc, param in enumerate(scenarios):
        try:
            RPM_value = rpm_values[ports][sc]
        except:
            RPM_value = np.mean(rpm_values[ports])
        br = param[1]
        curr_pps = pps(br, mean_packet_size)
        num_ports = int(label.split("_")[0])
        if label.find("sfp") != -1:
            p_mdi = p_mdi_sfp
        else:
            p_mdi = p_mdi_dac
        p_idle_curr = p_idle - p_mdi_dac * num_ports
        model_res.append(model_results(br, curr_pps, RPM_value, p_mdi * num_ports, p_idle_curr, M, spline))
    model_power.append(model_res)

power_diffs = list(np.array(real_power_means) - np.array(model_power))
print(power_diffs)

plt.rcParams.update({'font.size': 26})
plt.rcParams['figure.figsize'] = [10, 5]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0, 1, len(base_paths)))

legend_handles = []
for ind, curr in enumerate(power_diffs):
    plt.plot(model_power[ind], color=colors[label_colors[ind]], ls="--")
    plt.plot(real_power_means[ind], color=colors[label_colors[ind]], label=labels[ind])
    plt.fill_between([i for i in range(len(real_power_means[ind]))],
                     [real_power_means[ind][i] - real_power_measurement_uncertainties[ind][i] for i in
                      range(len(real_power_means[ind]))],
                     [real_power_means[ind][i] + real_power_measurement_uncertainties[ind][i] for i in
                      range(len(real_power_means[ind]))], color=colors[label_colors[ind]], alpha=0.5)
    legend_handles.append(Patch(color=colors[label_colors[ind]], label=labels[ind]))

plt.xlabel('applied bit rate [MBit/s]')
plt.ylabel('real power [W]')
# ax.set_xticklabels(Ticklabels)
# plt.xscale('log')
#plt.ylim(25,27)
plt.grid(which='major')
plt.legend(handles=legend_handles, loc="lower right", ncol=1, bbox_to_anchor=(1.51, 0))
plt.gcf().subplots_adjust(bottom=0.17)
plt.gcf().subplots_adjust(left=0.15)
#plt.gcf().subplots_adjust(top=0.8)
plt.gcf().subplots_adjust(top=0.95)
plt.gcf().subplots_adjust(right=0.72)
tick_range=[0]
for curr in range(1,len(scenarios),2):
    tick_range.append(curr)
ax.set_xticks([i for i in tick_range], [scenarios[i][1] for i in tick_range])
#plt.xticks(rotation=30)
ax.grid(which='major', alpha=0.2)
# plt.show()
plt.savefig('realistic_traffic.pdf')
plt.close()

mses = []
rmses = []
maes = []
mapes = []
max_errors = []
max_uncertainties=[]
for curr in real_power_measurement_uncertainties:
    max_uncertainties.append(np.max(curr))
print(max_uncertainties)
num_exceeding=0
num_total=0
max_cons=[]
for ind, sc in enumerate(base_paths):
    mse = (1 / len(real_power_means[ind])) * np.nansum(
        [(real_power_means[ind][x] - model_power[ind][x]) ** 2 for x in range(len(real_power_means[ind]))])
    diffs = power_diffs[ind]
    mses.append(mse)
    rmses.append(math.sqrt(mse))
    maes.append((1 / len(real_power_means[ind])) * np.nansum(
        [abs(real_power_means[ind][x] - model_power[ind][x]) for x in range(len(real_power_means[ind]))]))
    mapes.append((1 / len(real_power_means[ind])) * np.nansum(
        [abs((real_power_means[ind][x] - model_power[ind][x]) / real_power_means[ind][x]) for x in
         range(len(real_power_means[ind]))])*100)
    max_errors.append((np.nanmax(np.abs(diffs))))
    for ind2 in range(len(real_power_means[ind])):
        if np.abs(real_power_means[ind][ind2] - model_power[ind][ind2])>max_uncertainties[ind]:
            num_exceeding += 1
        num_total += 1
    max_cons.append(np.nanmax(real_power_means[ind]))
print("MSEs:" + str(mses))
print("RMSEs:" + str(rmses))
print("MAEs:" + str(maes))
print("MAPEs:" + str(mapes))
print("Max Errors:" + str(max_errors))
print(1-num_exceeding/num_total)
print(max_cons)