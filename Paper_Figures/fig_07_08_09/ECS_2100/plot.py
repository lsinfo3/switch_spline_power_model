import json

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import SmoothBivariateSpline


def calculate_transformation_matrix(p1_orig, p2_orig, p3_orig, p1_new, p2_new, p3_new):
    x1 = p1_orig[0]
    x2 = p2_orig[0]
    x3 = p3_orig[0]
    y1 = p1_orig[1]
    y2 = p2_orig[1]
    y3 = p3_orig[1]
    x1h = p1_new[0]
    x2h = p2_new[0]
    x3h = p3_new[0]
    y1h = p1_new[1]
    y2h = p2_new[1]
    y3h = p3_new[1]
    d = x1 * (y2 - y3) - x2 * (y1 - y3) + x3 * (y1 - y2)
    transform_matrix = [[0, 0], [0, 0]]
    offset_matrix = [0, 0]
    transform_matrix[0][0] = -(-x1h * (y2 - y3) + x2h * (y1 - y3) - x3h * (y1 - y2)) / d
    transform_matrix[0][1] = -(-y1h * (y2 - y3) + y2h * (y1 - y3) - y3h * (y1 - y2)) / d
    transform_matrix[1][0] = (-x1h * (x2 - x3) + x2h * (x1 - x3) - x3h * (x1 - x2)) / d
    transform_matrix[1][1] = (-y1h * (x2 - x3) + y2h * (x1 - x3) - y3h * (x1 - x2)) / d
    offset_matrix[0] = -(-x1h * (x2 * y3 - y2 * x3) + x2h * (x1 * y3 - y1 * x3) - x3h * (x1 * y2 - y1 * x2)) / d
    offset_matrix[1] = -(-y1h * (x2 * y3 - y2 * x3) + y2h * (x1 * y3 - y1 * x3) - y3h * (x1 * y2 - y1 * x2)) / d
    return transform_matrix, offset_matrix


def transform_points(transformation_matrix, point):
    res = list(np.array(transformation_matrix).dot(np.array([point[0], point[1], 1])))
    return [res[0] / res[2], res[1] / res[2]]


def pps(rb, ps):
    return ((rb * 1000000) / (ps * 8)) / 1000000


scenarios = [[1, 0, 64]]
for br in range(50, 201, 10):
    for ps in range(64, 129, 20):
        scenarios.append([1, br, ps])
for br in range(50, 1001, 50):
    for ps in range(128, 1501, 20):
        scenarios.append([1, br, ps])
base_path = "1755073458"

all_res = [[] for i in range(len(scenarios))]
all_res_dev = [[] for i in range(len(scenarios))]

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
                    all_res[ind].append(float((data["Real_Power"][curr])))
                    all_res_dev[ind].append(float((data_dev["RPM"][curr]["1"])))
                except:
                    pass
# plt.plot(list(range(len(scenarios))), [np.mean(x) for x in all_res_dev])
# plt.plot(list(range(len(scenarios))), [np.mean(x) for x in all_res], label="All")


# without perspective transformation
br_range = list(range(50, 1001, 50))
ps_range = list(range(128, 1501, 20))
max_extend = [[0 for br in br_range] for ps in ps_range]
for ind, curr in enumerate(scenarios):
    try:
        max_extend[ps_range.index(curr[2])][br_range.index(curr[1])] = np.mean(all_res[ind])
    except:
        pass

###### PLOT 1: Raw Data

plt.rcParams.update({'font.size': 24})
plt.rcParams['figure.figsize'] = [10, 6]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0,1,5))

c = plt.pcolor(max_extend,cmap="copper")
plt.colorbar(c, label='Real Power')

plt.xlabel('Applied Bitrate [MBit/s]')
plt.ylabel('Applied Packet Size [B]')

xlabels=[]
xtick_positions=list(range(0,len(br_range),3))
for curr in xtick_positions:
    xlabels.append(str(br_range[curr]))
ylabels=[]
ytick_positions=list(range(0,len(ps_range),5))
for curr in ytick_positions:
    ylabels.append(str(ps_range[curr]))

ax.set_xticks(xtick_positions,labels=xlabels)
ax.set_yticks(ytick_positions,labels=ylabels)
#plt.xticks(rotation=70)
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
#plt.show()
plt.savefig('raw_measurements.pdf')
plt.close()


# Bitrate: x-axis, Packet size: y-axis
pts1 = np.float32([[50, pps(50, 1500)], [50, pps(50, 64)], [1000, pps(1000, 1500)], [1000, pps(1000, 64)]])
pts2 = np.float32([[0, 0], [0, 1], [1, 0], [1, 1]])

M = cv.getPerspectiveTransform(pts1, pts2)
M2 = cv.getPerspectiveTransform(pts2, pts1)

print(M)
combined = []
for ind, curr in enumerate(scenarios):
    if (curr[2]) > 127:
        combined.append([transform_points(M, [curr[1], pps(curr[1], curr[2])]), np.mean(all_res[ind])])
combined = sorted(combined, key=lambda x: x[0])
brs = [x[0][0] for x in combined]
prs = [x[0][1] for x in combined]
real_powers = [x[1] for x in combined]
spline = SmoothBivariateSpline(brs, prs, real_powers, kx=1, ky=1)
x_new = np.linspace(0, 1, 1000)  # 100 points from 0 to 10 on x-axis
y_new = np.linspace(0, 1, 1000)  # 100 points from 0 to 10 on y-axis
# Interpolate Z values on the new grid
Z_new = spline(x_new, y_new)
plt.pcolormesh(np.meshgrid(x_new, y_new)[1], np.meshgrid(x_new, y_new)[0], Z_new, shading='auto', cmap='copper')
plt.colorbar(label='Real Power')
knots = spline.get_knots()
knotx = knots[0]
knoty = knots[1]
for ind in range(len(knotx)):
    coords = [knotx[ind], knoty[ind]]
    print(coords)
    plt.scatter(coords[0], coords[1], s=50, c="red")
plt.xlabel('Applied Bitrate')
plt.ylabel('Processed Packet Rate')
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig('Interpolation_Mapping.png',dpi=600)
plt.close()

combined = []

for br in range(50, 1000):
    for ps in range(64, 1500):
        transformed_coords = transform_points(M, [br, pps(br, ps)])
        combined.append([br, pps(br, ps), spline.ev(transformed_coords[0], transformed_coords[1])])

data = np.array(combined)

# Separate columns
x, y, values = data[:, 0], data[:, 1], data[:, 2]
plt.scatter(x, y, c=values, cmap='copper', s=10, marker='s')
plt.colorbar(label='Real Power')

knots = spline.get_knots()
knotx = knots[0]
knoty = knots[1]
for ind in range(len(knotx)):
    coords = transform_points(M2, [knotx[ind], knoty[ind]])
    print(coords)
    plt.scatter(coords[0], coords[1], s=2, c="red")

plt.xlabel('Applied Bitrate [MBit/s]')
plt.ylabel('Processed Packet Rate [Mpps]')
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig("final_interpolation.png", dpi=600)
