import csv
import json
import math
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import SmoothBivariateSpline

random.seed(42)
# P_env Basisparameter
a_env=-0.00011215736513100496
b_env=0.0002996332246959438
c_env=-0.00017340552076487947
t_env=0

p_mdi=(8*0.035951219512195154)
# Idle (0C) - P_mdi - Offset
p_idle = 1.6079999999999999-p_mdi


def p_env(x,a,b,c,t):
    return (a * x) ** 3 + (b * x) ** 2 + c * x + t



def transform_points(transformation_matrix, point):
    res = list(np.array(transformation_matrix).dot(np.array([point[0], point[1], 1])))
    return [res[0] / res[2], res[1] / res[2]]


def pps(rb, ps):
    return ((rb * 1000000) / (ps * 8)) / 1000000


all_scenarios = [[0, 0, 64]]
curr_scenario = 1
for br in range(50, 201, 10):
    for ps in range(64, 129, 20):
        all_scenarios.append([curr_scenario, br*8, ps])
        curr_scenario += 1
for br in range(50, 1001, 50):
    for ps in range(128, 1501, 20):
        all_scenarios.append([curr_scenario, br*8, ps])
        #print(str(br*8))
        curr_scenario += 1
base_path = "p_load"


random.shuffle(all_scenarios)
scenarios = all_scenarios[:int(len(all_scenarios)*0.7)]
eval_scenarios = all_scenarios[int(len(all_scenarios)*0.7):]
#scenarios = all_scenarios[:int(len(all_scenarios)*1)]
#eval_scenarios = all_scenarios[int(len(all_scenarios)*0):]

print(len(all_scenarios))
print(len(scenarios))
print(len(eval_scenarios))

all_res = [[] for i in range(len(scenarios))]
all_res_dev = [[] for i in range(len(scenarios))]

for ind, currfile in enumerate(scenarios):
    for currun in range(5):
        #print(str(base_path) + "/results_" + str(currfile[0]) + "_" + str(currun) + "dev_metrics.json")
        #with open(str(base_path) + "/results_" + str(currfile[0]) + "_" + str(currun) + "dev_metrics.json") as jsonfile:
        #    strfile = str(jsonfile.read())
        #    strfile = strfile.split("}]}")
        #    try:
        #        data_dev = json.loads(str(strfile[-2] + "}]}"))
        #    except:
        #        data_dev = {}
        #print(str(base_path) + "/results_" + str(currfile[0]) + "_" + str(currun) + ".json")
        with open(str(base_path) + "/results_" + str(currfile[0]) + "_" + str(currun) + ".json") as jsonfile:
            data = dict(json.load(jsonfile))
        for curr in range(len(data["Time"])):
            ts = data["Time"][curr] - data["Start_Time"][0]
            if ts > 10:
                try:
                    # P_env beinhaltet bereits P_idle, 8*P_mdi
                    all_res[ind].append(float((data["Real_Power"][curr])-p_idle)-p_mdi)
                    #all_res_dev[ind].append(float((data_dev["RPM"][curr]["1"])))
                except:
                    pass
print("Mean RPM: " + str(np.mean([x for xs in all_res_dev for x in xs])))
print("Idle: "+str(np.mean(all_res[0])))

adjusted=[np.mean(x) for x in all_res]

max_diff=[]
for x in all_res:
    if(len(x)>0):
        max_diff.append(np.mean(x))
    else:
        max_diff.append(-float("inf"))

max_diff=np.max(max_diff)
min_diff=[]
for x in all_res:
    if(len(x)>0):
        min_diff.append(np.mean(x))
    else:
        min_diff.append(float("inf"))
min_diff=np.min(min_diff)
print("Max Diff: "+str(max_diff-min_diff))
print("Offset: "+str(np.min(adjusted)))
# Idle: 26.132
# plt.plot(list(range(len(scenarios))), [np.mean(x) for x in all_res_dev])
# plt.plot(list(range(len(scenarios))), [np.mean(x) for x in all_res], label="All")


# without perspective transformation
br_range = [x*8 for x in list(range(50, 1001, 50))]
ps_range = list(range(128, 1501, 20))
max_extend = [[0 for br in br_range] for ps in ps_range]
for ind, curr in enumerate(scenarios):
    try:
        max_extend[ps_range.index(curr[2])][br_range.index(curr[1])] = np.mean(all_res[ind])
    except:
        pass

###### PLOT 1: Raw Data

plt.rcParams.update({'font.size': 26})
plt.rcParams['figure.figsize'] = [10, 6]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0,1,5))

c = plt.pcolor(max_extend,cmap="copper")
plt.colorbar(c, label='real power [W]')

plt.xlabel('carried bit rate [MBit/s]')
plt.ylabel('frame size [B]')

xlabels=[]
xtick_positions=list(range(0,len(br_range),2))
for curr in xtick_positions:
    xlabels.append(int(br_range[curr]))
ylabels=[]
ytick_positions=list(range(0,len(ps_range),6))
for curr in ytick_positions:
    ylabels.append(str(ps_range[curr]))

xlabels=["2000","4000","6000","8000"]
xtick_positions=[]
for curr in xlabels:
    val = int(curr)
    ratio=(max(br_range)-min(br_range))/(len(br_range))
    xtick_positions.append((val-min(br_range))/ratio)

ylabels=["128","256","512","1024","1500"]
ytick_positions=[]
for curr in ylabels:
    val = int(curr)
    ratio=(max(ps_range)-min(ps_range))/(len(ps_range))
    ytick_positions.append((val-min(ps_range))/ratio)

ax.set_xticks(xtick_positions,labels=xlabels)
ax.set_yticks(ytick_positions,labels=ylabels)
#plt.gca().xaxis.set_major_locator(MaxNLocator(prune='lower'))
#plt.gca().yaxis.set_major_locator(MaxNLocator(prune='lower'))
#plt.xticks(rotation=25)
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(top=0.75)
#plt.show()
plt.savefig('raw_measurements.pdf')
plt.close()


# Bitrate: x-axis, Packet size: y-axis
pts1 = np.float32([[50*8, pps(50*8, 1500)], [50*8, pps(50*8, 64)], [1000*8, pps(1000*8, 1500)], [1000*8, pps(1000*8, 64)]])
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
std_devs=[]
for curr in all_res:
    std_devs.append(np.std(curr))

measurement_uncertainty=(((0.001 * 300) * (0.001 * 50) * 2)**2)*2
uncertainty=math.sqrt(measurement_uncertainty+np.max(std_devs)**2)
print("Uncertainty: " + str(uncertainty))
plt.scatter(brs,prs)
#plt.show()
plt.close()

spline = SmoothBivariateSpline(brs, prs, real_powers, kx=1, ky=1,s=1-uncertainty)
x_new = np.linspace(0, 1, 1000)  # 100 points from 0 to 10 on x-axis
y_new = np.linspace(0, 1, 1000)  # 100 points from 0 to 10 on y-axis
# Interpolate Z values on the new grid
Z_new = spline(x_new, y_new)
plt.pcolormesh(np.meshgrid(x_new, y_new)[1], np.meshgrid(x_new, y_new)[0], Z_new, shading='auto', cmap='copper')
plt.colorbar(label='real rower [W]')
knots = spline.get_knots()
knotx = knots[0]
knoty = knots[1]
for ind in range(min([len(knoty),len(knotx)])):
    coords = [knotx[ind], knoty[ind]]
    print(coords)
    plt.scatter(coords[0], coords[1], s=50, c="blue")
plt.xlabel('Applied Bitrate')
plt.ylabel('Processed Packet Rate')
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig('Interpolation_Mapping.png',dpi=600)
plt.close()

combined=[]
for ind, curr in enumerate(scenarios):
    if (curr[2]) > 127:
        transformed_coords=transform_points(M, [curr[1], pps(curr[1], curr[2])])
        combined.append([np.mean(all_res[ind]),spline.ev(transformed_coords[0],transformed_coords[1]),curr[1], pps(curr[1], curr[2]),np.mean(all_res[ind])-spline.ev(transformed_coords[0],transformed_coords[1])])

print(max(combined,key=lambda x:x[4]))
residuals=np.array([x[0]-x[1] for x in combined])
z = np.array([x[0] for x in combined])
ss_res = np.sum((residuals) ** 2)
ss_tot = np.sum((z - np.mean(z)) ** 2)
r_squared = 1 - (ss_res / ss_tot)
print("R^2: "+str(r_squared))


data = np.array(combined)

# Separate columns
x, y, values = data[:, 2], data[:, 3], data[:, 4]
plt.scatter(x, y, c=values, cmap='copper', s=10, marker='s')
plt.colorbar(label='residuals [W]')

plt.xlabel('carried bit rate [MBit/s]')
plt.ylabel('processed frame rate [Mpps]')
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig("residuals.png", dpi=600)
plt.close()


combined = []
combined_export=[]

for br in range(50, 1000):
    for ps in range(128, 1500):
        transformed_coords = transform_points(M, [br*8, pps(br*8, ps)])
        combined.append([br*8, pps(br*8, ps), spline.ev(transformed_coords[0], transformed_coords[1])])
        combined_export.append([br*8, pps(br*8, ps), spline.ev(transformed_coords[0], transformed_coords[1])])

with open('model_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(combined_export)

data = np.array(combined)

# Separate columns
x, y, values = data[:, 0], data[:, 1], data[:, 2]
plt.scatter(x, y, c=values, cmap='copper', s=10, marker='s')
plt.colorbar(label='real power [W]')

knots = spline.get_knots()
knotx = knots[0]
knoty = knots[1]
for ind in range(min([len(knoty),len(knotx)])):
    coords = transform_points(M2, [knotx[ind], knoty[ind]])
    print(coords)
    plt.scatter(coords[0], coords[1], s=2, c="blue")






plt.xlabel('carried bit rate [MBit/s]')
plt.ylabel('processed frame rate [Mpps]')
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig("final_interpolation.png", dpi=600)
plt.close()

### Model Evaluation

all_res = [[] for i in range(len(eval_scenarios))]
all_res_dev = [[] for i in range(len(eval_scenarios))]

for ind, currfile in enumerate(eval_scenarios):
    for currun in range(5):
        #with open(str(base_path) + "/results_" + str(currfile[0]) + "_" + str(
        #        currun) + "dev_metrics.json") as jsonfile:
        #    strfile = str(jsonfile.read())
        #    strfile = strfile.split("}]}")
        #    try:
        #        data_dev = json.loads(str(strfile[-2] + "}]}"))
        #    except:
        #        data_dev = {}
        with open(str(base_path) + "/results_" + str(currfile[0]) + "_" + str(currun) + ".json") as jsonfile:
            data = dict(json.load(jsonfile))
        for curr in range(len(data["Time"])):
            ts = data["Time"][curr] - data["Start_Time"][0]
            if ts > 10 and currfile[2]>127:
                try:
                    # P_env beinhaltet bereits P_idle, 8*P_mdi
                    all_res[ind].append(                        float((data["Real_Power"][curr])))
                    #all_res_dev[ind].append(float((data_dev["RPM"][curr]["1"])))
                except:
                    pass
measurement_results=[np.nanmean(x) for x in all_res]
model_results=[]
for ind, currfile in enumerate(eval_scenarios):
    br=currfile[1]
    ps=currfile[2]
    transformed_coords = transform_points(M, [br, pps(br, ps)])
    p_load=spline.ev(transformed_coords[0], transformed_coords[1])
    model_results.append(p_idle+p_load+p_mdi)
plt.plot(model_results,color='red')
plt.plot(measurement_results,color='blue')
#plt.show()
plt.close()
diff=[]
for ind, currfile in enumerate(eval_scenarios):
    br=currfile[1]
    ps=currfile[2]
    diff.append([br,pps(br,ps),measurement_results[ind] - model_results[ind]])
data = np.array(diff)

# Separate columns
x, y, values = data[:, 0], data[:, 1], data[:, 2]
plt.scatter(x, y, c=values, cmap='copper', s=50, marker='o')
plt.colorbar(label=r'$\Delta$ Measurements-Model [W]')
plt.xlabel('Applied Bitrate [MBit/s]')
plt.ylabel('Processed Packet Rate [Mpps]')
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig("evaluation.pdf")
plt.close()

plt.scatter(measurement_results,model_results,color='black')
plt.plot([x/10 for x in range(int(min(measurement_results))*10,math.ceil(max(measurement_results))*10)],[x/10 for x in range(int(min(measurement_results))*10,math.ceil(max(measurement_results))*10)],color="red")
plt.xlim(min(measurement_results),max(measurement_results))
plt.ylim(min(model_results),max(model_results))
plt.xlabel('measured power [W]')
plt.ylabel('predicted power [W]')
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig("comparison.pdf")
plt.close()



mse=(1/len(measurement_results))*np.nansum([(measurement_results[x]-model_results[x])**2 for x in range(len(measurement_results))])
diffs=[(measurement_results[x]-model_results[x]) for x in range(len(measurement_results))]
print("MSE: "+str(mse))
print("RMSE: "+str(math.sqrt(mse)))
print("MAE: "+str((1/len(measurement_results))*np.nansum([abs(measurement_results[x]-model_results[x]) for x in range(len(measurement_results))])))
print("MAPE: "+str((1/len(measurement_results))*np.nansum([abs((measurement_results[x]-model_results[x])/measurement_results[x]) for x in range(len(measurement_results))])))
print("Max Error: "+str(np.nanmax(np.abs(diffs))))

num_outliers=0
for curr in diffs:
    if np.abs(curr)>uncertainty:
        num_outliers+=1
print("Not Outlier Percentage: "+str(100-(num_outliers/len(diffs))*100))

model_results_traffic=model_results
measurement_results_traffic=measurement_results

model_response = np.array(model_results)
comparison_response = np.array(measurement_results)

def ecdf(data):
    sorted_data = np.sort(data)
    n = len(data)
    return sorted_data, np.arange(1, n+1) / n

# Compute ECDFs
model_sorted, model_ecdf = ecdf(model_response)
comparison_sorted, comparison_ecdf = ecdf(comparison_response)

# Interpolate: for each quantile, find the value in the ECDF
# Then plot ECDF(model_response_values) vs ECDF(comparison_response_values)

# We'll use the sorted values of the shared quantiles
# and compute their ECDF values in both distributions

# To use common quantiles, sort both datasets and assume they align
# This only works properly when both lists are the same length
if len(model_response) != len(comparison_response):
    raise ValueError("For a P–P plot, both datasets must have the same number of samples.")

n = len(model_response)
quantiles = np.linspace(0, 1, n, endpoint=False) + 0.5 / n  # midpoints

# Get the percentile values from both distributions
model_quantile_values = np.percentile(model_response, quantiles * 100)
comparison_quantile_values = np.percentile(comparison_response, quantiles * 100)

# Now compute the ECDF values of those quantile values in each distribution
def get_ecdf_values(data, values):
    sorted_data = np.sort(data)
    return np.searchsorted(sorted_data, values, side='right') / len(data)

model_cdf_vals = get_ecdf_values(model_response, model_quantile_values)
comparison_cdf_vals = get_ecdf_values(comparison_response, model_quantile_values)

# Plot P–P plot
plt.plot(model_cdf_vals, comparison_cdf_vals, 'o', label='P–P Plot',color=colors[0])
plt.plot([0, 1], [0, 1], 'r--', label='y = x')

plt.xlabel('model ECDF')
plt.ylabel('measurement ECDF')
plt.legend()
plt.grid(True)
#plt.axis('square')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.gcf().subplots_adjust(bottom=0.2)
plt.gcf().subplots_adjust(left=0.15)
plt.gcf().subplots_adjust(top=0.8)
plt.savefig("p-p-plot.pdf")
plt.close()