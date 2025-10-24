import json
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats as st

plt.rcParams.update({'font.size': 24})
plt.rcParams['figure.figsize'] = [10, 5]
fig, ax = plt.subplots()

scenarios=[[1,x,128] for x in range(0,1001,10)]
scenarios = scenarios[1:-1]
colors = plt.cm.copper(np.linspace(0, 1, 6))

def monoExp(x, a, b,t):
    return a * np.exp(b * x)+t

a, b,t = 1.3249078669796963, 0.00016745066619824263, 23.978159637075045

def plot_basic(base_path,scale,label,color):
    all_res=[[]for i in range(len(scenarios))]
    for ind, currfile in enumerate(scenarios):
        for currun in [0,1]:

            with open(str(base_path) + "/results_" + str(ind) +"_"+str(currun)+ "dev_metrics.json") as jsonfile:
                data_dev = dict(json.load(jsonfile))
            with open(str(base_path) + "/results_" + str(ind) +"_"+str(currun)+ ".json") as jsonfile:
                data = dict(json.load(jsonfile))
            for curr in range(len(data["Time"])):
                ts = data["Time"][curr] - data["Start_Time"][0]
                if ts>10:
                    try:
                        all_res[ind].append(float((data["Real_Power"][curr])))
                        #all_res[ind].append(monoExp(float((data_dev["RPM"][curr]["1"])),a,b,t))
                        #all_res[ind].append(float((data_dev["RPM"][curr]["1"])))
                    except:
                        pass
    plt.plot( [x[1]*scale for x in scenarios],[np.mean(x) for x in all_res], color=color,label=label)
    #confidence = [st.bootstrap((res,), statistic=np.mean, confidence_level=0.99).confidence_interval for res in all_res]
    #ax.fill_between([x[1]*scale for x in scenarios], [i[1] for i in confidence], [i[0] for i in confidence], color=colors[0], alpha=0.5)
    #ax.fill_between([x[1]*scale for x in scenarios], [i[1] for i in confidence], [i[0] for i in confidence], color=colors[0], alpha=0.5)

    #plt.plot([x[1] for x in scenarios], [np.mean(x) for x in all_res], color=colors[0])
    #confidence = [st.bootstrap((res,), statistic=np.mean, confidence_level=0.99).confidence_interval for res in all_res]
    #ax.fill_between([x[1] for x in scenarios], [i[1] for i in confidence], [i[0] for i in confidence], color=colors[0], alpha=0.5)

    measurement_uncertainty=((0.001 * 300) * (0.001 * 50) * 2)**2 + 0.092544**2
    uncertainty=[[i-math.sqrt(measurement_uncertainty+np.std(all_res[curr])**2),i+math.sqrt(measurement_uncertainty+np.std(all_res[curr])**2)] for ind,i in enumerate([np.mean(x) for x in all_res])]
    ax.fill_between([x[1]*scale for x in scenarios], [i[1] for i in uncertainty], [i[0] for i in uncertainty], color=color, alpha=0.25)

    #coef = np.polyfit([x[1] for x in scenarios], [x[0] for x in uncertainty], 1)
    #poly1d_fn = np.poly1d(coef)
    #plt.plot([x[1]*scale for x in scenarios], poly1d_fn([x[1] for x in scenarios]), color=colors[3],label="Reference")

ports=["4_ports"]
multiplicator=[4,6,12,24,48]
labels=[4,6,12,24,48]
#ports=["12_ports"]
#multiplicator=[12,24]
for curr in range(len(ports)):
    plot_basic(ports[curr],multiplicator[curr],labels[curr],colors[curr])

plt.grid(which='major')
#plt.legend(loc="lower right",ncol=4,bbox_to_anchor=(1.1,1))
plt.gcf().subplots_adjust(bottom=0.17)
plt.gcf().subplots_adjust(left=0.15)
#plt.gcf().subplots_adjust(top=0.8)
plt.gcf().subplots_adjust(top=0.95)
plt.gcf().subplots_adjust(right=0.95)
plt.ylabel("real power [W]")
plt.xlabel("carried bit rate [MBit/s]")
ax.grid(which='major', alpha=0.2)
#plt.show()
plt.savefig('Pre_study_R_b.pdf')
plt.close()
