import csv
import json
import math

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

from Waystream_SFP_Comparison.plot_ee_comparison_waystream_reference_SFP_only import max_uncertainty

plt.rcParams.update({'font.size': 26})
plt.rcParams['figure.figsize'] = [10, 6]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0,1,5))
max_uncertainty=(0.001*300)*(0.001*50)*2



def get_results(path,runs,baseline):
    all_results=[]
    for currfile in runs:
        current_results=[]
        with open(str(path)+str(currfile) + ".json") as jsonfile:
            data = dict(json.load(jsonfile))
        for curr in range(len(data["Voltage"])):
            if(data["Time"][curr]-data["Start_Time"][0])>130:
                all_results.append((data["Real_Power"][curr]))
                #all_results.append((data["Real_Power"][curr] - baseline) / 2)
                current_results.append(data["Real_Power"][curr])
        #print(str(currfile)+": "+str(np.mean(current_results)))
    return all_results


configs=["Idle","125MBit/s","500 MBit/s","1000 MBit/s"]
results=[]
baseline=[]
toexport={}

# Baseline for DAC Cable based on specs: 30AWG, 0.5m -> 0.1647993 Ohm resistance -> 0.014832W assuming 300mA (maximum receive current for SFP)

#results.append(get_results(["results_idle_0","results_idle_1","results_idle_2","results_idle_3","results_idle_4"]))
#results.append(get_results(["results_0_0","results_0_1","results_0_2","results_0_3","results_0_4"]))
#results.append(get_results(["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"]))
#results.append(get_results(["results_2_0","results_2_1","results_2_2","results_2_3","results_2_4"]))
baseline.append(get_results("./results_DAC/",["results_idle_0","results_idle_4"],0.014832))
baseline.append(get_results("./results_DAC/",["results_0_0","results_0_1","results_0_3","results_0_4"],0.014832))
baseline.append(get_results("./results_DAC/",["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"],0.014832))
baseline.append(get_results("./results_DAC/",["results_2_0","results_2_3"],0.014832))

baseline_means=[]
for curr in baseline:
    baseline_means.append(np.mean(curr)*2)

results.append(get_results("./results_LR/",["results_idle_0","results_idle_2","results_idle_4"],baseline_means[0]))
results.append(get_results("./results_LR/",["results_0_0","results_0_3"],baseline_means[1]))
results.append(get_results("./results_LR/",["results_1_1","results_1_2","results_1_3","results_1_4"],baseline_means[2]))
results.append(get_results("./results_LR/",["results_2_0","results_2_2"],baseline_means[3]))

confidence_intervals=[st.bootstrap([x], statistic=np.mean, confidence_level=0.95).confidence_interval for x in results]
#toexport["DAC"]=[float(np.mean(x)) for x in baseline_means]
toexport["LR"]=[float(np.mean(x)) for x in results]
ci=[[],[]]
for curr in confidence_intervals:
    ci[0].append(curr[0])
    ci[1].append(curr[1])
yerr = [np.array([np.mean(x) for x in results]) - np.array(ci[0]), np.array(ci[1]) - np.array([np.mean(x) for x in results])]
plt.bar([0,1,2,3],[np.mean(x) for x in results],width=0.8,color=colors[3],label="Regular SFP")
plt.errorbar([0,1,2,3],[np.mean(x) for x in results],yerr=yerr,fmt='none',ecolor="black",elinewidth=20)

results=[]
results.append(get_results("./results_LR_Waystream/",["results_idle_0","results_idle_1","results_idle_2","results_idle_4"],baseline_means[0]))
results.append(get_results("./results_LR_Waystream/",["results_0_0","results_0_1","results_0_3","results_0_4"],baseline_means[1]))
results.append(get_results("./results_LR_Waystream/",["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"],baseline_means[2]))
results.append(get_results("./results_LR_Waystream/",["results_2_0","results_2_1","results_2_2"],baseline_means[3]))

confidence_intervals=[st.bootstrap([x], statistic=np.mean, confidence_level=0.95).confidence_interval for x in results]
toexport["LR_waystream"]=[float(np.mean(x)) for x in results]
ci=[[],[]]
for curr in confidence_intervals:
    ci[0].append(curr[0])
    ci[1].append(curr[1])
yerr = [[math.sqrt((((ci[0][0]-np.mean(results[0]))**2+(max_uncertainty)**2))/2)], [math.sqrt((((ci[0][1]-np.mean(results[0]))**2+(max_uncertainty)**2))/2)]]
plt.bar([0,1,2,3],[np.mean(x) for x in results],width=0.8,color=colors[1],label="Waystream SFP")
plt.errorbar([0,1,2,3],[np.mean(x) for x in results],yerr=yerr,fmt='none',ecolor="black",elinewidth=20)



plt.axhline(0.55,color="red")
plt.xticks([0,1,2,3],configs)
plt.grid(which='major')
plt.legend(loc="lower right",ncol=2,bbox_to_anchor=(1.1,1),fontsize=26)#, handleheight=0.9, labelspacing=0.2, handlelength=0.7,fontsize=13)#,
plt.gcf().subplots_adjust(bottom=0.22)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(top=0.85)
plt.ylabel("Real Power [W]")
plt.xticks(rotation=25)
ax.grid(which='major', alpha=0.2)
#plt.show()
plt.savefig('LR_power_comparison.pdf')
plt.close()


results=[]
results.append(get_results("./results_BX/",["results_idle_1","results_idle_4"],baseline_means[0]))
results.append(get_results("./results_BX/",["results_0_0","results_0_1","results_0_2","results_0_3","results_0_4"],baseline_means[1]))
results.append(get_results("./results_BX/",["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"],baseline_means[2]))
results.append(get_results("./results_BX/",["results_2_1","results_2_4"],baseline_means[3]))

confidence_intervals=[st.bootstrap([x], statistic=np.mean, confidence_level=0.95).confidence_interval for x in results]
toexport["BX"]=[float(np.mean(x)) for x in results]
ci=[[],[]]
for curr in confidence_intervals:
    ci[0].append(curr[0])
    ci[1].append(curr[1])
yerr = [np.array([np.mean(x) for x in results]) - np.array(ci[0]), np.array(ci[1]) - np.array([np.mean(x) for x in results])]
plt.bar([0,1,2,3],[np.mean(x) for x in results],width=0.8,color=colors[3],label="Regular SFP")
plt.errorbar([0,1,2,3],[np.mean(x) for x in results],yerr=yerr,fmt='none',ecolor="black",elinewidth=20)

results=[]
results.append(get_results("./results_BX_Waystream/",["results_idle_0","results_idle_1","results_idle_2","results_idle_3"],baseline_means[0]))
results.append(get_results("./results_BX_Waystream/",["results_0_1","results_0_3"],baseline_means[1]))
results.append(get_results("./results_BX_Waystream/",["results_1_1","results_1_2","results_1_3"],baseline_means[2]))
results.append(get_results("./results_BX_Waystream/",["results_2_0","results_2_1","results_2_2"],baseline_means[3]))

confidence_intervals=[st.bootstrap([x], statistic=np.mean, confidence_level=0.95).confidence_interval for x in results]
toexport["BX_waystream"]=[float(np.mean(x)) for x in results]
ci=[[],[]]
for curr in confidence_intervals:
    ci[0].append(float(curr[0])-max_uncertainty)
    ci[1].append(float(curr[1])+max_uncertainty)
print(ci)
yerr = [np.array([np.mean(x) for x in results]) - np.array(ci[0]), np.array(ci[1]) - np.array([np.mean(x) for x in results])]
print(yerr)
plt.bar([0,1,2,3],[np.mean(x) for x in results],width=0.8,color=colors[1],label="Waystream SFP")
plt.errorbar([0,1,2,3],[np.mean(x) for x in results],yerr=yerr,fmt='none',ecolor="black",elinewidth=20)



plt.axhline(0.3,color="red")
plt.xticks([0,1,2,3],configs)
plt.grid(which='major')
plt.legend(loc="lower right",ncol=2,bbox_to_anchor=(1.1,1),fontsize=26)#, handleheight=0.9, labelspacing=0.2, handlelength=0.7,fontsize=13)#,
plt.gcf().subplots_adjust(bottom=0.22)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(top=0.85)
plt.ylabel("Real Power [W]")
plt.xticks(rotation=25)
ax.grid(which='major', alpha=0.2)
#plt.show()
#plt.savefig('BX_power_comparison.pdf')
plt.close()
print(toexport)
with open('dict.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["module_type","idle","125","500","1000"])
    for key, value in toexport.items():
        line=[key]
        for curr in value:
            line.append(curr)
        writer.writerow(line)