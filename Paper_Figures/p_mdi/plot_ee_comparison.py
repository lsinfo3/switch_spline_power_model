import json

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np

plt.rcParams.update({'font.size': 26})
plt.rcParams['figure.figsize'] = [10, 6]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0,1,5))



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


configs=["Idle","125MBit/s","500 MBit/s","1000 MBit/s"]
configs_numeric=[0.1,125,500,1000]
results=[]
baseline=[]

# Baseline for DAC Cable based on specs: 30AWG, 0.5m -> 0.1647993 Ohm resistance -> 0.014832W assuming 300mA (maximum receive current for SFP)

#results.append(get_results(["results_idle_0","results_idle_1","results_idle_2","results_idle_3","results_idle_4"]))
#results.append(get_results(["results_0_0","results_0_1","results_0_2","results_0_3","results_0_4"]))
#results.append(get_results(["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"]))
#results.append(get_results(["results_2_0","results_2_1","results_2_2","results_2_3","results_2_4"]))
baseline.append(get_results("./results_DAC/",["results_idle_0","results_idle_4"],0))
baseline.append(get_results("./results_DAC/",["results_0_0","results_0_1","results_0_3","results_0_4"],0))
baseline.append(get_results("./results_DAC/",["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"],0))
baseline.append(get_results("./results_DAC/",["results_2_0","results_2_3"],0))

baseline_means=[]
for curr in baseline:
    baseline_means.append(np.mean(curr))


results=[]
results.append(get_results("./results_LR_Waystream/",["results_idle_0","results_idle_1","results_idle_2","results_idle_4"],0))
results.append(get_results("./results_LR_Waystream/",["results_0_0","results_0_1","results_0_3","results_0_4"],0))
results.append(get_results("./results_LR_Waystream/",["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"],0))
results.append(get_results("./results_LR_Waystream/",["results_2_0","results_2_1","results_2_2"],0))

results_means=[]
for curr in results:
    results_means.append(np.mean(curr))

h_hat=[]
for ind,curr in enumerate(baseline_means):
    h_hat.append(curr)
f_hat=[]
for ind,curr in enumerate(results_means):
    f_hat.append(curr)
cr_ee=[(h_hat[x]/f_hat[x])*100 for x in range(len(f_hat))]
print(cr_ee)
plt.bar([1,3,5,7],cr_ee,width=0.8,color=colors[1],label="Waystream SFP")

results=[]
results.append(get_results("./results_LR/",["results_idle_0","results_idle_2","results_idle_4"],0))
results.append(get_results("./results_LR/",["results_0_0","results_0_3"],0))
results.append(get_results("./results_LR/",["results_1_1","results_1_2","results_1_3","results_1_4"],0))
results.append(get_results("./results_LR/",["results_2_0","results_2_2"],0))

results_means=[]
for curr in results:
    results_means.append(np.mean(curr))

h_hat=[]
for ind,curr in enumerate(baseline_means):
    h_hat.append(curr)
f_hat=[]
for ind,curr in enumerate(results_means):
    f_hat.append(curr)
cr_ee=[(h_hat[x]/f_hat[x])*100 for x in range(len(f_hat))]
print(cr_ee)
#plt.plot(configs_numeric,f_hat)
#plt.plot(configs_numeric,h_hat)
plt.bar([0,2,4,6],cr_ee,width=0.8,color=colors[3],label="Regular SFP")







plt.xticks([0.5,2.5,4.5,6.5],configs)
plt.grid(which='major')
plt.legend(loc="lower right",ncol=2,bbox_to_anchor=(1.1,1),fontsize=26)#, handleheight=0.9, labelspacing=0.2, handlelength=0.7,fontsize=13)#,
plt.gcf().subplots_adjust(bottom=0.22)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(top=0.85)
plt.ylabel(r"$CrEE (\eta) [\%]$")
#plt.ylabel("Real Power [W]")
plt.xticks(rotation=25)
ax.grid(which='major', alpha=0.2)
#plt.show()
plt.savefig('ee_comparison_LR.pdf')
plt.close()



results=[]
results.append(get_results("./results_BX_Waystream/",["results_idle_0","results_idle_1","results_idle_2","results_idle_3"],0))
results.append(get_results("./results_BX_Waystream/",["results_0_1","results_0_3"],0))
results.append(get_results("./results_BX_Waystream/",["results_1_1","results_1_2","results_1_3"],0))
results.append(get_results("./results_BX_Waystream/",["results_2_0","results_2_1","results_2_2"],0))

results_means=[]
for curr in results:
    results_means.append(np.mean(curr))

h_hat=[]
for ind,curr in enumerate(baseline_means):
    h_hat.append(curr)
f_hat=[]
for ind,curr in enumerate(results_means):
    f_hat.append(curr)
cr_ee=[(h_hat[x]/f_hat[x])*100 for x in range(len(f_hat))]
print(cr_ee)
plt.bar([1,3,5,7],cr_ee,width=0.8,color=colors[1],label="Waystream SFP")

results=[]
results.append(get_results("./results_BX/",["results_idle_1","results_idle_4"],0))
results.append(get_results("./results_BX/",["results_0_0","results_0_1","results_0_2","results_0_3","results_0_4"],0))
results.append(get_results("./results_BX/",["results_1_0","results_1_1","results_1_2","results_1_3","results_1_4"],0))
results.append(get_results("./results_BX/",["results_2_1","results_2_4"],0))

results_means=[]
for curr in results:
    results_means.append(np.mean(curr))

h_hat=[]
for ind,curr in enumerate(baseline_means):
    h_hat.append(curr)
f_hat=[]
for ind,curr in enumerate(results_means):
    f_hat.append(curr)
cr_ee=[(h_hat[x]/f_hat[x])*100 for x in range(len(f_hat))]
print(cr_ee)
#plt.plot(configs_numeric,f_hat)
#plt.plot(configs_numeric,h_hat)
plt.bar([0,2,4,6],cr_ee,width=0.8,color=colors[3],label="Regular SFP")







plt.xticks([0.5,2.5,4.5,6.5],configs)
plt.grid(which='major')
plt.legend(loc="lower right",ncol=2,bbox_to_anchor=(1.1,1),fontsize=26)#, handleheight=0.9, labelspacing=0.2, handlelength=0.7,fontsize=13)#,
plt.gcf().subplots_adjust(bottom=0.22)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(top=0.85)
plt.ylabel(r"$CrEE (\eta) [\%]$")
#plt.ylabel("Real Power [W]")
plt.xticks(rotation=25)
ax.grid(which='major', alpha=0.2)
#plt.show()
plt.savefig('ee_comparison_BX.pdf')
plt.close()