import json

import matplotlib.pyplot as plt
import scipy.optimize
import numpy as np


plt.rcParams.update({'font.size': 26})
plt.rcParams['figure.figsize'] = [10, 5]
fig, ax = plt.subplots()
colors = plt.cm.copper(np.linspace(0,1,4))
base_path = "./1748424905/"
runs=["idle","500","1000"]
offsets=[-700,-2200,-70]

def get_timeseries_data(data_row,currfile,ind):
    with open(str(base_path) +"results_"+ str(currfile) + ".json") as jsonfile:
        data = dict(json.load(jsonfile))
    xaxis = []
    yaxis = []
    for curr in range(len(data["Time"])):
        try:
            if(data["Time"][curr]-data["Start_Time"][0]+offsets[ind] > 0 and data["Time"][curr]-data["Start_Time"][0]+offsets[ind] < 1300):
                yaxis.append(data[data_row][curr])
                xaxis.append(data["Time"][curr]-data["Start_Time"][0]+offsets[ind])
        except:
            pass
    return xaxis,yaxis

def get_timeseries_data_dev_metrics(data_row,data_row_2,currfile,ind):
    with open(str(base_path) +"results_"+ str(currfile) + "_dev_metrics.json") as jsonfile:
        data = dict(json.load(jsonfile))
    xaxis = []
    yaxis = []
    for curr in range(len(data["Time"])):
        try:
            if(data["Time"][curr]-data["Start_Time"][0]+offsets[ind] > 0 and data["Time"][curr]-data["Start_Time"][0]+offsets[ind] < 1400):
                yaxis.append(float(data[data_row][curr][data_row_2]))
                xaxis.append(data["Time"][curr]-data["Start_Time"][0]+offsets[ind])
        except:
            pass
    return xaxis,yaxis

def plot_timeseries(data_row,ylabel):
    for ind,currfile in enumerate(runs):
        with open(str(base_path) +"results_"+ str(currfile) + ".json") as jsonfile:
            data = dict(json.load(jsonfile))
        xaxis = []
        yaxis = []
        for curr in range(len(data["Time"])):
            try:
                if(data["Time"][curr]-data["Start_Time"][0]+offsets[ind] > 0):
                    yaxis.append(data[data_row][curr])
                    xaxis.append(data["Time"][curr]-data["Start_Time"][0]+offsets[ind])
            except:
                pass
        plt.plot(xaxis,yaxis,label=currfile,color=colors[ind],lw=3)
        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)

def plot_timeseries_dev_metrics(data_row,data_row_2,ylabel):
    for ind,currfile in enumerate(runs):
        with open(str(base_path) +"results_"+ str(currfile) + "_dev_metrics.json") as jsonfile:
            data = dict(json.load(jsonfile))
        xaxis = []
        yaxis = []
        for curr in range(len(data["Time"])):
            try:
                if(data["Time"][curr]-data["Start_Time"][0]+offsets[ind] > 0):
                    yaxis.append(float(data[data_row][curr][data_row_2]))
                    xaxis.append(data["Time"][curr]-data["Start_Time"][0]+offsets[ind])
            except:
                pass
        plt.plot(xaxis,yaxis,label=currfile,color=colors[ind],lw=3)
        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)

def plot_timeseries_raw(data_row,ylabel):
    for ind,currfile in enumerate(runs):
        with open(str(base_path) +"results_"+ str(currfile) + ".json") as jsonfile:
            data = dict(json.load(jsonfile))
        xaxis = []
        yaxis = []
        for curr in range(len(data["Time"])):
            try:
                yaxis.append(data[data_row][curr])
                xaxis.append(data["Time"][curr]-data["Start_Time"][0])
            except:
                pass
        plt.plot(xaxis,yaxis,label=currfile,color=colors[ind],lw=3)
        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)

def plot_timeseries_dev_metrics_raw(data_row,data_row_2,ylabel):
    for ind,currfile in enumerate(runs):
        with open(str(base_path) +"results_"+ str(currfile) + "_dev_metrics.json") as jsonfile:
            data = dict(json.load(jsonfile))
        xaxis = []
        yaxis = []
        for curr in range(len(data["Time"])):
            try:
                yaxis.append(float(data[data_row][curr][data_row_2]))
                xaxis.append(data["Time"][curr]-data["Start_Time"][0])
            except:
                pass
        plt.plot(xaxis,yaxis,label=currfile,color=colors[ind],lw=3)
        plt.xlabel('Time [s]')
        plt.ylabel(ylabel)



plot_timeseries("Real_Power","Real Power [W]")



plt.grid(which='major')
plt.legend()
plt.gcf().subplots_adjust(bottom=0.19)
plt.gcf().subplots_adjust(left=0.18)
plt.xlim(0,1400)
ax.grid(which='major', alpha=0.2)
plt.savefig('Power_curve.pdf')
plt.close()


plot_timeseries("Temp_env","Env. Temperature [°C]")



plt.grid(which='major')
plt.legend()
plt.gcf().subplots_adjust(bottom=0.19)
plt.gcf().subplots_adjust(left=0.18)
plt.xlim(0,1400)
ax.grid(which='major', alpha=0.2)
plt.savefig('Temp_curve.pdf')
plt.close()


plot_timeseries_dev_metrics("Temp","cpu","CPU Temperature [°C]")



plt.grid(which='major')
plt.legend()
plt.gcf().subplots_adjust(bottom=0.19)
plt.gcf().subplots_adjust(left=0.18)
plt.xlim(0,1400)
ax.grid(which='major', alpha=0.2)
plt.savefig('CPU_Temp_curve.pdf')
plt.close()

plot_timeseries_dev_metrics("RPM","1","Fan Speed [RPM]")



plt.grid(which='major')
plt.legend()
plt.gcf().subplots_adjust(bottom=0.19)
plt.gcf().subplots_adjust(left=0.18)
plt.xlim(0,1400)
ax.grid(which='major', alpha=0.2)
plt.savefig('RPM_curve.pdf')
plt.close()

plt.rcParams.update({'font.size': 26})
plt.rcParams['figure.figsize'] = [10, 5]
fig, ax = plt.subplots()

x_1,y_1 = get_timeseries_data_dev_metrics("Temp","cpu",runs[0],0)
x_2,y_2 = get_timeseries_data_dev_metrics("RPM","1",runs[0],0)

y2_match=[]
for ind,currx in enumerate(x_1):
    best_y = 0
    best_y_dist = float("inf")
    for ind2,currx2 in enumerate(x_2):
        if(abs(currx-currx2) < best_y_dist):
            best_y = float(y_2[ind2])
            best_y_dist = abs(currx-currx2)
    y2_match.append(best_y)

combined=[[y_1[x],y2_match[x]] for x in range(len(x_1))]
combined.sort()
y_1=[x[0] for x in combined]
y2_match = [x[1] for x in combined]
plt.scatter(y_1,y2_match,color=colors[0],lw=1)

target_ind =0
for ind,curr in enumerate(y_1):
    if curr <= 50:
        target_ind = ind+1

m,b = np.polyfit(y_1[0:target_ind],y2_match[0:target_ind],1)
print(m,b)
plt.plot(y_1[0:target_ind],[m*x+b for x in y_1[0:target_ind]],color=colors[1],lw=3)
m,b = np.polyfit(y_1[target_ind-1:-1],y2_match[target_ind-1:-1],1)
plt.plot(y_1[target_ind-1:-1],[m*x+b for x in y_1[target_ind-1:-1]],color=colors[2],lw=3)
print(m,b)

plt.xlabel('CPU temperature [°C]')
plt.ylabel('fan speed [RPM]')
plt.grid(which='major')
plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.18)
plt.gcf().subplots_adjust(top=0.75)
ax.grid(which='major', alpha=0.2)
plt.savefig('Fan_curve.pdf')
plt.close()

x_1,y_1 = get_timeseries_data_dev_metrics("Temp","cpu",runs[2],2)
x_2,y_2 = get_timeseries_data_dev_metrics("RPM","1",runs[2],2)

y2_match=[]
for ind,currx in enumerate(x_1):
    best_y = 0
    best_y_dist = float("inf")
    for ind2,currx2 in enumerate(x_2):
        if(abs(currx-currx2) < best_y_dist):
            best_y = float(y_2[ind2])
            best_y_dist = abs(currx-currx2)
    y2_match.append(best_y)

combined=[[y_1[x],y2_match[x]] for x in range(len(x_1))]
combined.sort()
y_1=[x[0] for x in combined]
y2_match = [x[1] for x in combined]
plt.plot(y_1,y2_match,color=colors[0],lw=3)

target_ind =0
for ind,curr in enumerate(y_1):
    if curr <= 50:
        target_ind = ind+1

m,b = np.polyfit(y_1[0:target_ind],y2_match[0:target_ind],1)
print(m,b)
plt.plot(y_1[0:target_ind],[m*x+b for x in y_1[0:target_ind]])
m,b = np.polyfit(y_1[target_ind-1:-1],y2_match[target_ind-1:-1],1)
plt.plot(y_1[target_ind-1:-1],[m*x+b for x in y_1[target_ind-1:-1]])
print(m,b)

plt.xlabel('CPU temperature [°C]')
plt.ylabel('fan speed [RPM]')
plt.grid(which='major')
plt.gcf().subplots_adjust(bottom=0.19)
plt.gcf().subplots_adjust(left=0.18)
ax.grid(which='major', alpha=0.2)
plt.savefig('Fan_curve_full_load.pdf')
plt.close()


















plt.rcParams.update({'font.size': 26})
plt.rcParams['figure.figsize'] = [10, 5]
fig, ax = plt.subplots()


x_1,y_1 = get_timeseries_data_dev_metrics("RPM","1",runs[0],0)
x_2,y_2 = get_timeseries_data("Real_Power",runs[0],0)



y2_match=[]
for ind,currx in enumerate(x_1):
    best_y = 0
    best_y_dist = float("inf")
    for ind2,currx2 in enumerate(x_2):
        if(abs(currx-currx2) < best_y_dist):
            best_y = float(y_2[ind2])
            best_y_dist = abs(currx-currx2)
    y2_match.append(best_y)
#plt.scatter(y_1,y2_match,color=colors[0],lw=3)
p = np.polyfit(y_1, np.log(y2_match), 1,w=np.sqrt(y2_match))
a = np.exp(p[1])
b = p[0]
x_fitted = np.linspace(np.min(y_1), np.max(y_1), 100)
y_fitted = a * np.exp(b * x_fitted)

def monoExp(x, a, b,c,t):
    return (a *x)**3+(b*x)**2+c*x+t
p0 = (a, b,20,0) # start with values near those we expect
params, cv = scipy.optimize.curve_fit(monoExp, y_1, y2_match, p0)
a, b,c,t = params
print("RPM Curve")
print(a,b,c,t)

# determine quality of the fit
squaredDiffs = np.square(y2_match - monoExp(np.array(y_1), a, b,c,t))
squaredDiffsFromMean = np.square(y2_match - np.mean(y2_match))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"R² = {rSquared}")
x_fitted = np.linspace(np.min(y_1), np.max(y_1), 100)
y_fitted = (a *x_fitted)**3+(b*x_fitted)**2+c*x_fitted+t
y_fitted = (a *x_fitted)**3+(b*x_fitted)**2+c*x_fitted
plt.plot(x_fitted, y_fitted, color="red", label=f"R² = %.4f" % round(rSquared,4))












x_1,y_1 = get_timeseries_data_dev_metrics("RPM","1",runs[0],0)
x_2,y_2 = get_timeseries_data("Real_Power",runs[0],0)

y2_match=[]
for ind,currx in enumerate(x_1):
    best_y = 0
    best_y_dist = float("inf")
    for ind2,currx2 in enumerate(x_2):
        if(abs(currx-currx2) < best_y_dist):
            best_y = float(y_2[ind2])
            best_y_dist = abs(currx-currx2)
    #y2_match.append(best_y)
    y2_match.append(best_y-t)
print("Offset P_env: " + str(np.min(y2_match[0])))
print(np.max(y2_match[-1]))
plt.scatter(y_1,y2_match,color=colors[0],lw=3,label="idle")
#p = np.polyfit(y_1, np.log(y2_match), 1,w=np.sqrt(y2_match))
#a = np.exp(p[1])
#b = p[0]
#x_fitted = np.linspace(np.min(y_1), np.max(y_1), 100)
#y_fitted = a * np.exp(b * x_fitted)


#p0 = (a, b,20) # start with values near those we expect
#params, cv = scipy.optimize.curve_fit(monoExp, y_1, y2_match, p0)
#a, b,t = params
#print("RPM Curve")
#print(a,b,t)

# determine quality of the fit
squaredDiffs = np.square(y2_match - monoExp(np.array(y_1), a, b,c,t))
squaredDiffsFromMean = np.square(y2_match - np.mean(y2_match))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"R² = {rSquared}")
x_fitted = np.linspace(np.min(y_1), np.max(y_1), 100)
y_fitted = a * (b * x_fitted)+t
#plt.plot(x_fitted, y_fitted, color="red", label=f"R² = %.4f" % round(rSquared,4))

x_1,y_1 = get_timeseries_data_dev_metrics("RPM","1",runs[1],1)
x_2,y_2 = get_timeseries_data("Real_Power",runs[1],1)

y2_match=[]
for ind,currx in enumerate(x_1):
    best_y = 0
    best_y_dist = float("inf")
    for ind2,currx2 in enumerate(x_2):
        if(abs(currx-currx2) < best_y_dist):
            best_y = float(y_2[ind2])
            best_y_dist = abs(currx-currx2)
    #y2_match.append(best_y)
    y2_match.append(best_y - t)
plt.scatter(y_1,y2_match,color=colors[1],lw=3,label="500MBit/s")
#p = np.polyfit(y_1, np.log(y2_match), 1,w=np.sqrt(y2_match))
#a = np.exp(p[1])
#b = p[0]
#x_fitted = np.linspace(np.min(y_1), np.max(y_1), 100)
#y_fitted = a * np.exp(b * x_fitted)


#p0 = (a, b,20) # start with values near those we expect
#params, cv = scipy.optimize.curve_fit(monoExp, y_1, y2_match, p0)
#a, b,t = params
#print("RPM Curve")
#print(a,b,t)

# determine quality of the fit
squaredDiffs = np.square(y2_match - monoExp(np.array(y_1), a, b,c,t))
squaredDiffsFromMean = np.square(y2_match - np.mean(y2_match))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"R² = {rSquared}")
x_fitted = np.linspace(np.min(y_1), np.max(y_1), 100)
y_fitted = a * (b * x_fitted)+t
#plt.plot(x_fitted, y_fitted, color="red", label=f"R² = %.4f" % round(rSquared,4))

x_1,y_1 = get_timeseries_data_dev_metrics("RPM","1",runs[2],2)
x_2,y_2 = get_timeseries_data("Real_Power",runs[2],2)

y2_match=[]
for ind,currx in enumerate(x_1):
    best_y = 0
    best_y_dist = float("inf")
    for ind2,currx2 in enumerate(x_2):
        if(abs(currx-currx2) < best_y_dist):
            best_y = float(y_2[ind2])
            best_y_dist = abs(currx-currx2)
    #y2_match.append(best_y)
    y2_match.append(best_y - t)
plt.scatter(y_1,y2_match,color=colors[2],lw=3,label="1000MBit/s")
#p = np.polyfit(y_1, np.log(y2_match), 1,w=np.sqrt(y2_match))
#a = np.exp(p[1])
#b = p[0]
#x_fitted = np.linspace(np.min(y_1), np.max(y_1), 100)
#y_fitted = a * np.exp(b * x_fitted)


#p0 = (a, b,20) # start with values near those we expect
#params, cv = scipy.optimize.curve_fit(monoExp, y_1, y2_match, p0)
#a, b,t = params
#print("RPM Curve")
#print(a,b,t)

# determine quality of the fit
squaredDiffs = np.square(y2_match - monoExp(np.array(y_1), a, b,c,t))
squaredDiffsFromMean = np.square(y2_match - np.mean(y2_match))
rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
print(f"R² = {rSquared}")
x_fitted = np.linspace(np.min(y_1), np.max(y_1), 100)
y_fitted = a * (b * x_fitted)+t
#plt.plot(x_fitted, y_fitted, color="red", label=f"R² = %.4f" % round(rSquared,4))






plt.ylabel('real power [W]')
plt.xlabel('fan speed [RPM]')
plt.grid(which='major')
plt.legend(loc="upper left", ncol=1)
plt.gcf().subplots_adjust(bottom=0.17)
plt.gcf().subplots_adjust(left=0.15)
#plt.gcf().subplots_adjust(top=0.8)
plt.gcf().subplots_adjust(top=0.95)
plt.gcf().subplots_adjust(right=0.95)
ax.grid(which='major', alpha=0.2)
plt.savefig('Power_Consumption_RPM.pdf')
plt.close()

plot_timeseries_raw("Temp_env","Temp")
#plt.show()
