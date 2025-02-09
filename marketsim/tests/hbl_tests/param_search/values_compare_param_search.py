import numpy as np
import matplotlib.pyplot as plt
import os

valsPath = ["2e2", "6e3"]
# valsPath = ["2e2", "6e3"]
pathToTake = [{"A1":"s", "A2":"s", "A3":"s", "B1":"s", "B2":"s", "B3":"s", "C1":"s", "C2":"s", "C3":"s"},
{"A1":"n", "A2":"s", "A3":"s", "B1":"s", "B2":"s", "B3":"n", "C1":"s", "C2":"s", "C3":"n"}
]
for indPath, valPath in enumerate(valsPath):
    paths = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    pathNames = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]

    vals = []
    stdVals = []
    staticVals = []
    stdStaticVals = []
    
    for path in paths:
        with open(os.path.join(os.getcwd(), 'param_search/{}_optimal'.format(valPath), '{}/graphs'.format(path), 'valuesCompare.txt'), 'r') as file:
            tempVals = []
            tempStd = []
            data = file.read()
            data = data.replace('\n', " ").replace("[", " ").replace("]", " ").replace(",", "")
            dataStatic = data.split()[10:]
            data = data.split()[1:9]
            for ind, val in enumerate(data):
                if ind <= 3:
                    tempVals.append(float(val))
                else:
                    tempStd.append(float(val))
            vals.append(tempVals)
            stdVals.append(tempStd)
            
            tempVals = []
            tempStd = []
            for ind, val in enumerate(dataStatic):
                if ind <= 3:
                    tempVals.append(float(val))
                else:
                    tempStd.append(float(val))
            staticVals.append(tempVals)
            stdStaticVals.append(tempStd)

    markers = {'Spoofer': 'o', 'ZI': '^', 'HBL': 'x', 'MM':'d'}
    colors = {"Spoofer": 'blue', 'ZI':"green", 'HBL': 'orange', 'MM':'magenta'}
    print(vals, stdVals)
    print(staticVals, stdStaticVals)
    plt.figure(constrained_layout=True)
    for ind,i in enumerate(staticVals):
        if pathToTake[indPath][pathNames[ind]] == "s":
            plot = i
            std = stdStaticVals
        else:
            plot = vals[ind]
            std = stdVals
        plt.scatter(ind, plot[0], marker=markers["ZI"], color=colors["ZI"], label="ZI", s=60)
        plt.errorbar(ind, plot[0], yerr=std[ind][0], capsize=3, fmt='none', color=colors["ZI"], alpha=0.5, label='_nolegend_')
        plt.scatter(ind, plot[1], marker=markers["HBL"], color=colors["HBL"], label="HBL", s=60)
        plt.errorbar(ind, plot[1], yerr=std[ind][1], capsize=3, fmt='none', color=colors["HBL"], alpha=0.5, label='_nolegend_')
        # plt.scatter(ind, i[2]), marker=markers["MM"], color=colors["Spoofer"], markersize=10, label="4e-2 Spoofer")
        # plt.errorbar(ind, i[2], yerr=stdVals[ind][2], capsize=3, fmt=markers["MM"], color=colors["MM"], label="MM")
        # plt.scatter(ind, i[-1]), marker=markers["Spoofer"], color=colors["Spoofer"], markersize=10, label="4e-2 Spoofer")
        plt.scatter(ind, plot[3], marker=markers["Spoofer"], color=colors["Spoofer"], label="Spoofer")
        plt.errorbar(ind, plot[3], yerr=std[ind][3], capsize=3, fmt='none', color=colors['Spoofer'], alpha=0.5, label='_nolegend_')

    plt.xticks(ticks=np.linspace(0,len(staticVals), 9, endpoint=False), labels=pathNames)
    plt.xlabel("Market Configuration")
    plt.ylabel("Surplus")
    handles, labels = plt.gca().get_legend_handles_labels()
    # input(labels)
    # newHandles = [handles[:3],handles[-3:]]
    # newLabels = [labels[:3], labels[:-3:]]
    # input(handles[-3:])
    newHandles = handles[:3]
    # newHandles.extend(handles[-3:])
    newLabels = labels[:3]
    # newLabels.extend(labels[-3:])
    plt.legend(newHandles, newLabels)
    # print(os.getcwd())
    plt.savefig(os.getcwd() + "/new_data_compil/param_search/{}_spoofer".format(valPath) + "/surplus_chart.png")




