import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import sys
import os.path
import operator
from sklearn import utils

#Purpose: this function turns each (0,1,-1) feature into two binary features:
# is1 and is-1.
#Parameter: data, features, dataset name
#Return: data, features
def newBinarize(data, features, name):

    newData = np.zeros((len(data),1), dtype=float)
    #initialize one col so it's easier to use hstack
    newFeatures = []

    for i in range(0, len(features)):
        newFeatures.append(features[i]+"=1")
        newFeatures.append(features[i]+"=-1")
        temp = np.zeros((len(data),2), dtype=float)

        for index, item in enumerate(data[:, i]):
            if item == 1:
                temp[index, 0] = 1
            elif item == -1:
                temp[index, 1] = 1

        newData = np.hstack((newData, temp))

    newData2 = newData[:,1:] #get rid of the initial column

    #storing as sparse matrix
    sdata = sparse.csr_matrix(newData2)
    sparse.save_npz(name+"Binarize.npz", sdata)

    return newData2, newFeatures

#Purpose: add PolynomialFeatures
def PolynomialFeatures(data,features,name):
    #deal with the first column, dummy value of all 1s
    features.insert(0, "dummy")

    #generate feature names and values for combos
    for i in range(1, len(features)+1):
        for j in range(i+1, len(features)+1):
            features.append(features[i]+features[j])

    poly = preprocessing.PolynomialFeatures(2, interaction_only=True)
    data = poly.fit_transform(data)

    #storing as sparse matrix
    sdata = sparse.csr_matrix(data)
    sparse.save_npz(name+"added.npz", sdata)

    np.save(name+"AddedLabel.npy", features)
    return data, features

# calculate the synergy given hero a, b
def comboRate(a, b, data, label):
    win = 0
    total = 0

    for i,item in enumerate(label):
        if data[i, a] == 1 and data[i, b] == 1:
            total += 1
            if item == 1:
                win += 1
        if data[i, a] == -1 and data[i, b] == -1:
            total += 1
            if item == -1:
                win += 1

    if total == 0:
        return 0
    else:
        return float(win)/total

# create the synergy chart
def combos(data, label):
    a = np.zeros((113, 113), dtype=float)

    count = 0
    for i in range(0, 113):
        for j in range(i+1, 113):
            a[i, j] = comboRate(i, j, data, label)
    return a

# a debugging function to see the pair with highest synergy, and use our domain
# knowledge to make sure they make sense
def comboRank(chart):
    # create a tuple list (rate, hero1, hero2)
    rankTuple = []

    for i in range(0, 113):
        for j in range(i+1, 113):
            rankTuple.append((chart[i,j], i, j))

    rankTuple.sort(key = operator.itemgetter(0), reverse=True)

    f = open("comboRank", "w")
    for item in rankTuple:
        f.write(str(item[0])+": "+str(item[1])+","+str(item[2])+"\n")
    f.close()

    return rankTuple

# draw a heatmap of synergy
def heatmap():
    a = sparse.load_npz("comboChart.npz").todense()
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()
    plt.savefig("comboHM.png")

# add a synergy column to the data
def comboCol(data, comboChart):
    combo = np.zeros((1,1), dtype=float) #initialize so it's easier to stack
    # comboChart = sparse.load_npz("comboChart.npz").todense()

    for index, item in enumerate(data):
        team1 = np.where(item == 1)[1]
        team2 = np.where(item == -1)[1]

        sr = 0
        sd = 0

        for i in range(0, 5):
            for j in range(i+1, 5):
                sr += comboChart[team1[i], team1[j]]
                sd += comboChart[team2[i], team2[j]]

        combo = np.vstack((combo, sr-sd))

    combo = np.delete(combo, (0), axis=0) #delete first row
    return combo

# calculate winning rate given counter a and b
def counterRate(a, b, data, label):
    win = 0
    total = 0

    for i,item in enumerate(label):
        if data[i, a] == 1 and data[i, b] == -1:
            total += 1
            if item == 1:
                win += 1
        if data[i, a] == -1 and data[i, b] == 1:
            total += 1
            if item == -1:
                win += 1

    if total == 0:
        return 0
    else:
        return float(win)/total

# calculate counter chart
def counters(data, label):
    a = np.zeros((113, 113), dtype=float)

    for i in range(0, 113):
        for j in range(i+1, 113):
            a[i, j] = counterRate(i, j, data, label)
    return a

# a debugging function to see the pair with highest counter, and use our domain
# knowledge to make sure they make sense
def counterRank(chart):
    # create a tuple list (rate, hero1, hero2)
    rankTuple = []

    for i in range(0, 113):
        for j in range(i+1, 113):
            rankTuple.append((chart[i,j], i, j))

    rankTuple.sort(key = operator.itemgetter(0), reverse=True)

    f = open("counterRank", "w")
    for item in rankTuple:
        f.write(str(item[0])+": "+str(item[1])+","+str(item[2])+"\n")
    f.close()

    return rankTuple

# add a synergy column to the data
def counterCol(data, counterChart):
    counter = np.zeros((1,1), dtype=float) #initialize so it's easier to stack
    # counterChart = sparse.load_npz("counterChart.npz").todense()

    for index, item in enumerate(data):
        team1 = np.where(item == 1)[1]
        team2 = np.where(item == -1)[1]

        result = 0

        for i in range(0, 5):
            for j in range(0, 5):
                result += counterChart[team1[i], team1[j]]

        counter = np.vstack((counter, result))

    counter = np.delete(counter, (0), axis=0) #delete first row
    return counter

# add synergy and counter columns for training set
def addSynergyCounterTrain(data, label):
    comboChart = combos(data, label)
    counterChart = counters(data, label)
    combo = comboCol(data, comboChart)
    counter = counterCol(data, counterChart)

    return np.hstack((data, combo, counter)), comboChart, counterChart

# add synergy and counter columns for test set (need to pass in synergy chart
# and counter chart)
def addSynergyCounterTest(data, comboChart, counterChart):
    combo = comboCol(data, comboChart)
    counter = counterCol(data, counterChart)

    return np.hstack((data, combo, counter))

# save as sparse matrix in a separate file
def saveSparse(name, array):
    sdata = sparse.csr_matrix(array)
    sparse.save_npz(name, sdata)
