import csv
import numpy as np
import os
import glob
import pandas as pd
import json
from operator import itemgetter
from collections import OrderedDict
from scipy import sparse
from sklearn.cross_validation import train_test_split

def createRow(heroes, heroPicks1, mark1, heroPicks2, mark2):
    temp = np.zeros((1,115), dtype=int)
    harray1 = heroPicks1.split()
    harray2 = heroPicks2.split()
    counter = 0
    num = 0

    while num < 5:
        fourWord = ""
        threeWord = ""
        twoWord = ""
        oneWord = ""

        if counter < len(harray1)-3:
            fourWord = harray1[counter]+" "+harray1[counter+1]+" "+harray1[counter+2]+" "+harray1[counter+3]
        if counter < len(harray1)-2:
            threeWord = harray1[counter]+" "+harray1[counter+1]+" "+harray1[counter+2]
        if counter < len(harray1)-1:
            twoWord = harray1[counter]+" "+harray1[counter+1]
        oneWord = harray1[counter]


        if fourWord in heroes:
            temp[0][heroes[fourWord]] = mark1
            counter += 4
        elif threeWord in heroes:
            temp[0][heroes[threeWord]] = mark1
            counter += 3
        elif twoWord in heroes:
            temp[0][heroes[twoWord]] = mark1
            counter += 2
        else:
            temp[0][heroes[oneWord]] = mark1
            counter += 1
        num += 1

    counter = 0
    num = 0
    while num < 5:
        fourWord = ""
        threeWord = ""
        twoWord = ""
        oneWord = ""

        if counter < len(harray2)-3:
            fourWord = harray2[counter]+" "+harray2[counter+1]+" "+harray2[counter+2]+" "+harray2[counter+3]
        if counter < len(harray2)-2:
            threeWord = harray2[counter]+" "+harray2[counter+1]+" "+harray2[counter+2]
        if counter < len(harray2)-1:
            twoWord = harray2[counter]+" "+harray2[counter+1]
        oneWord = harray2[counter]

        if fourWord in heroes:
            temp[0][heroes[fourWord]] = mark2
            counter += 4
        elif threeWord in heroes:
            temp[0][heroes[threeWord]] = mark2
            counter += 3
        elif twoWord in heroes:
            temp[0][heroes[twoWord]] = mark2
            counter += 2
        else:
            temp[0][heroes[oneWord]] = mark2
            counter += 1
        num += 1

    return temp

def main():
    path = "/home/yren2/cs66/labs/FinalProject-yren2-hhuang2/code/Dota2"
    allFiles = glob.glob(os.path.join(path,"*.csv"))

    np_array_list = []
    for file_ in allFiles:
        df = pd.read_csv(file_,index_col=None, header=0)
        np_array_list.append(df.as_matrix())

    comb_np_array = np.vstack(np_array_list)
    big_frame = pd.DataFrame(comb_np_array)
    big_frame.columns = ["Match ID", "League", "Start Date", "Duration (s)", "Duration (mm:ss)", \
        "Radiant Team", "Team A", "Team A Heroes",	"Team B", "Team B Heroes", "Winner"]

    rawData = big_frame.drop_duplicates(["Match ID"], keep='last')

    # get data from only 2014
    rawData = rawData[rawData["Start Date"].str.contains("2013")]


    with open('heroes.json') as json_data:
        d = json.load(json_data)

    heroes = {}
    for i in range(len(d["heroes"])):
        name = d["heroes"][i]["localized_name"]
        hid = d["heroes"][i]["id"]
        heroes[name] = hid-1

    sorted_heores = OrderedDict(sorted(heroes.items(), key=lambda t: t[1]))

    data = np.zeros((1,115), dtype=float) #initialize so it's easier to stack
    label = np.zeros((1,1), dtype=float) #initialize so it's easier to stack

    for index, row in rawData.iterrows():
        if row["Radiant Team"] == "Team A":
            team1 = 1
            team2 = -1
            if row["Winner"] == "Team A":
                label = np.vstack((label, [1]))
            else:
                label = np.vstack((label, [-1]))
        else:
            team1 = -1
            team2 = 1
            if row["Winner"] == "Team A":
                label = np.vstack((label, [-1]))
            else:
                label = np.vstack((label, [1]))

        hero1 = row["Team A Heroes"]
        hero2 = row["Team B Heroes"]

        data = np.vstack((data, createRow(heroes, hero1, team1, hero2, team2)))

    data = np.delete(data, (0), axis=0) #delete first row of 0
    label = np.delete(label, (0), axis=0) #delete first row of 0
    # sdata = sparse.csr_matrix(data.astype(float))
    # sparse.save_npz("data.npz", sdata)

    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, random_state = 42)
    sdata = sparse.csr_matrix(X_train.astype(float))
    sparse.save_npz("trainData2014.npz", sdata)
    sdata = sparse.csr_matrix(X_test.astype(float))
    sparse.save_npz("testData2014.npz", sdata)

    np.save("trainLabel2014.npy", y_train)
    np.save("testLabel2014.npy", y_test)

main()
