import csv
header = ["class", "clusterID", "mode", "type"]
for i in range(0, 113):
    col = "id" + str(i+1)
    header.append(col)

with open('dota2Test.csv') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('dota2Test_withHeader.csv','w') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(data)
f.close()

with open('dota2Train.csv') as f:
    r = csv.reader(f)
    data = [line for line in r]
with open('dota2Train_withHeader.csv','w') as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(data)
f.close()
