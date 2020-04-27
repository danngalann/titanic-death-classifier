import csv, math, sys, random
from sklearn.model_selection import train_test_split

data = {
    "survived": [],
    "pClass": [],
    "sex": [],
    "age": []
}
features, labels = [], []

# Loads data from csv
def loadData():
    with open('data.csv', 'r') as csvFile:
        plots = list(csv.reader(csvFile, delimiter=","))
        # next(plots, None)  # skip the headers
        
        for row in plots[1:100]:            
            # Is the row missing data?
            incompleteRow = False
            for i in range(1,7):
                if row[i] == '':
                    incompleteRow = True

            # Skip rows with missing data
            if not incompleteRow:
                data["survived"].append(int(row[1]))
                data["pClass"].append(int(row[2]))
                data["sex"].append(1 if row[4] == 'male' else 0) # Male = 1, Female = 0
                data["age"].append(math.floor(float(row[5]))) # Estimated ages come in the format of x.5; floor them.

# Imports the data to the features and labels variables
def importData():
    for i in range(len(data['pClass'])):
        features.append([data['pClass'][i], data['age'][i], data['sex'][i]])
        labels.append(data['survived'][i])

# Returns training and testing splits of features and labels
def preprocess():
    loadData()
    importData()
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.20)
    return features_train, features_test, labels_train, labels_test

# Returns all the features and labels, without splits
def getAll():
    loadData()
    importData()
    return features, labels
