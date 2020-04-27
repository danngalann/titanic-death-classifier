from sklearn.model_selection import train_test_split
import pandas as pd

"""
int PassengerId
int Survived
int Pclass [1-3]
string Name
string Sex (male/female)
float Age
int SibSp
int Parch
String Ticket
float Fare
String Cabin
char Embarked
"""



# Loads data from csv
def loadData():
    # Load the data
    df = pd.read_csv("../data.csv")

    # Drop unused cols
    unused_cols = ["Ticket", "Fare", "Cabin", "Embarked", "PassengerId", "Name"]
    df = df.drop(unused_cols, axis=1)

    # Fill age nulls with the median of the same sex ages
    df["Age"].fillna(df.groupby("Sex")["Age"].transform("median"), inplace=True)

    # Map sex to ints
    sex_mapping = {"male": 0, "female": 1}
    df["Sex"] = df["Sex"].map(sex_mapping)

    # Convert everything to ints
    df = df.astype("int")

    # Split data into features and labels
    labels = pd.DataFrame(df["Survived"])
    features = df.drop("Survived", axis=1) # Pclass, Sex, Age, SibSp, Parch

    # Return features and labels as a list
    return features.values.tolist(), labels.values.tolist()
    

# Returns all the features and labels, without splits
def getAll():
    features, labels = loadData()
    return features, labels
