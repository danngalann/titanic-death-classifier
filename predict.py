import train_model
import os, sys
from tools import classifier_saver as saver

# Gets classifier from file or trains a new one
def getClassifier():
  clf = saver.load() if not os.path.exists("model.pkl") else train_model.train()
  return clf

def predict(values):
  clf = getClassifier()
  return clf.predict(values)[0]