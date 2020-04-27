import time
from tools import classifier_saver as saver
from tools.preprocess import loadData
from sklearn import svm

features_train, features_test, labels_train, labels_test = loadData("data.csv")

def train():
  print("Training...")
  clf = svm.SVC(gamma="auto")
  start = time.time()
  clf.fit(features_train, labels_train)
  end = time.time()
  print("Training time: ", end-start)
  acc = clf.score(features_test, labels_test)
  print("Accuracy: " + str(acc))
  saver.save(clf)

  return clf