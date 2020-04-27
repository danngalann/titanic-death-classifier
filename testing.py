from random import random
from train_model import predict, getClassifier
from matplotlib import pyplot as plt

# lastpred = 0
# for i in range(1000):
#     age = round((random() * 79) +1)
#     sex = round(random())
#     pClass = round((random() * 2) +1)
#     print(predict([[pClass, sex, age]]), [pClass, sex, age])

# while lastpred != 1:
#     age = round((random() * 79) +1)
#     sex = round(random())
#     pClass = round((random() * 2) +1)
#     pred = predict([[pClass, sex, age]])
#     lastpred = pred
#     print(pred, [pClass, sex, age])


def f_importances(coef, names):
    imp = coef
    imp, names = zip(*sorted(zip(imp, names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()


clf = getClassifier()
# f_importances(clf.coef_.ravel(), ['pClass', 'age', 'sex'])
print(clf.coef_)