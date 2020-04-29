# Titanic Death classifier with SkLearn and Pandas
This classifier will predict if you'd die on the Titanic, based on a number of features. It was able to achieve around 80% accuracy on an SVC using `Pclass, Sex, Age, SibSp, Parch` as features with 20% of the data on the testing set. 

A better accuracy could be achieved by tweaking the classifier hyperparameters on `train_classifier.py` or changing the features on `tools/preprocess.py`. You can find an overview of the dataset [here](http://campus.lakeforest.edu/frank/FILES/MLFfiles/Bio150/Titanic/TitanicMETA.pdf).

To run this scrips you will need `sklearn`, `pandas`, `numpy` and `pickle`.

A Django implementation of this classifier can be found [here](https://github.com/danngalann/django-titanic-survial).
