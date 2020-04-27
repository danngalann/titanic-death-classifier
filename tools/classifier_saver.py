import pickle

def save(classifier):
    with open('model.pkl', 'wb') as fid:
        pickle.dump(classifier, fid)

def load():
    classifier = None
    with open('model.pkl', 'rb') as fid:
        classifier = pickle.load(fid)
    
    return classifier