from utils import seed

from joblib import dump, load
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import random
from teacher.datasets import load_compas, load_german, load_adult, load_heloc, load_iris, load_wine, load_beer


# %%
DATASETS = {
    'adult': load_adult,
    'compas': load_compas,
    'fico': load_heloc,
    'german': load_german,
    'iris': load_iris,
    'wine': load_wine,
    'beer': load_beer
}


BLACK_BOXES = {
    'RF': RandomForestClassifier(random_state=seed),
    'NN': MLPClassifier(random_state=seed),
    'SVM': SVC(random_state=seed),
}

def train_models(ds, bb, random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    path_models = './models/'

    black_box = BLACK_BOXES[bb]

    dataset = DATASETS[ds](normalize=True)
    X, y = dataset['X'], dataset['y']

    # TRAIN 60, DEV 25, TEST 15
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=seed)
    black_box.fit(X_train, y_train)
    score = black_box.score(X_test, y_test)
    
    dump(black_box, path_models + '%s_%s.joblib' % (ds, bb))
    print(f'Accuracy for model {bb} and dataset {ds}: {score}')

if __name__ == "__main__":
    for ds in ['iris', 'wine', 'beer']:
        for bb in ['RF', 'NN', 'SVM']:
            print(f'Training model {bb} for {ds}')
            train_models(ds, bb, seed)