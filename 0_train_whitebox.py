from utils import seed

import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from joblib import dump, load


import random
from teacher.datasets import load_compas, load_german, load_adult, load_heloc, load_iris, load_wine
from teacher.fuzzy import get_fuzzy_variables, get_fuzzy_points
from teacher.tree import FDT


# %%
DATASETS = {
    'adult': load_adult,
    'compas': load_compas,
    'fico': load_heloc,
    'german': load_german,
    'iris': load_iris,
    'wine': load_wine
}


def train_models(ds, bb, random_state):
    random.seed(random_state)
    np.random.seed(random_state)
    path_models = './models/'

    dataset = DATASETS[ds](normalize=True)
    class_name = dataset['class_name']
    discrete = dataset['discrete']
    continuous = dataset['continuous']
    df = dataset['df']
    X = df.drop(class_name, axis=1)
    y = df[class_name]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=random_state)

    X_num = X_train[continuous]
    print(f'Extracting fuzzy points...')
    fuzzy_points = get_fuzzy_points('entropy', continuous, X_num, y_train)
    cate = [col for col in discrete if col != class_name]

    discrete_fuzzy_values = {col: X_train[col].unique() for col in cate}
    fuzzy_variables_order = {col: i for i, col in enumerate(X_train.columns)}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)
    print(f'Training FDT...')
    fdt = FDT(fuzzy_variables)
    fdt.fit(X_train, y_train)
    
    dump(fdt, path_models + '%s_%s.joblib' % (ds, bb))
    score = fdt.score(X_test, y_test)
    print(f'Accuracy for model {bb} and dataset {ds}: {score}')

if __name__ == "__main__":
    for ds in ['iris', 'wine']:
        print(f'Training model fdt for {ds}')
        train_models(ds, 'FDT', seed)