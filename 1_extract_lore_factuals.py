from utils import seed, LORE_PATH

import sys

sys.path.insert(1, LORE_PATH)

import lore

from prepare_dataset import *
from neighbor_generator import *
from teacher.datasets import load_compas, load_german, load_adult, load_heloc, load_iris, load_wine, load_beer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from numpy import inf
import numpy as np

import json
import random
import sys
import warnings
from joblib import dump, load


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


warnings.filterwarnings("ignore")

def parse_exp(explanation, dataset):
    new_exp = {}
    rule = explanation[1]
    for premise in rule:
        val = rule[premise]
        if premise in dataset['discrete']:
            new_exp[premise] = val
        else:
            if val[0] == '<':
                new_exp[premise] = (-inf, float(val[2:]))
            elif val[0] == '>':
                new_exp[premise] = (float(val[1:]), inf)
            else:
                new_exp[premise] = (float(val.split('<')[0]), float(val.split('<=')[1]))
    new_exp['label'] = list(explanation[0].values())[0]
    return new_exp


def main():

    print('Loading dataset')
    random_state = seed
    ds = sys.argv[1]
    black_box = sys.argv[2]
    random.seed(seed)
    np.random.seed(seed)

    n_path_models = './models/'
    print(n_path_models + '%s_%s.joblib' % (ds, black_box))
    bb = load(n_path_models + '%s_%s.joblib' % (ds, black_box))

    dataset = DATASETS[ds]()

    X, y = dataset['X'], dataset['y']
    # TRAIN 60, DEV 30, TEST 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=random_state)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, train_size=0.75, random_state=random_state)

    print('Training model')

    X2E = X_dev
    y2E = bb.predict(X2E)
    y2E = np.asarray([dataset['possible_outcomes'][i] for i in y2E])

    for idx_record2explain in range(len(X2E)):
        try:
            print(f'Explaining instance {idx_record2explain}')

            explanation, infos = lore.explain(idx_record2explain, X2E, dataset, bb,
                                            ng_function=genetic_neighborhood,
                                            discrete_use_probabilities=True,
                                            continuous_function_estimation=False,
                                            returns_infos=True,
                                            path='./', sep=';', log=False)

            dfX2E = build_df2explain(bb, X2E, dataset).to_dict('records')
            dfx = dfX2E[idx_record2explain]
            # x = build_df2explain(blackbox, X2E[idx_record2explain].reshape(1, -1), dataset).to_dict('records')[0]

            exp = parse_exp(explanation[0], dataset)
            with open(f'./rulesets/crisp/{ds}_lore_{black_box}_{idx_record2explain}.json', 'w+') as f:
                json.dump(exp, f)
        except:
            print('Explanation not extracted')

 
if __name__ == "__main__":
    main()
