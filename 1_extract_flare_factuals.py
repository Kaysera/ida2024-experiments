from utils import seed

import pickle
import numpy as np
import warnings
import sys
import json
from joblib import dump, load

from sklearn.model_selection import train_test_split
from scipy.stats import median_abs_deviation

import random
import logging


# %%
from teacher.explanation import FDTExplainer
from teacher.neighbors import SamplingNeighborhood
from teacher.datasets import load_iris, load_wine, load_beer

# %%
DATASETS = {
    'iris': load_iris,
    'wine': load_wine,
    'beer': load_beer
}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


# %%
def get_dataset_mad(X, cont_idx):
    mad = {}
    for i in cont_idx:
        mad[i] = median_abs_deviation(X[:, i])
        if mad[i] == 0:
            mad[i] += 1
    return mad

# %%
def decode_instance(instance, dataset, fuzzy_variables):
    decoded_instance = []
    for i, var in enumerate(fuzzy_variables):
        try:
            decoded_instance.append(dataset['label_encoder'][var.name].inverse_transform(np.array([instance[i]], dtype=int))[0])
        except:
            decoded_instance += [instance[i]]
    return np.array(decoded_instance, dtype='object')

# %%
def build_explainer(instance, 
                    target, 
                    mad, 
                    class_name, 
                    blackbox, 
                    dataset, 
                    X_train, 
                    get_division, 
                    df_numerical_columns, 
                    df_categorical_columns, 
                    f_method, 
                    cf_method, 
                    cont_idx, 
                    disc_idx, 
                    cf_dist='moth', 
                    size=1000, 
                    max_depth=10, 
                    min_num_examples=10, 
                    neighrange='std', 
                    threshold=None, 
                    fuzzy_threshold=0.0001):
    neighborhood = SamplingNeighborhood(instance, 
                                        size, 
                                        class_name, 
                                        blackbox, 
                                        dataset, 
                                        np.row_stack([X_train, instance]), 
                                        len(X_train), 
                                        neighbor_generation='fast', 
                                        neighbor_range=neighrange)
    neighborhood.fit()
    neighborhood.fuzzify(get_division,
                         class_name=class_name,
                         df_numerical_columns=df_numerical_columns,
                         df_categorical_columns=df_categorical_columns,
                         th=threshold)
    decoded_instance = decode_instance(instance, dataset, neighborhood.get_fuzzy_variables())
    explainer = FDTExplainer()
    if max_depth == -1:
        max_depth = len(X_train[0])
    
    explainer.fit(decoded_instance.reshape(1, -1), 
                  target, 
                  neighborhood, 
                  df_numerical_columns, 
                  f_method, 
                  cf_method, 
                  max_depth=max_depth, 
                  min_num_examples=min_num_examples, 
                  fuzzy_threshold=fuzzy_threshold, 
                  cont_idx=cont_idx, 
                  disc_idx=disc_idx, 
                  mad=mad, 
                  cf_dist=cf_dist)
    
    return explainer, neighborhood, decoded_instance




def get_factual(ds, black_box, random_state, idx_record2explain):
    # %%
    random.seed(random_state)
    np.random.seed(random_state)

    n_path_models = './models/'
    print(n_path_models + '%s_%s.joblib' % (ds, black_box))
    bb = load(n_path_models + '%s_%s.joblib' % (ds, black_box))

    neighrange = 0.5
    cf_dist = 'moth'
    dataset = DATASETS[ds]()

    X, y = dataset['X'], dataset['y']

    # TRAIN 60, DEV 30, TEST 10
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=random_state)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, train_size=0.75, random_state=random_state)


    class_name = dataset['class_name']
    get_division = 'entropy'

    df_numerical_columns = [col for col in dataset['continuous'] if col != class_name]
    df_categorical_columns = [col for col in dataset['discrete'] if col != class_name]

    cont_idx = [key for key, val in dataset['idx_features'].items() if val in df_numerical_columns]
    disc_idx = [key for key, val in dataset['idx_features'].items() if val in df_categorical_columns]

    f_method = 'mr_factual'
    cf_method = 'd_counterfactual'
    # %%
    logging_message = f'Dataset: {ds}, blackbox: {black_box}, exp_id: {idx_record2explain}'
    logging.info(logging_message)
    instance = X_dev[idx_record2explain]
    target = bb.predict(instance.reshape(1, -1))
    mad = get_dataset_mad(X_train, cont_idx)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        explainer, neighborhood, decoded_instance = build_explainer(instance, 
                                                                    target, 
                                                                    mad, 
                                                                    class_name, 
                                                                    bb, 
                                                                    dataset, 
                                                                    X_train, 
                                                                    get_division, 
                                                                    df_numerical_columns, 
                                                                    df_categorical_columns, 
                                                                    f_method, 
                                                                    cf_method, 
                                                                    cont_idx, 
                                                                    disc_idx, 
                                                                    cf_dist, 
                                                                    neighrange=neighrange, 
                                                                    fuzzy_threshold=0.0001)
    

    # %%
    factual, counterfactual = explainer.explain()
    return factual, neighborhood.get_fuzzy_variables()

if __name__ == "__main__":
    random_state = seed
    ds = sys.argv[1]
    bb = sys.argv[2]

    dataset = DATASETS[ds]()

    X, y = dataset['X'], dataset['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=random_state)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, train_size=0.75, random_state=random_state)

    ruleset = []
    instance_number = len(X_dev)
    for idx_record2explain in range(instance_number):
        print(f'Processing instance {idx_record2explain}/{instance_number}: {X_dev[idx_record2explain]}')
        factual, fuzzy_vars = get_factual(
            ds, 
            bb, 
            random_state,
            idx_record2explain,
        )
        for rule in factual:
            ruleset.append(rule.to_json(fuzzy_vars))
        if idx_record2explain % 10 == 0:
            with open(f'rulesets/fuzzy/{ds}_flare_{bb}_{idx_record2explain}.json', 'w') as f:
                json.dump(ruleset, f, indent=4, cls=NpEncoder)
            ruleset = []
                
    # And we save those rules in JSON format in the folder rulesets/fuzzy
    # In the filename we specify that this is a fuzzy decision tree

    with open(f'rulesets/fuzzy/{ds}_flare_{bb}_{idx_record2explain}.json', 'w') as f:
        json.dump(ruleset, f, indent=4, cls=NpEncoder)
