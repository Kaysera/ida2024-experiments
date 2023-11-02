# %%
from flocalx.utils import FuzzyContinuousSet, get_fuzzy_variables
from teacher.datasets import load_german, load_compas, load_adult, load_heloc, load_iris, load_wine, load_beer
import numpy as np
import json
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from teacher.fuzzy import get_fuzzy_points
from flocalx.rule import FLocalX
from flocalx.genetic import GeneticAlgorithm
from os import listdir
from os.path import isfile, join
from joblib import load


from utils import seed
import hydra
from omegaconf import DictConfig, OmegaConf

# %%
DATASETS = {
    'adult': load_adult,
    'compas': load_compas,
    'german': load_german,
    'fico': load_heloc,
    'iris': load_iris,
    'wine': load_wine,
    'beer': load_beer
}

def convert_ruleset(ruleset, discrete, X, dataset, class_name):
    new_ruleset = []
    for rule in ruleset:
        new_antecedent = {}
        new_fuzzy_sets = []
        for premise in rule:
            if premise == 'label':
                continue
            if premise in discrete:
                new_antecedent[premise] = rule[premise]
                new_fuzzy_sets.append([premise, rule[premise]])
            else:
                minn, maxx = rule[premise]
                if minn == -np.inf:
                    mid = (maxx - X[premise].min()) / 2
                    new_fs = [X[premise].min(), X[premise].min(), maxx + mid]
                    new_antecedent[premise] = str(X[premise].min())
                    new_fuzzy_sets.append([premise, str(X[premise].min()), new_fs])
                elif maxx == np.inf:
                    mid = (X[premise].max() - minn) / 2
                    new_fs = [minn - mid, X[premise].max(), X[premise].max()]
                    new_antecedent[premise] = str(X[premise].max())
                    new_fuzzy_sets.append([premise, str(X[premise].max()), new_fs])
                else:
                    mid = (maxx - minn) / 2
                    new_fs = [minn - mid, minn + mid, maxx + mid]
                    new_antecedent[premise] = str(minn + mid)
                    new_fuzzy_sets.append([premise, str(minn + mid), new_fs])
        new_label = dataset['label_encoder'][class_name].transform(np.array(rule['label']).reshape(1, -1))[0]

        new_rule = [new_antecedent, new_label, 1, new_fuzzy_sets]
        if new_antecedent:
            new_ruleset.append(new_rule)

    return new_ruleset

def get_variables_metadata(fuzzy_variables, discrete_fuzzy_values, fuzzy_variables_order):
    variables_metadata = {}
    variables_metadata['discrete_fuzzy_values'] = discrete_fuzzy_values
    variables_metadata['fuzzy_variables_order'] = fuzzy_variables_order
    counter = 0
    variables_metadata['continuous'] = {}
    variables_metadata['sets'] = 0
    for i, var in enumerate(fuzzy_variables):
        if isinstance(var.fuzzy_sets[0], FuzzyContinuousSet):
            variables_metadata[i] = {
                'name': var.name,
                'min': float(var.fuzzy_sets[0].name),
                'max': float(var.fuzzy_sets[-1].name),
                'points': {float(fs.name): i for i, fs in enumerate(var.fuzzy_sets)}
            }
            variables_metadata['continuous'][i] = counter
            counter += 1
            variables_metadata['sets'] = len(var.fuzzy_sets)
        else:
            variables_metadata[i] = {
                'name': var.name,
                'values': {fs.name: i for i, fs in enumerate(var.fuzzy_sets)}
            }
    return variables_metadata


def load_ruleset_from_local_explanations(my_path, db, method, bb='', n_explanations=None):
    ruleset = []
    onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f)) and db in f and method in f and bb in f]
    for file in onlyfiles:
        with open(join(my_path, file), 'r') as f:
            ruleset.append(json.load(f))
    if n_explanations is None:
        return ruleset
    
    if n_explanations > len(ruleset):
        n_explanations = len(ruleset)

    positive_ruleset = [r for r in ruleset if r[1] == 1]
    negative_ruleset = [r for r in ruleset if r[1] == 0]

    if n_explanations > 2*len(positive_ruleset):
        n_explanations = 2*len(positive_ruleset)

    if n_explanations > 2*len(negative_ruleset):
        n_explanations = 2*len(negative_ruleset)

    positive_ruleset = random.sample(positive_ruleset, n_explanations//2)
    negative_ruleset = random.sample(negative_ruleset, n_explanations//2)

    ruleset = positive_ruleset + negative_ruleset
    return ruleset


@hydra.main(version_base=None, config_path="conf", config_name="multiseed_lore")
def experiment(cfg : DictConfig) -> None:
    # random.seed(seed)
    # np.random.seed(seed)
    
    # Fixed Hyperparameters
    fuzzy_sets = 5
    mutation_prob = 0.15
    crossover_prob = 0.8
    
    # Moving Hyperparameters
    db = cfg.dataset
    method = cfg.method
    population_size = cfg.population_size
    size_pressure =  cfg.size_pressure
    kappa = cfg.kappa
    epsilon = cfg.epsilon
    seed = cfg.seed
    bb = cfg.bb


    # Parameters
    random_state = np.random.default_rng(seed=seed)

    # list all files in folder
    my_path = './rulesets/crisp/'

    print('Loading dataset')
    dataset = DATASETS[db]()
    class_name = dataset['class_name']
    df = dataset['df']
    continuous = dataset['continuous']  
    discrete = dataset['discrete']
    n_path_models = './models/'
    clf = load(n_path_models + '%s_%s.joblib' % (db, bb))

    X = df.drop(class_name, axis=1)
    y = df[class_name]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=seed)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, train_size=0.75, random_state=seed)
    clf_results = clf.predict(X_test)

    X_num = X_dev[continuous]

    antecedent_order = {v: k for k, v in dataset['idx_features'].items()}
    dataset_info = {
        'antecedent_order': antecedent_order,
        'discrete': set(dataset['discrete']),
        'continuous': set(dataset['continuous']),
        'max_class': max(dataset['y'][y_dev.index])
    }

    print('Loading ruleset')
    ruleset = load_ruleset_from_local_explanations(my_path, db, method, bb)
    ruleset = convert_ruleset(ruleset, discrete, X_dev, dataset, class_name)

    flocal = FLocalX.from_json(ruleset, dataset_info, random_state)
    flocal_score = flocal.score(X_test, dataset['y'][y_test.index])
    flocal_size = flocal.size()
    flocal_rule_size = flocal.rule_size()
    flocal_fidelity = accuracy_score(clf_results, flocal.predict(X_test))

    print('Mapping ruleset to global variables')
    fuzzy_points = get_fuzzy_points('equal_width', continuous, X_num, sets=fuzzy_sets)
    cate = [col for col in discrete if col != class_name]
    discrete_fuzzy_values = {col: X_dev[col].unique() for col in cate}
    fuzzy_variables_order = {col: i for i, col in enumerate(X_dev.columns)}
    fuzzy_variables = get_fuzzy_variables(fuzzy_points, discrete_fuzzy_values, fuzzy_variables_order)
    var_metadata = get_variables_metadata(fuzzy_variables, discrete_fuzzy_values, fuzzy_variables_order)
    var_metadata['max_class'] = max(dataset['y'][y_dev.index])
    flocal.variable_mapping(fuzzy_variables)
    initial_chromosome = flocal.chromosome(var_metadata)

    print('Fitting global rule system')
    gen = GeneticAlgorithm(var_metadata, 
                        X_dev, 
                        dataset['y'][X_dev.index], 
                        mutation_prob=mutation_prob,
                        crossover_prob=crossover_prob,
                        kappa=kappa,
                        stagnation=True, 
                        size_pressure=size_pressure, 
                        population_size=population_size,
                        epsilon=epsilon, 
                        initial_chromosomes=[initial_chromosome],
                        random_state=random_state)

    gen()
    best = max(gen.population)
    best_flocal = best.to_rule_based_system(var_metadata)
    best_flocal_size = best_flocal.size()
    best_flocal_score = best_flocal.score(X_test, dataset['y'][y_test.index])
    best_flocal_rule_size = best_flocal.rule_size()
    best_flocal_fidelity = accuracy_score(clf_results, best_flocal.predict(X_test))

    print((best_flocal_size, best_flocal_score, best_flocal_rule_size, best_flocal_fidelity, flocal_score, flocal_size, flocal_rule_size, flocal_fidelity))

    file_name = f'seed_flocallore_{seed}_{db}_{method}_{population_size}_{size_pressure}_{kappa}_{epsilon}_{bb}.txt'
    with open(f'./new_results/{file_name}', 'w+') as f:
        f.write(str((best_flocal_size, best_flocal_score, best_flocal_rule_size, best_flocal_fidelity, flocal_score, flocal_size, flocal_rule_size, flocal_fidelity)))


if __name__ == '__main__':
    experiment()