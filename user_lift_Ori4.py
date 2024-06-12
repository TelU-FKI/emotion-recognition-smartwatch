import argparse
import yaml
import numpy as np
from collections import defaultdict
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

SEED = 1
np.random.seed(SEED)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mu", metavar='mu', type=str, nargs='+', help="file containing music features", default=[])
    parser.add_argument("-mw", metavar='mw', type=str, nargs='+', help="file containing music+walking features", default=[])
    parser.add_argument("-mo", metavar='mo', type=str, nargs='+', help="file containing movie features", default=[])
    parser.add_argument("-e", "--estimators", help="number of estimators for meta-classifiers", type=int, default=100)
    parser.add_argument("-o", "--output_file", help="output with pickle results", type=str)
    parser.add_argument("--neutral", action='store_true', help="classify happy-sad-neutral")
    args = parser.parse_args()

    output_file = args.output_file
    N_ESTIMATORS = args.estimators
    neutral = args.neutral

    def process_condition(fnames, condition):
        if not fnames: 
            return
        print('Condition:', condition)

        results = {'labels':[], 'baseline': defaultdict(list),
                    'logit': defaultdict(list), 
                    'rf': defaultdict(list),
                    'lgb': defaultdict(list),
                    'catboost': defaultdict(list)}

        for fname in fnames:
            print('Classifying:', fname)
            label = fname.split('/')[-1]

            data = np.loadtxt(fname, delimiter=',')
            print('Data shape:', data.shape)

            if not neutral:
                data = np.delete(data, np.where(data[:,-1] == 0), axis=0)

            np.random.shuffle(data)
            x_data = data[:,:-1]
            y_data = data[:,-1]

            x_data = preprocessing.scale(x_data)

            models = [
                    ('baseline', DummyClassifier(strategy='most_frequent')),
                    ('logit', linear_model.LogisticRegression(max_iter=1000)),
                    ('rf', RandomForestClassifier(n_estimators=N_ESTIMATORS)),
                    ('lgb', lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, verbose=-1)),  # Set verbose for detailed logs
                    ('catboost', CatBoostClassifier(n_estimators=N_ESTIMATORS, verbose=0))
            ]

            results['labels'].append(label)
            repeats = 2
            folds = 2
            rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)

            for key, clf in models:
                scores = {'f1':[], 'acc':[], 'roc_auc':[]}
                for train, test in rskf.split(x_data, y_data):
                    x_train, x_test = x_data[train], x_data[test]
                    y_train, y_test = y_data[train], y_data[test]
                    clf.fit(x_train, y_train)
                    y_pred = clf.predict(x_test)
                    _f1 = metrics.f1_score(y_test, y_pred, average='weighted')
                    _acc = metrics.accuracy_score(y_test, y_pred)
                    y_proba = clf.predict_proba(x_test)
                    _roc_auc = metrics.roc_auc_score(y_test, y_proba[:, 1], average='weighted')
                    scores['f1'].append(_f1)
                    scores['acc'].append(_acc)
                    scores['roc_auc'].append(_roc_auc)

                results[key]['f1'].append(np.mean(scores['f1']))
                results[key]['acc'].append(np.mean(scores['acc']))
                results[key]['roc_auc'].append(np.mean(scores['roc_auc']))

        yaml.dump(results, open(condition + '_lift_scores_' + output_file + '.yaml', 'w'))

    if args.mu:
        process_condition(args.mu, 'mu')
    if args.mw:
        process_condition(args.mw, 'mw')
    if args.mo:
        process_condition(args.mo, 'mo')

if __name__ == "__main__":
    main()
