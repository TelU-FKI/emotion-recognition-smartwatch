import argparse
# import glob
import yaml
import numpy as np
from collections import defaultdict

from sklearn import linear_model, metrics, model_selection, preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV  # Change this import

from permute.core import one_sample


SEED = 1
np.random.seed(SEED)

def main():
    '''
    Run as:
    python user_lift.py -mo features/features_mo* -mw features/features_mw* -mu features/features_mu*

    Takes features generated by extract_windows.py script, runs classifier, and prints accuracy results.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-mu", metavar='mu', type=str, nargs='+', help="file containing music features, input to model", default=[])
    parser.add_argument("-mw", metavar='mw', type=str, nargs='+', help="file containing music+walking features, input to model", default=[])
    parser.add_argument("-mo", metavar='mo', type=str, nargs='+', help="file containing movie features, input to model", default=[])
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
        print('condition', condition)

        results = {'labels':[], 'baseline': defaultdict(list),
                    'logit': defaultdict(list), 
                    'rf': defaultdict(list),
                    'rf_tuning': defaultdict(list)}  # Add 'rf_tuning' key to the 'results' dictionary

        for fname in fnames:
            print('classifying: %s' % fname)
            label = fname.split('/')[-1]

            data = np.loadtxt(fname, delimiter=',')

            # only acc: acc + y_label as column vector
            #data = np.hstack([data[:,:51], data[:,-1].reshape(data.shape[0], 1)])

            # acc features + heart rate + y label
            #data = np.hstack([data[:,:51], data[:,-2:]])
            print(data.shape)

            if not neutral:
                # delete neutral to see if we can distinguish between
                # happy/sad
                data = np.delete(data, np.where(data[:,-1]==0), axis=0)

            np.random.shuffle(data)

            x_data = data[:,:-1]
            y_data = data[:,-1]

            # scaled
            x_data = preprocessing.scale(x_data)

            # Number of trees in random forest
            n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
            # Number of features to consider at every split
            max_features = ['sqrt', 'log2', None]
            # Maximum number of levels in tree
            max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
            max_depth.append(None)
            # Minimum number of samples required to split a node
            min_samples_split = [2, 5, 10]
            # Minimum number of samples required at each leaf node
            min_samples_leaf = [1, 2, 4]
            # Method of selecting samples for training each tree
            bootstrap = [True, False]

            # Create the random grid
            param_dist = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap}
            
            # Create a separate instance of RandomForestClassifier for tuning
            base_rf = RandomForestClassifier(n_estimators=N_ESTIMATORS)

            # Use RandomizedSearchCV for hyperparameter tuning
            rf_tuner = RandomizedSearchCV(estimator=base_rf, param_distributions=param_dist, n_iter=100, cv=3, random_state=42, n_jobs=-1, verbose=0)

            models = [
                ('baseline', DummyClassifier(strategy='most_frequent')),
                ('logit', linear_model.LogisticRegression(max_iter=10000)),
                ('rf', base_rf),  # Use the base_rf instance here
                ('rf_tuning', rf_tuner)  # Use rf_tuner for hyperparameter tuning
            ]
                    
            results['labels'].append(label)
            repeats = 10
            folds = 10
            rskf = RepeatedStratifiedKFold(n_splits=folds, 
                                        n_repeats=repeats,
                                        random_state=SEED)

            for key, clf in models:
                    scores = {'f1':[], 'acc':[], 'roc_auc':[]}
                    for i, (train,test) in enumerate(rskf.split(x_data, y_data)):
                        x_train, x_test = x_data[train], x_data[test]
                        y_train, y_test = y_data[train], y_data[test]
                        clf.fit(x_train, y_train)
                        if key == 'rf_tuning':
                            print("Best Parameters:", clf.best_params_)
                            print("Best Score:", clf.best_score_)
                        y_pred = clf.predict(x_test)
                        _f1 = metrics.f1_score(y_test, y_pred, average='weighted')
                        _acc = metrics.accuracy_score(y_test, y_pred)
                        if hasattr(clf, 'predict_proba'):
                            y_proba = clf.predict_proba(x_test)
                            if len(np.unique(y_test)) > 2:  # Multi-class scenario
                                _roc_auc = metrics.roc_auc_score(y_test, y_proba, average='weighted', multi_class='ovr')
                            else:  # Binary classification
                                _roc_auc = metrics.roc_auc_score(y_test, y_proba[:, 1], average='weighted', multi_class='ovr')
                            if not np.isnan(_roc_auc):
                                scores['roc_auc'].append(_roc_auc)

                        else:
                            _roc_auc = None
                        scores['f1'].append(_f1)
                        scores['acc'].append(_acc)

                    results[key]['f1'].append(np.mean(scores['f1']))
                    results[key]['acc'].append(np.mean(scores['acc']))
                    if scores['roc_auc']:  # Check if the list is not empty
                        results[key]['roc_auc'].append(np.mean(scores['roc_auc']))
                    else:
                        results[key]['roc_auc'].append(None)

                # #results[key] = {'f1': np.mean(scores['f1']), 'acc': np.mean(scores['acc']), 'f1_all': scores['f1'], 'acc_all':scores['acc']}
                # results[key]['f1'].append(np.mean(scores['f1']))
                # results[key]['acc'].append(np.mean(scores['acc']))
                # results[key]['roc_auc'].append(np.mean(scores['roc_auc']))

        #for key, model in models:
        #    print key, np.mean(results[key]), np.std(results[key])

        yaml.dump(results, open(condition+'_lift_scores_'+output_file+'.yaml', 'w'))

    # end of function
    #-------------

    if args.mu:
        process_condition(args.mu, 'mu')
    if args.mw:
        process_condition(args.mw, 'mw')
    if args.mo:
        process_condition(args.mo, 'mo')


if __name__ == "__main__":
    main()
