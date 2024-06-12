import argparse
import yaml
import numpy as np
from collections import defaultdict
from sklearn import metrics
from sklearn import preprocessing
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, LSTM, Dropout, Input
from keras._tf_keras.keras.optimizers import Adam
import tensorflow as tf
from sklearn import linear_model

SEED = 1
np.random.seed(SEED)
tf.random.set_seed(SEED)

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
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
                    'rnn': defaultdict(list)}

        for fname in fnames:
            print('classifying: %s' % fname)
            label = fname.split('/')[-1]
            data = np.loadtxt(fname, delimiter=',')
            print(data.shape)
            if not neutral:
                data = np.delete(data, np.where(data[:,-1]==0), axis=0)
            np.random.shuffle(data)
            x_data = data[:,:-1]
            y_data = data[:,-1]
            x_data = preprocessing.scale(x_data)

            # Reshape data for RNN
            x_data_rnn = x_data.reshape((x_data.shape[0], x_data.shape[1], 1))

            models = [
                    ('baseline', DummyClassifier(strategy='most_frequent')),
                    ('logit', linear_model.LogisticRegression(max_iter=1000)),
                    ('rf', RandomForestClassifier(n_estimators=N_ESTIMATORS)),
                    ('rnn', build_rnn_model(input_shape=(x_data_rnn.shape[1], 1)))
                    ]

            results['labels'].append(label)
            repeats = 2
            folds = 2
            rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=SEED)

            for key, clf in models:
                scores = {'f1':[], 'acc':[], 'roc_auc':[]}
                for i, (train, test) in enumerate(rskf.split(x_data, y_data)):
                    x_train, x_test = x_data[train], x_data[test]
                    x_train_rnn, x_test_rnn = x_data_rnn[train], x_data_rnn[test]
                    y_train, y_test = y_data[train], y_data[test]
                    if key == 'rnn':
                        clf.fit(x_train_rnn, y_train, epochs=100, batch_size=32, verbose=0)
                        y_pred = (clf.predict(x_test_rnn, verbose=0) > 0.5).astype(int).flatten()
                    else:
                        clf.fit(x_train, y_train)
                        y_pred = clf.predict(x_test)
                    _f1 = metrics.f1_score(y_test, y_pred, average='weighted')
                    _acc = metrics.accuracy_score(y_test, y_pred)
                    if hasattr(clf, 'predict_proba'):
                        y_proba = clf.predict_proba(x_test)
                        _roc_auc = metrics.roc_auc_score(y_test, y_proba[:, 1], average='weighted')
                    else:
                        _roc_auc = 0.0  # Assuming no predict_proba for RNN
                    scores['f1'].append(_f1)
                    scores['acc'].append(_acc)
                    scores['roc_auc'].append(_roc_auc)
                results[key]['f1'].append(np.mean(scores['f1']))
                results[key]['acc'].append(np.mean(scores['acc']))
                results[key]['roc_auc'].append(np.mean(scores['roc_auc']))
        yaml.dump(results, open(condition+'_lift_scores_'+output_file+'.yaml', 'w'))

    if args.mu:
        process_condition(args.mu, 'mu')
    if args.mw:
        process_condition(args.mw, 'mw')
    if args.mo:
        process_condition(args.mo, 'mo')

if __name__ == "__main__":
    main()
