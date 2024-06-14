import argparse
import yaml
import numpy as np
from permute.core import one_sample

def permute(results):
    baseline = np.array(results['baseline']['acc'])
    logit = np.array(results['logit']['acc'])
    rf = np.array(results['rf']['acc'])
    gb = np.array(results['gb']['acc'])  # Add 'gb' to the dictionary 'results'
    nb = np.array(results['nb']['acc'])  # Add 'nb' to the dictionary 'results'
    adaboost = np.array(results['adaboost']['acc'])  # Add 'adaboost' to the dictionary 'results'

    baseline_f1 = np.array(results['baseline']['f1'])
    logit_f1 = np.array(results['logit']['f1'])
    rf_f1 = np.array(results['rf']['f1'])
    gb_f1 = np.array(results['gb']['f1'])  # Add 'gb' to the dictionary 'results'
    nb_f1 = np.array(results['nb']['f1'])  # Add 'nb' to the dictionary 'results'
    adaboost_f1 = np.array(results['adaboost']['f1'])  # Add 'adaboost' to the dictionary 'results'

    baseline_roc = np.array(results['baseline']['roc_auc'])
    logit_roc = np.array(results['logit']['roc_auc'])
    rf_roc = np.array(results['rf']['roc_auc'])
    gb_roc = np.array(results['gb']['roc_auc'])  # Add 'gb' to the dictionary 'results'
    nb_roc = np.array(results['nb']['roc_auc'])  # Add 'nb' to the dictionary 'results'
    adaboost_roc = np.array(results['adaboost']['roc_auc'])  # Add 'adaboost' to the dictionary 'results'

    logit_lift = logit - baseline
    rf_lift = rf - baseline
    gb_lift = gb - baseline  # Calculate lift for 'gb'
    nb_lift = nb - baseline  # Calculate lift for 'nb'
    adaboost_lift = adaboost - baseline  # Calculate lift for 'adaboost'

    print('model\t\tAUC\t\tF1\t\tmean\t\tdiff\tp-value')
    print(('baseline\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})'.format(
                    roc=baseline_roc.mean(), roc_std=baseline_roc.std(),
                    f1=baseline_f1.mean(), f1_std=baseline_f1.std(),
                    acc=baseline.mean(), acc_std=baseline.std())))

    (p, diff_means) = one_sample(logit_lift, stat='mean')
    print(('Logit\t\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})\t{diff:.3f}\t{p:.3f}'.format(
            roc=logit_roc.mean(), roc_std=logit_roc.std(),
            f1=logit_f1.mean(), f1_std=logit_f1.std(),
            acc=logit.mean(), acc_std=logit.std(),
            diff=diff_means, p=p)))

    (p, diff_means) = one_sample(rf_lift, stat='mean')
    print(('RF\t\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})\t{diff:.3f}\t{p:.3f}'.format(
            roc=rf_roc.mean(), roc_std=rf_roc.std(),
            f1=rf_f1.mean(), f1_std=rf_f1.std(),
            acc=rf.mean(), acc_std=rf.std(),
            diff=diff_means, p=p)))
    
    (p, diff_means) = one_sample(gb_lift, stat='mean')  # Calculate p-value and difference for 'gb'
    print(('gb\t\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})\t{diff:.3f}\t{p:.3f}'.format(
            roc=gb_roc.mean(), roc_std=gb_roc.std(),
            f1=gb_f1.mean(), f1_std=gb_f1.std(),
            acc=gb.mean(), acc_std=gb.std(),
            diff=diff_means, p=p)))

    (p, diff_means) = one_sample(nb_lift, stat='mean')  # Calculate p-value and difference for 'nb'
    print(('nb\t\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})\t{diff:.3f}\t{p:.3f}'.format(
            roc=nb_roc.mean(), roc_std=nb_roc.std(),
            f1=nb_f1.mean(), f1_std=nb_f1.std(),
            acc=nb.mean(), acc_std=nb.std(),
            diff=diff_means, p=p)))

    (p, diff_means) = one_sample(adaboost_lift, stat='mean')  # Calculate p-value and difference for 'adaboost'
    print(('adaboost\t{roc:.3f} ({roc_std:.3f})\t{f1:.3f} ({f1_std:.3f})\t{acc:.3f} ({acc_std:.3f})\t{diff:.3f}\t{p:.3f}'.format(
            roc=adaboost_roc.mean(), roc_std=adaboost_roc.std(),
            f1=adaboost_f1.mean(), f1_std=adaboost_f1.std(),
            acc=adaboost.mean(), acc_std=adaboost.std(),
            diff=diff_means, p=p)))

    # apa itu std? apa itu p-value? apa itu diff? 
    # std = standard deviation = seberapa jauh data dari mean (rata-rata)
    # p-value = probability value = seberapa besar kemungkinan data yang dihasilkan adalah kebetulan
    # diff = difference = perbedaan antara mean data dengan mean baseline

def main():
    np.random.seed(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    args = parser.parse_args()

    if args.mo:
        with open(args.mo, 'r') as file:
            movie_results = yaml.load(file, Loader=yaml.Loader)
        print('^' * 20)
        print('Movie')
        permute(movie_results)

    if args.mu:
        with open(args.mu, 'r') as file:
            music_results = yaml.load(file, Loader=yaml.Loader)
        print('^' * 20)
        print('Music')
        permute(music_results)

    if args.mw:
        with open(args.mw, 'r') as file:
            music_walk_results = yaml.load(file, Loader=yaml.Loader)
        print('^' * 20)
        print('Music + walking')
        permute(music_walk_results)


if __name__ == "__main__":
    main()
