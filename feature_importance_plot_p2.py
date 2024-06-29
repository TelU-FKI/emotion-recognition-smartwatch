import argparse
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def aggregate_impor(impor):
    new_impor = []
    for user_impor in impor:
        user_impor = np.array(user_impor)
        sum_impor = np.mean(user_impor, axis=0)
        sum_impor = sum_impor / np.max(sum_impor)
        new_impor.append(sum_impor)

    medians = np.median(np.array(new_impor), axis=0)
    indices = range(len(medians))
    indices = sorted(indices, key=lambda i: medians[i], reverse=True)
    return np.array(new_impor).T[indices], indices

def make_legend(colors):
    """Make a legend with adjusted positioning."""
    acc_patch = mpatches.Patch(label='Acc', facecolor=colors[0], edgecolor='black')
    gyro_patch = mpatches.Patch(label='Gyro', facecolor=colors[1], edgecolor='black')
    heart_patch = mpatches.Patch(label='Heart', facecolor=colors[2], edgecolor='black', hatch=r'\\')

    plt.rcParams["legend.fontsize"] = 11.5
    
    l = plt.legend(handles=[heart_patch, acc_patch, gyro_patch],
                  loc='upper center', bbox_to_anchor=(0.5, 1.2),  # Moved higher
                  frameon=False, ncol=3, borderaxespad=0.2,
                  columnspacing=0.7, handletextpad=0.3,
                  fancybox=True, framealpha=0.3)
    l.get_frame().set_facecolor('white')
    return l

def main():
    parser = argparse.ArgumentParser("plot accelerometer data")
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    parser.add_argument("-o", "--output_file", type=str, help="file name for saving the plot.", default="output3")
    parser.add_argument("-r", "--dpi", type=int, help="resolution of image", default=300)

    args = parser.parse_args()
    dpi = args.dpi
    output_file = args.output_file

    plt.style.use('seaborn-whitegrid')
    plt.rcParams.update({'font.size': 10})

    with open(args.mo, 'r') as file:
        movie = yaml.load(file, Loader=yaml.Loader)
    with open(args.mu, 'r') as file:
        music = yaml.load(file, Loader=yaml.Loader)
    with open(args.mw, 'r') as file:
        music_walk = yaml.load(file, Loader=yaml.Loader)
    data = [movie['lgb'], music['lgb'], music_walk['lgb']]

    colors = ['#cce5ff', '#ffcccc', '#fff8ab']
    titles = ['Movie', 'Music', 'Music while walking']
    labels_unsorted = [s.strip() for s in open('feature_list').readlines()]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 18), dpi=dpi, sharex=True)

    for i, (ax, group, title) in enumerate(zip(axes, data, titles)):
        feature_impor = group
        feature_impor_all, indices = aggregate_impor(feature_impor)
        labels = [labels_unsorted[s] for s in indices]

        limit = 20
        bp = ax.boxplot(feature_impor_all[:limit].T, patch_artist=True, sym='+')

        for median in bp['medians']:
            median.set(color='red', linewidth=1.8)

        for j, box in enumerate(bp['boxes']):
            if 'acc_' in labels[j] or 'mag' in labels[j]:
                box.set(facecolor=colors[0])
            elif 'gyro_' in labels[j]:
                box.set(facecolor=colors[1])
            elif 'heart' in labels[j]:
                box.set(facecolor=colors[2])
                box.set(hatch=r'\\')

        short_labels = [re.sub(r'^(acc_|gyro_)', '', l) for l in labels[:limit]]
        ax.set_xticks(range(1, limit + 1))
        ax.set_xticklabels(short_labels, fontsize=9, rotation=70)
        ax.set_ylabel('Feature Importance')
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
        ax.tick_params(labelright=True)
        ax.set_title(title, fontsize=12)

    make_legend(colors)  # Use the modified legend function

    plt.suptitle('Feature Importances per Condition', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout 
    plt.subplots_adjust(top=0.92)            # More space at the top
    plt.savefig(output_file + '.png', bbox_inches='tight')  

if __name__ == "__main__":
    main()
