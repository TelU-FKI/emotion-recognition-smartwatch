import argparse
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_combined_heatmaps(conf_matrices, labels, title, output_file):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    models = list(conf_matrices.keys())
    
    for i, model in enumerate(models):
        combined_conf_matrix = np.sum(conf_matrices[model], axis=0)
        sns.heatmap(combined_conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels, ax=axes[i])
        axes[i].set_title(model)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("True")
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def permute_and_collect(results):
    models = ['baseline', 'logit', 'rf', 'adaboost', 'gb', 'lgb']
    conf_matrices = {model: [] for model in models}

    for model in models:
        total_conf_matrix = np.sum(results[model]['conf_matrix'], axis=0)
        conf_matrices[model].append(total_conf_matrix)
    
    return conf_matrices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    parser.add_argument("--classification", type=str, choices=["hs", "hns"], required=True, help="Type of classification: 'hs' for Happy vs Sad, 'hns' for Happy vs Neutral vs Sad")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="File name for saving the generated plot.")
    args = parser.parse_args()

    labels = ["Happy", "Sad"] if args.classification == "hs" else ["Happy", "Neutral", "Sad"]

    all_conf_matrices = {model: [] for model in ['baseline', 'logit', 'rf', 'adaboost', 'gb', 'lgb']}

    if args.mo:
        with open(args.mo, 'r') as file:
            movie_results = yaml.load(file, Loader=yaml.Loader)
        movie_conf_matrices = permute_and_collect(movie_results)
        for model in all_conf_matrices:
            all_conf_matrices[model].extend(movie_conf_matrices[model])

    if args.mu:
        with open(args.mu, 'r') as file:
            music_results = yaml.load(file, Loader=yaml.Loader)
        music_conf_matrices = permute_and_collect(music_results)
        for model in all_conf_matrices:
            all_conf_matrices[model].extend(music_conf_matrices[model])

    if args.mw:
        with open(args.mw, 'r') as file:
            music_walk_results = yaml.load(file, Loader=yaml.Loader)
        music_walk_conf_matrices = permute_and_collect(music_walk_results)
        for model in all_conf_matrices:
            all_conf_matrices[model].extend(music_walk_conf_matrices[model])

    plot_title = "Combined Confusion Matrices: Happy vs Sad" if args.classification == "hs" else "Combined Confusion Matrices: Happy vs Neutral vs Sad"
    output_file = f"{args.output_file}.png"

    plot_combined_heatmaps(all_conf_matrices, labels, plot_title, output_file)

if __name__ == "__main__":
    main()
