import argparse
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_combined_heatmaps(conf_matrices, labels, title, output_file, conditions):
    num_conditions = len(conditions)
    num_models = len(conf_matrices)
    
    fig, axes = plt.subplots(num_conditions, num_models, figsize=(num_models * 5, num_conditions * 5))
    
    for row, condition in enumerate(conditions):
        for col, model in enumerate(conf_matrices.keys()):
            combined_conf_matrix = np.sum(conf_matrices[model][condition], axis=0)
            sns.heatmap(combined_conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels, ax=axes[row, col], square=True)
            axes[row, col].set_title(f"{model} - {condition}")
            axes[row, col].set_xlabel("Predicted")
            axes[row, col].set_ylabel("True")

    plt.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect to make room for the title
    plt.subplots_adjust(top=0.93)  # Further adjustment to ensure title visibility
    plt.savefig(output_file)
    plt.close()

def permute_and_collect(results):
    models = ['baseline', 'logit', 'rf', 'adaboost', 'gb', 'lgb']
    conf_matrices = {model: {'movie': [], 'music': [], 'music_walk': []} for model in models}

    for model in models:
        for condition in ['movie', 'music', 'music_walk']:
            if condition in results:
                conf_matrix = np.sum(results[condition][model]['conf_matrix'], axis=0)
                conf_matrices[model][condition].append(conf_matrix)
    
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
    conditions = ['movie', 'music', 'music_walk']
    all_conf_matrices = {model: {condition: [] for condition in conditions} for model in ['baseline', 'logit', 'rf', 'adaboost', 'gb', 'lgb']}

    if args.mo:
        with open(args.mo, 'r') as file:
            movie_results = yaml.load(file, Loader=yaml.Loader)
        movie_conf_matrices = permute_and_collect({'movie': movie_results})
        for model in all_conf_matrices:
            for condition in conditions:
                all_conf_matrices[model][condition].extend(movie_conf_matrices[model][condition])

    if args.mu:
        with open(args.mu, 'r') as file:
            music_results = yaml.load(file, Loader=yaml.Loader)
        music_conf_matrices = permute_and_collect({'music': music_results})
        for model in all_conf_matrices:
            for condition in conditions:
                all_conf_matrices[model][condition].extend(music_conf_matrices[model][condition])

    if args.mw:
        with open(args.mw, 'r') as file:
            music_walk_results = yaml.load(file, Loader=yaml.Loader)
        music_walk_conf_matrices = permute_and_collect({'music_walk': music_walk_results})
        for model in all_conf_matrices:
            for condition in conditions:
                all_conf_matrices[model][condition].extend(music_walk_conf_matrices[model][condition])

    plot_title = "Combined Confusion Matrices: Happy vs Sad" if args.classification == "hs" else "Combined Confusion Matrices: Happy vs Neutral vs Sad"
    output_file = f"{args.output_file}.png"

    plot_combined_heatmaps(all_conf_matrices, labels, plot_title, output_file, conditions)

if __name__ == "__main__":
    main()
