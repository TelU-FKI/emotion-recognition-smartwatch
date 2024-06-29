import argparse
import yaml
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_combined_heatmap(conf_matrix, labels, title, output_file):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
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
    parser.add_argument("-o", "--output_file", type=str, required=True, help="Base file name for saving the generated plot.")
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

    for model in all_conf_matrices:
        combined_conf_matrix = np.sum(all_conf_matrices[model], axis=0)
        plot_title = f"Combined Confusion Matrix: {model} ({'Happy vs Sad' if args.classification == 'hs' else 'Happy vs Neutral vs Sad'})"
        output_file = f"{args.output_file}_{model}.png"
        plot_combined_heatmap(combined_conf_matrix, labels, plot_title, output_file)

if __name__ == "__main__":
    main()
