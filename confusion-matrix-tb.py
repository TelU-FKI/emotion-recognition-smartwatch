import argparse
import yaml
import numpy as np

def save_consolidated_conf_matrices(conf_matrices, labels, output_file):
    with open(output_file, 'w') as f:
        f.write(f"{'Model':<15}{'Condition':<20}{'True/Predicted':<20}{' '.join(f'{label:<10}' for label in labels)}\n")
        f.write("=" * 80 + "\n")
        for model, conditions in conf_matrices.items():
            for condition, matrix in conditions.items():
                for i, row in enumerate(matrix):
                    f.write(f"{model:<15}{condition:<20}{labels[i]:<20}{' '.join(f'{val:<10}' for val in row)}\n")
                f.write("\n")

def permute_and_collect(results, condition):
    models = ['baseline', 'logit', 'rf', 'adaboost', 'gb', 'lgb']
    conf_matrices = {model: {} for model in models}

    for model in models:
        total_conf_matrix = np.sum(results[model]['conf_matrix'], axis=0)
        conf_matrices[model][condition] = total_conf_matrix
    
    return conf_matrices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    parser.add_argument("--classification", type=str, choices=["hs", "hns"], required=True, help="Type of classification: 'hs' for Happy vs Sad, 'hns' for Happy vs Neutral vs Sad")
    parser.add_argument("-o", "--output_file", type=str, required=True, help="File name for saving the generated table.")
    args = parser.parse_args()

    labels = ["Happy", "Sad"] if args.classification == "hs" else ["Happy", "Neutral", "Sad"]

    all_conf_matrices = {model: {} for model in ['baseline', 'logit', 'rf', 'adaboost', 'gb', 'lgb']}

    if args.mo:
        with open(args.mo, 'r') as file:
            movie_results = yaml.load(file, Loader=yaml.Loader)
        movie_conf_matrices = permute_and_collect(movie_results, "Movie")
        for model in all_conf_matrices:
            all_conf_matrices[model].update(movie_conf_matrices[model])

    if args.mu:
        with open(args.mu, 'r') as file:
            music_results = yaml.load(file, Loader=yaml.Loader)
        music_conf_matrices = permute_and_collect(music_results, "Music")
        for model in all_conf_matrices:
            all_conf_matrices[model].update(music_conf_matrices[model])

    if args.mw:
        with open(args.mw, 'r') as file:
            music_walk_results = yaml.load(file, Loader=yaml.Loader)
        music_walk_conf_matrices = permute_and_collect(music_walk_results, "Music+Walk")
        for model in all_conf_matrices:
            all_conf_matrices[model].update(music_walk_conf_matrices[model])

    output_file = f"{args.output_file}.txt"
    save_consolidated_conf_matrices(all_conf_matrices, labels, output_file)

if __name__ == "__main__":
    main()
