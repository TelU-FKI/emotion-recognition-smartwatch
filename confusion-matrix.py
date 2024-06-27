import argparse
import yaml
import numpy as np
from permute.core import one_sample

def permute(results):
    models = ['baseline', 'logit', 'rf']

    for model in models:
        print(f"Model: {model}")
        total_conf_matrix = np.sum(results[model]['conf_matrix'], axis=0)
        print(total_conf_matrix)
        print("=" * 50)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mo", type=str, help="movie yaml")
    parser.add_argument("-mu", type=str, help="music yaml")
    parser.add_argument("-mw", type=str, help="music+walk yaml")
    args = parser.parse_args()

    if args.mo:
        with open(args.mo, 'r') as file:
            movie_results = yaml.load(file, Loader=yaml.Loader)
        print('^' * 20)
        print('Movie \n')
        permute(movie_results)

    if args.mu:
        with open(args.mu, 'r') as file:
            music_results = yaml.load(file, Loader=yaml.Loader)
        print('^' * 20)
        print('Music \n')
        permute(music_results)

    if args.mw:
        with open(args.mw, 'r') as file:
            music_walk_results = yaml.load(file, Loader=yaml.Loader)
        print('^' * 20)
        print('Music + walking \n')
        permute(music_walk_results)

if __name__ == "__main__":
    main()
