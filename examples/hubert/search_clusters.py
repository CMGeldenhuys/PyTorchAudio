#!/usr/bin/env python3

from argparse import ArgumentParser
from typing import Dict, Iterable
from itertools import product
from pathlib import Path
from utils.kmeans import learn_kmeans
from tqdm.auto import tqdm
import joblib


def grid_search(search_space: Dict[str, Iterable], silent=False):
    search_iter = [dict(zip(search_space.keys(), val)) for val in product(*search_space.values())]
    pbar = tqdm(search_iter, desc="Searching Hyperparameters", disable=silent)
    for hparams in pbar:
        postfix = ", ".join([f"{key}={val}" for key, val in hparams.items()])
        pbar.set_postfix_str(postfix)
        yield hparams


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--feat-dir", type=Path, required=True)
    parser.add_argument("--output", "-O", type=Path, required=True)

    args = parser.parse_args()

    search_space = dict(
        n_clusters=[50, 75, 100, 200, 350, 500, 1000],
        normalise_feat=[None, 1, 2],
    )

    search_results = []
    for hparams in grid_search(search_space):
        km, metrics = learn_kmeans(
            feat_dir=args.feat_dir,
            split="train",
            num_rank=5,
            n_clusters=100,
            return_metrics=True,
            km_dir=None,
            silent_pbar=True,
        )
        hparams.update(metrics)
        search_results.append(hparams)

    joblib.dump(search_results, args.output)
