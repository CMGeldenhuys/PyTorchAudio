#!/usr/bin/env python3

from argparse import ArgumentParser
from typing import Dict, Iterable
from itertools import product
from pathlib import Path
from utils.kmeans import learn_kmeans
from tqdm.auto import tqdm
import joblib


def grid_search(search_space: Dict[str, Iterable]):
    yield from (dict(zip(search_space.keys(), val)) for val in product(*search_space.values()))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--feat-dir", type=Path, required=True)
    parser.add_argument("--output", "-O", type=Path, required=True)

    args = parser.parse_args()

    search_space = dict(
        n_clusters=[50, 100, 500],
        normalise_feat=[None, 2],
    )

    p_bar = tqdm(
        list(grid_search(search_space)),
        desc="Searching hparam",
    )
    search_results = []
    for hparams in p_bar:
        km, inertia = learn_kmeans(
            feat_dir=args.feat_dir,
            split="train",
            num_rank=5,
            n_clusters=100,
            return_inertia=True,
            km_dir=None,
            silent_pbar=True,
        )
        hparams["inertia"] = inertia
        search_results.append(hparams)

    joblib.dump(search_results, args.output)
