#!/usr/bin/env python3
"""This is the preprocessing script for HuBERT model training.
The script includes:
    - File list creation
    - MFCC/HuBERT feature extraction
    - KMeans clustering model training
    - Pseudo-label generation
"""
import logging
from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path

import torch
from utils import create_tsv, dump_features, get_km_label, learn_kmeans

_LG = logging.getLogger(__name__)


def _init_logger(debug=False):
    message_fmt = "%(levelname)5s: %(funcName)10s: %(message)s" if debug else "%(message)s"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format=f"%(asctime)s: {message_fmt}",
    )


def _parse_args():
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug log")
    parser.add_argument("--dataset", default="librispeech", type=str, choices=["librispeech", "librilight"])
    parser.add_argument(
        "--root-dir",
        type=Path,
        help="The path to the directory where the directory ``LibriSpeech`` or ``LibriLight`` is stored.",
    )
    parser.add_argument("--num-rank", default=5, type=int)
    parser.add_argument("--feat-type", default="mfcc", choices=["mfcc", "hubert", "lfcc", "lfcc_wide"], type=str)
    parser.add_argument("--widen-feature-extractor", default=None, type=int)
    parser.add_argument(
        "--layer-index",
        default=6,
        type=int,
        help="The layer index in HuBERT model for feature extraction. (``1`` means the first layer output)",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        type=Path,
        help="The model checkpoint of hubert_pretrain_base model.",
    )
    parser.add_argument(
        "--model-num-classes",
        default=100,
        type=int,
        help="Number of classes for hubert_pretrain_base_model.",
    )
    parser.add_argument("--use-gpu", default=False, action="store_true")
    parser.add_argument("--kmeans-cpu", default=None, action="store_true")
    parser.add_argument(
        "--exp-dir",
        type=Path,
        help="The directory to store the experiment outputs.",
    )
    parser.add_argument(
        "--num-cluster",
        default=100,
        type=int,
        help="The number of clusters for KMeans clustering.",
    )
    parser.add_argument(
        "--percent",
        default=-1,
        type=float,
        help="The percent of data for KMeans clustering. If negative, use all data. (Default: -1)",
    )
    parser.add_argument(
        "--skip-tsv-if-exists", default=False, action="store_true", help="If tsv file already exists skip rebuilding."
    )
    parser.add_argument(
        "--skip-feat-if-exists", default=False, action="store_true", help="If feat dir already exists skip extraction."
    )
    parser.add_argument(
        "--sample-rate",
        default=16_000,
        type=int,
        help="Sample rate at which to compute features",
    )
    parser.add_argument(
        "--ext",
        default="flac",
        type=str,
        choices=["flac", "wav"],
    )
    args = parser.parse_args()
    return args


def main(args):
    _init_logger(args.debug)

    if not args.exp_dir.exists():
        args.exp_dir.mkdir()
    if args.feat_type == "mfcc":
        data_dir: Path = args.exp_dir / "data" / "mfcc"
    else:
        data_dir: Path = args.exp_dir / "data" / f"{args.feat_type}_{args.layer_index}"
    data_dir.mkdir(parents=True, exist_ok=True)

    tsv_dir = data_dir / "tsv"
    feat_dir = data_dir / "feat"
    km_dir = data_dir / "km_model"
    label_dir = data_dir / "label"

    if args.use_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.kmeans_cpu:
        km_device = torch.device("cpu")
    else:
        km_device = device

    if not (tsv_dir.exists() and args.skip_tsv_if_exists):
        # Create file lists for training and validation (optional)
        create_tsv(args.root_dir, tsv_dir, extension=args.ext)
    else:
        _LG.info("Skipping tsv file rebuilding, using existing")

    # Extract features for KMeans clustering
    if not (feat_dir.exists() and args.skip_feat_if_exists):
        if not feat_dir.exists():
            feat_dir.mkdir()

        for split in ["train", "valid"]:
            _LG.info(f"Processing features for split={split}")
            for rank in range(1, args.num_rank + 1):
                dump_features(
                    tsv_dir / f"{args.dataset}_{split}.tsv",
                    feat_dir,
                    split,
                    rank,
                    args.num_rank,
                    device,
                    args.feat_type,
                    args.layer_index,
                    args.checkpoint_path,
                    args.sample_rate,
                    args.model_num_classes,
                    args.widen_feature_extractor,
                )
    else:
        _LG.info("Skipping feat generation")

    # Fit KMeans clustering model
    learn_kmeans(
        feat_dir,
        "train",
        args.num_rank,
        km_dir,
        args.num_cluster,
        args.percent,
    )

    # Predict labels for MFCC or HuBERT features
    for split in ["train", "valid"]:
        get_km_label(
            feat_dir,
            km_dir,
            label_dir,
            split,
            args.num_rank,
            km_device,
        )


if __name__ == "__main__":
    main(_parse_args())
