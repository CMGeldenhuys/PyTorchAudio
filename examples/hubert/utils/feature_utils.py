#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# https://github.com/pytorch/fairseq/blob/265df7144c79446f5ea8d835bda6e727f54dad9d/LICENSE
import logging
import math
from pathlib import Path
from typing import Optional, Tuple, Union
import warnings

import torch
import torchaudio
from torch import Tensor
from torch.nn import Module

from .common_utils import _get_feat_lens_paths

_LG = logging.getLogger(__name__)
_DEFAULT_DEVICE = torch.device("cpu")


def get_shard_range(num_lines: int, num_rank: int, rank: int) -> Tuple[int, int]:
    r"""Get the range of indices for the current rank in multi-processing.
    Args:
        num_lines (int): The number of lines to process.
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        rank (int): The rank in the multi-processing.

    Returns:
        (int, int):
        int: The start index for the current rank.
        int: The end index for the current rank.
    """
    assert 1 <= rank <= num_rank, f"invalid rank/num_rank {rank}/{num_rank}"
    assert num_lines > 0, f"Found {num_lines} files, make sure you specify the correct root directory"
    start = round(num_lines / num_rank * (rank - 1))
    end = round(num_lines / num_rank * rank)
    _LG.info(f"rank {rank} of {num_rank}, process {end-start} " f"({start}-{end}) out of {num_lines}")
    return start, end


def extract_feature_spec(
    path: str,
    device: torch.device,
    sample_rate: int,
    feature_extractor: Module,
) -> Tensor:
    r"""Extract MFCC features for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        sample_rate (int): The sample rate of the audio.

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform, sr = torchaudio.load(path)
    assert sr == sample_rate
    waveform = waveform[0].to(device)
    mfccs = feature_extractor(waveform)  # (freq, time)
    deltas = torchaudio.functional.compute_deltas(mfccs)
    ddeltas = torchaudio.functional.compute_deltas(deltas)
    concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
    feat = concat.transpose(0, 1)  # (time, freq)
    return feat


def extract_feature_hubert(
    path: str,
    device: torch.device,
    sample_rate: int,
    model: Module,
    layer_index: int,
) -> Tensor:
    r"""Extract HuBERT features for KMeans clustering and pseudo label prediction.
    Args:
        path (str): The file path of the audio.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        sample_rate (int): The sample rate of the audio.
        model (Module): The loaded ``HuBERTPretrainModel`` model.
        layer_index (int): The index of transformer layers in
            ``torchaudio.models.HuBERTPretrainModel`` for extracting features.
            (``1`` means the first layer output).

    Returns:
        Tensor: The desired feature tensor of the given audio file.
    """
    waveform, sr = torchaudio.load(path)
    assert sr == sample_rate
    waveform = waveform.to(device)
    if hasattr(model, "extract_features"):
        ...
    else:
        model = model.wav2vec2
    with torch.inference_mode():
        feat = model.extract_features(waveform, num_layers=layer_index)[0][-1][0]  # (time, feat_dim)
    return feat


def _load_state(model: Module, checkpoint_path: Path, device=_DEFAULT_DEVICE) -> Module:
    """Load weights from HuBERTPretrainModel checkpoint into hubert_pretrain_base model.
    Args:
        model (Module): The hubert_pretrain_base model.
        checkpoint_path (Path): The model checkpoint.
        device (torch.device, optional): The device of the model. (Default: ``torch.device("cpu")``)

    Returns:
        (Module): The pretrained model.
    """
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model


def dump_features(
    tsv_file: Union[str, Path],
    out_dir: Union[str, Path],
    split: str,
    rank: int,
    num_rank: int,
    device: torch.device,
    feature_type: str = "mfcc",
    layer_index: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    sample_rate: int = 16_000,
    num_classes: int = 100,
    widen_feature_extractor: Optional[int] = None,
) -> None:
    r"""Dump the feature tensors given a ``.tsv`` file list. The feature and lengths tensors
        will be stored under ``out_dir`` directory.
    Args:
        tsv_file (str or Path): The path of the tsv file.
        out_dir (str or Path): The directory to store the feature tensors.
        split (str): The split of data. Options: [``train``, ``valid``].
        rank (int): The rank in the multi-processing.
        num_rank (int): The number of ranks for multi-processing in feature extraction.
        device (torch.device): The location to allocate for PyTorch Tensors.
            Options: [``torch.device('cpu')``, torch.device('cuda')``].
        feature_type (str, optional): The type of the desired feature. Options: [``mfcc``, ``hubert``].
            (Default: ``mfcc``)
        layer_index (int or None, optional): The index of transformer layers in
            ``torchaudio.models.HuBERTPretrainModel`` for extracting features.
            (``1`` means the first layer output). Only active when ``feature_type``
            is set to ``hubert``. (Default: ``None``)
        checkpoint_path(Path or None, optional): The checkpoint path of ``torchaudio.models.HuBERTPretrainModel``.
            Only active when ``feature_type`` is set to ``hubert``. (Default: ``None``)
        sample_rate (int, optional): The sample rate of the audio. (Default: ``16000``)

    Returns:
        None
    """
    if feature_type not in ["mfcc", "hubert", "lfcc", "lfcc_wide"]:
        raise ValueError(f"Expected feature type to be 'mfcc' or 'hubert'. Found {feature_type}.")
    if feature_type == "hubert" and layer_index is None:
        assert ValueError("Please set the layer_index for HuBERT feature.")
    features = []
    lens = []
    out_dir = Path(out_dir)

    feat_path, len_path = _get_feat_lens_paths(out_dir, split, rank, num_rank)

    if hasattr(torchaudio.pipelines, str(checkpoint_path)):
        _LG.info("using pretrained model")
        assert feature_type == "hubert"
        bundle = getattr(torchaudio.pipelines, str(checkpoint_path))
        model = bundle.get_model()
        model = model.to(device)

    elif feature_type == "hubert":
        from torchaudio.models import hubert_pretrain_base
        from lightning_modules import _resample_feature_extractor, _widen_feature_extractor

        model = hubert_pretrain_base(num_classes=num_classes)
        if sample_rate != 16_000:
            _LG.info(f"Resampling feature extractor 16000 -> {sample_rate}")
            model.wav2vec2.feature_extractor = _resample_feature_extractor(
                model.wav2vec2.feature_extractor, 16_000, sample_rate
            )
        if widen_feature_extractor:
            _LG.info(f"Widening feature extractor by x{widen_feature_extractor}")
            model.wav2vec2.feature_extractor = _widen_feature_extractor(
                model.wav2vec2.feature_extractor, widen_feature_extractor
            )
        model.to(device)
        model = _load_state(model, checkpoint_path, device)

    elif feature_type == "mfcc":
        n_fft = int(400 / 16_000 * sample_rate)
        hop_length = int(160 / 16_000 * sample_rate)
        feature_extractor = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=13, melkwargs={"n_fft": n_fft, "hop_length": hop_length, "center": False}
        ).to(device)
    elif feature_type == "lfcc":
        n_fft = int(400 / 16_000 * sample_rate)
        hop_length = int(160 / 16_000 * sample_rate)
        n_lfcc = min(13, n_fft)
        # Scale the number of filters log. with window size
        n_filters = int(min(n_lfcc, 128 // (math.log2(400) - math.log2(n_fft) + 1)))
        feature_extractor = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            n_filter=n_filters,
            speckwargs={"n_fft": n_fft, "hop_length": hop_length, "center": False},
        ).to(device)
    elif feature_type == "lfcc_wide":
        assert sample_rate <= 16_000, "Not supported"
        n_fft = int(400 * sample_rate // 16_000)
        hop_length = int(160 * sample_rate // 16_000)

        if widen_feature_extractor:
            # Hop length of W2v2 model is 320 not 160 so need to compensate for that
            n_fft = n_fft + (widen_feature_extractor - 1) * hop_length * 2
            warnings.warn(
                f"Not compatible with standard HuBERT model, will require expanding effective input window ('train.py ... --expand-feature-extractor {widen_feature_extractor}')"
            )
            _LG.info(f"Widen feature window is now: {n_fft}")

        n_lfcc = 13
        # Scale the number of filters log. with window size
        n_filters = int(min(n_lfcc, 128 // max(1, math.log2(400) - math.log2(n_fft) + 1)))

        _LG.info(f"n_lfcc={n_lfcc}, n_filter={n_filters}, n_fft={n_fft}, hop_length={hop_length}")

        feature_extractor = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            n_filter=n_filters,
            speckwargs={"n_fft": n_fft, "hop_length": hop_length, "center": False},
        ).to(device)
    else:
        raise ValueError("Unknown feature type")

    _LG.info("Extracting features with:")
    _LG.info(feature_extractor)

    with open(tsv_file, "r") as f:
        root = f.readline().rstrip()
        lines = [line.rstrip() for line in f]
        start, end = get_shard_range(len(lines), num_rank, rank)
        lines = lines[start:end]
        for line in lines:
            path, nsample = line.split("\t")
            path = f"{root}/{path}"
            nsample = int(nsample)
            if feature_type == "hubert":
                feature = extract_feature_hubert(path, device, sample_rate, model, layer_index)
            else:
                feature = extract_feature_spec(path, device, sample_rate, feature_extractor)
            # Skip empty feature
            if feature.numel() == 0:
                _LG.warning("skipping empty feature: %s", line)
                continue
            features.append(feature.cpu())
            lens.append(feature.shape[0])
    features = torch.cat(features)
    lens = torch.Tensor(lens)
    torch.save(features, feat_path)
    torch.save(lens, len_path)
    _LG.info(f"Finished dumping features for rank {rank} of {num_rank} successfully")
