from collections import OrderedDict
import os
from pathlib import Path
import shutil

from imageio.v3 import imread, imwrite
from PIL import Image
import pysaliency
from pysaliency.baseline_utils import BaselineModel, CrossvalidatedBaselineModel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import model_zoo

from tqdm import tqdm

from deepgaze_pytorch.layers import (
    Conv2dMultiInput,
    LayerNorm,
    LayerNormMultiInput,
    Bias,
    FlexibleScanpathHistoryEncoding
)

from deepgaze_pytorch.modules import DeepGazeIII, FeatureExtractor
from deepgaze_pytorch.features.densenet import RGBDenseNet201
from deepgaze_pytorch.data import ImageDataset, ImageDatasetSampler, FixationDataset, FixationMaskTransform
from deepgaze_pytorch.training import _train

def build_saliency_network(input_channels):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNorm(input_channels)),
        ('conv0', nn.Conv2d(input_channels, 8, (1, 1), bias=False)),
        ('bias0', Bias(8)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(8)),
        ('conv1', nn.Conv2d(8, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('layernorm2', LayerNorm(16)),
        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
        ('bias2', Bias(1)),
        ('softplus2', nn.Softplus()),
    ]))


def build_scanpath_network():
    return nn.Sequential(OrderedDict([
        ('encoding0', FlexibleScanpathHistoryEncoding(in_fixations=4, channels_per_fixation=3, out_channels=128, kernel_size=[1, 1], bias=True)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),
    ]))

def build_fixation_selection_network(scanpath_features=16):
    return nn.Sequential(OrderedDict([
        ('layernorm0', LayerNormMultiInput([1, scanpath_features])),
        ('conv0', Conv2dMultiInput([1, scanpath_features], 128, (1, 1), bias=False)),
        ('bias0', Bias(128)),
        ('softplus0', nn.Softplus()),

        ('layernorm1', LayerNorm(128)),
        ('conv1', nn.Conv2d(128, 16, (1, 1), bias=False)),
        ('bias1', Bias(16)),
        ('softplus1', nn.Softplus()),

        ('conv2', nn.Conv2d(16, 1, (1, 1), bias=False)),
    ]))

def prepare_spatial_dataset(stimuli, fixations, centerbias, batch_size, path=None):
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        lmdb_path = str(path)
    else:
        lmdb_path = None

    dataset = ImageDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        transform=FixationMaskTransform(sparse=False),
        average='image',
        lmdb_path=lmdb_path,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=False,
        num_workers=0,
    )

    return loader


def prepare_scanpath_dataset(stimuli, fixations, centerbias, batch_size, path=None):
    if path is not None:
        path.mkdir(parents=True, exist_ok=True)
        lmdb_path = str(path)
    else:
        lmdb_path = None

    dataset = FixationDataset(
        stimuli=stimuli,
        fixations=fixations,
        centerbias_model=centerbias,
        included_fixations=[-1, -2, -3, -4],
        allow_missing_fixations=True,
        transform=FixationMaskTransform(sparse=False),
        average='image',
        lmdb_path=lmdb_path,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=ImageDatasetSampler(dataset, batch_size=batch_size),
        pin_memory=False,
        num_workers=0,
    )

    return loader

device = 'cuda'

for crossval_fold in range(10):
    MIT1003_stimuli_train, MIT1003_fixations_train = pysaliency.dataset_config.train_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=crossval_fold)
    MIT1003_stimuli_val, MIT1003_fixations_val = pysaliency.dataset_config.validation_split(mit_stimuli_twosize, mit_fixations_twosize, crossval_folds=10, fold_no=crossval_fold)

    train_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_train, MIT1003_fixations_train, verbose=True, average='image')
    val_baseline_log_likelihood = MIT1003_centerbias.information_gain(MIT1003_stimuli_val, MIT1003_fixations_val, verbose=True, average='image')

    # finetune spatial model on MIT1003

    model = DeepGazeIII(
        features=FeatureExtractor(RGBDenseNet201(), [
                '1.features.denseblock4.denselayer32.norm1',
                '1.features.denseblock4.denselayer32.conv1',
                '1.features.denseblock4.denselayer31.conv2',
            ]),
        saliency_network=build_saliency_network(2048),
        scanpath_network=None,
        fixation_selection_network=build_fixation_selection_network(scanpath_features=0),
        downsample=2,
        readout_factor=4,
        saliency_map_factor=4,
        included_fixations=[],
    )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18, 21, 24])

    train_loader = prepare_spatial_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, batch_size=4, path=train_directory / 'lmdb_cache' / f'MIT1003_train_spatial_{crossval_fold}')
    validation_loader = prepare_spatial_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, batch_size=4, path=train_directory / 'lmdb_cache' / f'MIT1003_val_spatial_{crossval_fold}')

    _train(train_directory / 'MIT1003_spatial' / f'crossval-10-{crossval_fold}',
        model,
        train_loader, train_baseline_log_likelihood,
        validation_loader, val_baseline_log_likelihood,
        optimizer, lr_scheduler,
        minimum_learning_rate=1e-7,
        device=device,
        startwith=train_directory / 'pretraining' / 'final.pth',
    )


    # Train scanpath model

    train_loader = prepare_scanpath_dataset(MIT1003_stimuli_train, MIT1003_fixations_train, MIT1003_centerbias, batch_size=4, path=train_directory / 'lmdb_cache' / f'MIT1003_train_scanpath_{crossval_fold}')
    validation_loader = prepare_scanpath_dataset(MIT1003_stimuli_val, MIT1003_fixations_val, MIT1003_centerbias, batch_size=4, path=train_directory / 'lmdb_cache' / f'MIT1003_val_scanpath_{crossval_fold}')

    # first train with partially frozen saliency network


    model = DeepGazeIII(
        features=FeatureExtractor(RGBDenseNet201(), [
                '1.features.denseblock4.denselayer32.norm1',
                '1.features.denseblock4.denselayer32.conv1',
                '1.features.denseblock4.denselayer31.conv2',
            ]),
        saliency_network=build_saliency_network(2048),
        scanpath_network=build_scanpath_network(),
        fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
        downsample=2,
        readout_factor=4,
        saliency_map_factor=4,
        included_fixations=[-1, -2, -3, -4],
    )
    model = model.to(device)

    frozen_scopes = [
        "saliency_network.layernorm0",
        "saliency_network.conv0",
        "saliency_network.bias0",
        "saliency_network.layernorm1",
        "saliency_network.conv1",
        "saliency_network.bias1",
    ]

    for scope in frozen_scopes:
        for parameter_name, parameter in model.named_parameters():
            if parameter_name.startswith(scope):
                print("Fixating parameter", parameter_name)
                parameter.requires_grad = False


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 31, 32, 33, 34, 35])

    _train(train_directory / 'MIT1003_scanpath_partially_frozen_saliency_network' / f'crossval-10-{crossval_fold}',
        model,
        train_loader, train_baseline_log_likelihood,
        validation_loader, val_baseline_log_likelihood,
        optimizer, lr_scheduler,
        minimum_learning_rate=1e-7,
        device=device,
        startwith=train_directory / 'MIT1003_spatial' /  f'crossval-10-{crossval_fold}' / 'final.pth'
    )

    # Now finetune full scanpath model

    model = DeepGazeIII(
        features=FeatureExtractor(RGBDenseNet201(), [
                '1.features.denseblock4.denselayer32.norm1',
                '1.features.denseblock4.denselayer32.conv1',
                '1.features.denseblock4.denselayer31.conv2',
            ]),
        saliency_network=build_saliency_network(2048),
        scanpath_network=build_scanpath_network(),
        fixation_selection_network=build_fixation_selection_network(scanpath_features=16),
        downsample=2,
        readout_factor=4,
        saliency_map_factor=4,
        included_fixations=[-1, -2, -3, -4],
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18, 21, 24])

    _train(train_directory / 'MIT1003_scanpath' / f'crossval-10-{crossval_fold}',
        model,
        train_loader, train_baseline_log_likelihood,
        validation_loader, val_baseline_log_likelihood,
        optimizer, lr_scheduler,
        minimum_learning_rate=1e-7,
        device=device,
        startwith=train_directory / 'MIT1003_scanpath_partially_frozen_saliency_network' / f'crossval-10-{crossval_fold}' / 'final.pth'
    )