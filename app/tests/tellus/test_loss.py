from aplf.tellus.losses import msssim, SSIM
from pathlib import Path
from aplf.tellus.data import load_train_df, get_train_row, TellusDataset, kfold, ChunkSampler, get_test_row, load_test_df, Augment, batch_aug
from torch.utils.data import DataLoader
import numpy as np
import cv2
import pandas as pd
import pytest
from urllib.request import urlopen
from cytoolz.curried import keymap, filter, pipe, merge, map, compose, concatv, first, take, concat
from torch.utils.data import Subset
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from aplf import config
import torch
from torch.autograd import Variable


def test_ssim():

    output = load_train_df(
        dataset_dir='/store/tellus/train',
        output='/store/tmp/train.pqt'
    )
    df = pd.read_parquet(output)
    dataset = TellusDataset(
        df=df,
        has_y=True,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )
    sample = pipe(
        loader,
        first,
    )
    print(sample['palsar'].shape)
    print(msssim(sample['palsar'], sample['palsar'] / 2))
    ssim_loss = SSIM(window_size=11)
    print(ssim_loss(img1, img2))
