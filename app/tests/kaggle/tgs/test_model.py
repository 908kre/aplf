from aplf.kaggle.tgs.model import UNet, SEBlock, DownSample, UpSample, ResBlock
import torch
import pytest

@pytest.mark.parametrize("depth, feature_size", [
    (3, 4),
    (3, 8),
    (4, 8),
    (3, 16),
])
def test_model(depth, feature_size):
    with torch.no_grad():
        model = UNet(
            feature_size=feature_size,
            depth=depth
        )
        in_image = torch.empty(32, 1, 101, 101)
        out_image = model(in_image)
        assert out_image.size() == (32, 2, 101, 101)


def test_se_layer():
    model = SEBlock(32, 2)
    in_image = torch.empty(32, 32, 101, 101)
    output = model(in_image)
    assert output.size() == (32, 32, 101, 101)


def test_res_block():
    model = ResBlock(32, 16)
    in_image = torch.empty(32, 32, 101, 101)
    output = model(in_image)
    assert output.size() == (32, 16, 101, 101)


def test_downsample():
    model = DownSample(1, 32, kernel_size=3)
    in_image = torch.empty(32, 1, 101, 101)
    pooled, conved = model(in_image)
    assert pooled.size() == (32, 32, 50, 50)
    assert conved.size() == (32, 32, 101, 101)


def test_upsample():
    model = UpSample(16, 32, 32)
    in_image = torch.empty(32, 16, 48, 48)
    bypass_image = torch.empty(32, 32, 96, 96)
    output = model(in_image, bypass_image)
    assert output.size() == (32, 32, 96, 96)
