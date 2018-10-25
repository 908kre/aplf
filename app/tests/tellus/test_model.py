from aplf.tellus.model import Net, DownSample, AE
import torch
import pytest


@pytest.mark.parametrize("depth, feature_size", [
    (3, 32),
])
def test_unet(depth, feature_size):
    with torch.no_grad():
        model = Net(
            feature_size=feature_size,
        )
        before = torch.empty(32, 1, 40, 40)
        after = torch.empty(32, 1, 40, 40)
        out_image, b_rgb, a_rgb = model(before, after)
        assert out_image.size() == (32, 2)
        assert b_rgb.size() == (32, 3, 4, 4)
        assert a_rgb.size() == (32, 3, 4, 4)


def test_down():
    with torch.no_grad():
        model = DownSample(
            in_size=10,
            out_ch=4,
        )
        image = torch.empty(32, 10, 40, 40)
        down_image, out_image = model(image)
        assert down_image.size() == (32, 4, 20, 20)
        assert out_image.size() == (32, 4, 40, 40)

def test_ae():
    with torch.no_grad():
        model = AE(
            in_size=(2, 40, 40),
            out_size=(2, 40, 40),
            center_out_size=(6, 4, 4),
            feature_size=8,
        )
        image = torch.empty(32, 2, 40, 40)
        pl, la = model(image)
        assert pl.size() == (32, 2, 40, 40)
        assert la.size() == (32, 6, 4, 4)
