import torch
import pytest

from darea.augmentation.time_stretch import TimeStretchAugmentation
from darea.augmentation.container import AugmentationContainerKeywords

def test_time_stretch():

    stretch = TimeStretchAugmentation(sample_rate=22050, min_rate=0.5, max_rate=2.0)
    batch = 2
    channels = 1
    samples = 16000

    x = torch.randn(batch, channels, samples)

    y = stretch(x)

    assert y.shape == x.shape

def test_time_stretch_up():

    stretch = TimeStretchAugmentation(sample_rate=22050, min_rate=1.0, max_rate=2.0)
    batch = 2
    channels = 1
    samples = 16000

    x = torch.randn(batch, channels, samples)

    y = stretch(x)

    assert y.shape == x.shape

def test_time_stretch_down():

    stretch = TimeStretchAugmentation(sample_rate=22050, min_rate=0.5, max_rate=1.0)
    batch = 2
    channels = 1
    samples = 16000

    x = torch.randn(batch, channels, samples)

    y = stretch(x)

    assert y.shape == x.shape

def test_time_stretch_grad():

    stretch = TimeStretchAugmentation(sample_rate=22050, min_rate=0.5, max_rate=2.0)

    stretch.train()

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)

    x = torch.nn.Parameter(x)
    y = stretch(x)
    y.retain_grad()

    loss = y.sum()
    loss.backward()

    assert x.grad is not None

@pytest.mark.parametrize('keyword', ['time_stretch'])
def test_container(keyword):

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    container = AugmentationContainerKeywords(
        augmentations=[keyword],
        sample_rate=22050)

    x = torch.randn(1, 1, 16000).to(device)
    y = container(x)
    assert y.size() == x.size()