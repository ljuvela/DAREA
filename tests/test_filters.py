import torch
from darea.augmentation.filters import LowPassFilterAugmentation
from darea.augmentation.filters import HighPassFilterAugmentation
from darea.augmentation.filters import AllPassFilterAugmentation


def test_low_pass():

    low_pass = LowPassFilterAugmentation()
    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = low_pass(x)

    assert y.shape == x.shape

    low_pass = LowPassFilterAugmentation(
        cutoff_freq_min=2000,
        cutoff_freq_max=4000)
    y = low_pass(x)
    assert y.shape == x.shape


def test_high_pass():

    high_pass = HighPassFilterAugmentation()
    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = high_pass(x)

    assert y.shape == x.shape

    high_pass = HighPassFilterAugmentation(
        cutoff_freq_min=50,
        cutoff_freq_max=1000)
    y = high_pass(x)
    assert y.shape == x.shape


def test_all_pass():

    all_pass = AllPassFilterAugmentation()
    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = all_pass(x)

    assert y.shape == x.shape

    all_pass = AllPassFilterAugmentation(
        central_freq=2000,
        q=0.707)
    y = all_pass(x)
    assert y.shape == x.shape