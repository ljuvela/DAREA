import torch

from darea.augmentation.pitch_shift import PitchShiftAugmentation

def test_pitch_shift_forward():

    pitch_shift = PitchShiftAugmentation()

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = pitch_shift(x)

    assert y.shape == x.shape

def test_pitch_shift_gradient_pass():

    pitch_shift = PitchShiftAugmentation()
    pitch_shift.train()

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    x = torch.nn.Parameter(x)

    y = pitch_shift(x)
    y.retain_grad()

    loss = y.sum()
    loss.backward()

    assert x.grad is not None

    assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
