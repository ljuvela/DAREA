import pytest
import torch
from darea.augmentation.codecs import CodecAugmentation

formats = ['wav', 'mp3', 'mp3-32', 'mp3-8', 'ogg']

@pytest.mark.parametrize('format', formats)
def test_codecs_forward(format):
    
    codec = CodecAugmentation(format=format)

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = codec(x)

    assert y.shape == x.shape

@pytest.mark.parametrize('format', formats)
def test_codecs_gradient_pass(format):

    codec = CodecAugmentation(format=format)
    codec.train()

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    x = torch.nn.Parameter(x)

    y = codec(x)
    y.retain_grad()

    loss = y.sum()
    loss.backward()

    assert x.grad is not None

    assert torch.allclose(x.grad, y.grad)





    pass
