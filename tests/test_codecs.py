import pytest
import torch
from darea.augmentation.codecs import CodecAugmentation

formats = ['mp3', 'ogg-vorbis', 'ogg-opus']

bitrates = [8000, 16000, 32000, 92000]

@pytest.mark.parametrize('bitrate', bitrates)
@pytest.mark.parametrize('format', formats)
def test_codecs_forward(bitrate, format):
    
    codec = CodecAugmentation(format=format, bitrate=bitrate)

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = codec(x)

    assert y.shape == x.shape


@pytest.mark.parametrize('bitrate', bitrates)
@pytest.mark.parametrize('format', formats)
def test_codecs_gradient_pass(bitrate, format):

    codec = CodecAugmentation(format=format, bitrate=bitrate)
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

def test_codec_gradient_pass_normalized():

    codec = CodecAugmentation(format='ogg-vorbis', bitrate=32000, grad_clip_norm_level=1.0)
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

    assert torch.norm(x.grad, dim=-1).max() > 1.0


def test_g723_1_forward():
    
    format = 'g723_1'
    codec = CodecAugmentation(format=format)

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = codec(x)
    assert y.shape == x.shape

# def test_g726_forward():
        
#         format = 'g726'
#         codec = CodecAugmentation(format=format)
    
#         batch = 2
#         channels = 1
#         samples = 16000
#         x = torch.randn(batch, channels, samples)
#         y = codec(x)
#         assert y.shape == x.shape

def test_codecs_cuda():

    if not torch.cuda.is_available():
        return

    device = torch.device('cuda')

    codec = CodecAugmentation(format='ogg-vorbis').to(device)
    codec.train()

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples).to(device)
    x = torch.nn.Parameter(x)

    y = codec(x)
    y.retain_grad()

    loss = y.sum()
    loss.backward()

    assert x.grad is not None

    assert torch.allclose(x.grad, y.grad)