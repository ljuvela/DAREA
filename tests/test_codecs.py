import pytest
import torch
import itertools
from darea.augmentation.codecs import CodecAugmentation


formats = ['mp3', 'ogg-vorbis', 'ogg-opus']

bitrates = [16000, 32000, 64000, 92000, 128000]

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



def test_pcm16_forward():
    
    bitrate = 16
    sample_rate = 16000
    codec = CodecAugmentation(format='pcm16', bitrate=16 * bitrate, sample_rate=sample_rate)

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


def test_vorbis_q_factors():

    format = 'ogg-vorbis'

    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)

    q_factors = [0, 1, 2, 3, 4]
    results = {}

    for q_factor in q_factors:
        codec = CodecAugmentation(format=format, q_factor=q_factor)
        y = codec(x)
        assert y.shape == x.shape
        results[q_factor] = y

    # Check that the outputs are different (all combinations)
    for i, j in itertools.combinations(q_factors, 2):
        assert not torch.allclose(results[i], results[j]), f"results for q_factor {i} and {j} are the same"


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
    codec = CodecAugmentation(format=format, bitrate=64000)

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
