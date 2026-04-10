import torch

from darea.augmentation.neural_codecs import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_dac():

    sample_rate = 22050

    # Create a codec
    codec = DacAugmentation(sample_rate=sample_rate).to(device)

    # Create a random audio signal
    x = torch.randn(2, 1, 16000).to(device)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = codec(x)

    assert x_hat.size() == x.size()

    # test gradient
    loss = x_hat.mean()
    loss.backward()

    assert x.grad is not None



def test_encodec():

    sample_rate = 22050
    
    for bandwidth in [1.5, 3., 6, 12., 24.]:
        codec = EncodecAugmentation(sample_rate=sample_rate, bandwidth=bandwidth).to(device)
        # Create a random audio signal
        x = torch.randn(2, 1, 16000).to(device)
        x = torch.nn.Parameter(x, requires_grad=True)

        x_hat = codec(x)

        assert x_hat.size() == x.size()

        # test gradient
        loss = x_hat.mean()
        loss.backward()

        assert x.grad is not None


def test_mimi():
    sample_rate = 22050

    # Create a codec
    codec = MimiAugmentation(sample_rate=sample_rate).to(device)

    # Create a random audio signal
    x = torch.randn(2, 1, 16000).to(device)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = codec(x)

    assert x_hat.size() == x.size()

    loss = x_hat.mean()
    loss.backward()

    assert x.grad is not None


def test_st():
    sample_rate = 22050

    # Create a codec
    codec = SpeechTokenizerAugmentation(sample_rate=sample_rate).to(device)

    # Create a random audio signal
    x = torch.randn(2, 1, 16000).to(device)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = codec(x)

    assert x_hat.size() == x.size()

    loss = x_hat.mean()
    loss.backward()

    assert x.grad is not None


def test_snac():
    sample_rate = 22050

    # Create a codec
    codec = SnacAugmentation(sample_rate=sample_rate).to(device)

    # Create a random audio signal
    x = torch.randn(2, 1, 16000).to(device)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = codec(x)

    assert x_hat.size() == x.size()

    loss = x_hat.mean()
    loss.backward()

    assert x.grad is not None


def test_bc():
    sample_rate = 22050

    # Create a codec
    codec = BigCodecAugmentation(sample_rate=sample_rate)

    # Create a random audio signal
    x = torch.randn(2, 1, 16000)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = codec(x)

    assert x_hat.size() == x.size()

    loss = x_hat.mean()
    loss.backward()

    assert x.grad is not None
