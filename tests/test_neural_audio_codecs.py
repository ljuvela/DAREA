import torch

from darea.augmentation.neural_codecs import DacAugmentation
from darea.augmentation.neural_codecs import EncodecAugmentation
from darea.augmentation.neural_codecs import MimiAugmentation


def test_dac():

    sample_rate = 22050

    # Create a codec
    codec = DacAugmentation(sample_rate=sample_rate)

    # Create a random audio signal
    x = torch.randn(1, 1, 16000)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = codec(x)

    assert x_hat.size() == x.size()

    # test gradient
    loss = x_hat.mean()
    loss.backward()

    assert x.grad is not None



def test_encodec():

    sample_rate = 22050

    # Create a codec
    codec = EncodecAugmentation(sample_rate=sample_rate)

    # Create a random audio signal
    x = torch.randn(1, 1, 16000)
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
    codec = MimiAugmentation(sample_rate=sample_rate)

    # Create a random audio signal
    x = torch.randn(1, 1, 16000)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = codec(x)

    assert x_hat.size() == x.size()

    # test gradient
    loss = x_hat.mean()
    loss.backward()

    assert x.grad is not None


