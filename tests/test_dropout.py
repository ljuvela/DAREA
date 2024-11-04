import torch
import pytest

from darea.augmentation.dropout import SampleDropoutAugmentation
from darea.augmentation.dropout import StftDropoutAugmentation

from darea.augmentation.container import AugmentationContainerKeywords

def test_sample_dropout():
    
    dropout = SampleDropoutAugmentation()
    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = dropout(x)

    assert y.shape == x.shape

    dropout = SampleDropoutAugmentation(p=0.1)
    y = dropout(x)
    assert y.shape == x.shape

def test_stft_dropout():
    
    dropout = StftDropoutAugmentation()
    batch = 2
    channels = 1
    samples = 16000
    x = torch.randn(batch, channels, samples)
    y = dropout(x)

    assert y.shape == x.shape

    dropout = StftDropoutAugmentation(p=0.1)
    y = dropout(x)
    assert y.shape == x.shape

@pytest.mark.parametrize('keyword', ['sample_dropout', 'stft_dropout'])
def test_augmentation_container_dropout(keyword):

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    container = AugmentationContainerKeywords(
        augmentations=[keyword],
        sample_rate=22050,
        num_random_choose=1).to(device)

    x = torch.randn(1, 1, 16000).to(device)
    y = container(x)
    assert y.size() == x.size()