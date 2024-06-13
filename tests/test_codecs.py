import pytest
import torch
from darea.augmentation.codecs import CodecAugmentation

def test_codecs_inferece():
    
    codec = CodecAugmentation(format='ogg')

    x = torch.randn(1, 1, 16000)
    y = codec(x)

    assert y.shape == x.shape

def test_codecs_training():

    pass
