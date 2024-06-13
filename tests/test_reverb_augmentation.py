import pytest
import torch

from darea.datasets.mit_rir import MIT_RIR_Dataset
from darea.augmentation.room_impulse import ConvolutionReverbAugment


def test_convolution_reverb_augment():

    dataset = MIT_RIR_Dataset(sampling_rate=16000, segment_size=16000, partition='train', resample=True, shuffle=False)
    reverb = ConvolutionReverbAugment(dataset)
    batch_size = 2
    channels = 1
    timesteps = 16000
    audio = torch.randn(batch_size, channels, timesteps)
    
    audio_reverbed = reverb(audio)
    assert audio_reverbed.shape == audio.shape
