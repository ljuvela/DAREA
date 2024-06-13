import os
import pytest
import torch

from darea.utils import get_data_path

from darea.room_impulse import MIT_RIR_Dataset
from darea.room_impulse import ConvolutionReverbAugment

def test_get_data_path():
    path = get_data_path()
    assert os.path.exists(path)

def test_mit_rir_download():
    MIT_RIR_Dataset().download()
    assert True


def test_mit_rir_dataset_wrong_samplerate():
    with pytest.raises(ValueError):
        dataset = MIT_RIR_Dataset(sampling_rate=16000, segment_size=16000, partition='train', resample=False)
        audio, filepath = dataset[0]


def test_mit_rir_dataset():

    dataset = MIT_RIR_Dataset(sampling_rate=16000, segment_size=16000, partition='train', resample=True)
    assert len(dataset) > 0
    audio, filepath = dataset[0]
    assert audio.shape[-1] == 16000


def test_convolution_reverb_augment():

    dataset = MIT_RIR_Dataset(sampling_rate=16000, segment_size=16000, partition='train', resample=True, shuffle=False)
    reverb = ConvolutionReverbAugment(dataset)
    batch_size = 2
    channels = 1
    timesteps = 16000
    audio = torch.randn(batch_size, channels, timesteps)
    
    audio_reverbed = reverb(audio)
    assert audio_reverbed.shape == audio.shape
