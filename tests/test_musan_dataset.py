import pytest
import torch
from darea.datasets.musan import Musan_Dataset

def test_musan_download():
    Musan_Dataset(download=False).download()
    assert True

def test_musan_dataset_wrong_samplerate():
    with pytest.raises(ValueError):
        dataset = Musan_Dataset(sampling_rate=44100, segment_size=16000, partition='train', resample=False)
        audio, filepath = dataset[0]

def test_musan_dataset():

    dataset = Musan_Dataset(sampling_rate=16000, segment_size=16000, partition='train', resample=True)
    assert len(dataset) > 0
    audio, filepath = dataset[0]
    assert audio.shape[-1] == 16000