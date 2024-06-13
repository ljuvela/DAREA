import torch

import os

from darea.noise_augmentation import NoiseAugmentation
from darea.noise_augmentation import WhiteNoiseDataset

def test_noise_augmentation():
    
    segment_len = 16000
    sample_rate = 16000
    batch_size = 2

    dataset = WhiteNoiseDataset(
        num_samples=batch_size,
        segment_len=segment_len, sample_rate=sample_rate)

    noise_augmentation = NoiseAugmentation(
        dataset=dataset, batch_size=batch_size, num_workers=0,
        min_snr=-35, max_snr=-20.0)
    
    waveform = torch.randn(batch_size, 1, segment_len)

    noisy_waveform = noise_augmentation(waveform)

    noise = noisy_waveform - waveform

    assert noisy_waveform.size() == waveform.size()

    noise_energy = noise.pow(2).sum(dim=-1)


