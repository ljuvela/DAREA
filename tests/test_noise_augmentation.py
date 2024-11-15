import torch

from darea.augmentation.noise import NoiseAugmentation
from darea.datasets.musan import Musan_Dataset

def test_noise_augmentation():
    
    segment_len = 16000
    sample_rate = 16000
    batch_size = 32

    dataset = Musan_Dataset(sampling_rate=sample_rate, segment_size=segment_len, partition='train')

    noise_augmentation = NoiseAugmentation(
        dataset=dataset, batch_size=batch_size, num_workers=0,
        min_snr=20, max_snr=35.0)
    
    waveform = torch.randn(batch_size, 1, segment_len)

    noisy_waveform = noise_augmentation(waveform)

    noise = noisy_waveform - waveform

    assert noisy_waveform.size() == waveform.size()

    noise_rms = noise.pow(2).mean(dim=-1).sqrt()
    waveform_rms = waveform.pow(2).mean(dim=-1).sqrt()

    snr_estimated = 20 * torch.log10(waveform_rms / noise_rms)

    assert snr_estimated.max() < 35.0
    assert snr_estimated.min() > 20.0


def test_noise_augmentation_long():
    
    segment_len = 160000
    sample_rate = 16000
    batch_size = 4

    dataset = Musan_Dataset(sampling_rate=sample_rate, segment_size=segment_len, partition='train')

    noise_augmentation = NoiseAugmentation(
        dataset=dataset, batch_size=batch_size, num_workers=0,
        min_snr=20, max_snr=35.0)
    
    waveform = torch.randn(batch_size, 1, segment_len)

    noisy_waveform = noise_augmentation(waveform)

    noise = noisy_waveform - waveform

    assert noisy_waveform.size() == waveform.size()

    noise_rms = noise.pow(2).mean(dim=-1).sqrt()
    waveform_rms = waveform.pow(2).mean(dim=-1).sqrt()

    snr_estimated = 20 * torch.log10(waveform_rms / noise_rms)

    assert snr_estimated.max() < 35.0
    assert snr_estimated.min() > 20.0


