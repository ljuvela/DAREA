import torch
import torchaudio
import os 

import darea
from darea.augmentation.container import AugmentationContainerKeywords


x, fs = torchaudio.load('1995_1836_000035_000001.wav')

output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)  


batch_size = 4
segment_size = 2 * fs

num_samples = x.size(1)

x = x.unsqueeze(0).repeat(batch_size, 1, 1)

augmentations = [
    'codec_dac_8kbps',
    'musan_noise',
    'mit_rir_reverb',
    'time_stretch',
    'sample_dropout',
    'stft_dropout',
    'lowpass_4kHz',
    'highpass_500Hz',
]

# save original audio
torchaudio.save(f"{output_dir}/original.wav", x[0], fs)

for aug in augmentations:
    container = AugmentationContainerKeywords(
        augmentations=[aug],
        sample_rate=fs,
        segment_size=num_samples,
        partition="train",
        batch_size=batch_size,
    )
    y = container(x)
    for i, y_i in enumerate(y):
        torchaudio.save(f"{output_dir}/{aug}_{i}.wav", y_i, fs)
