import torch
import torchaudio
import numpy as np


class PitchShiftAugmentation(torch.nn.Module):

    def __init__(self, sample_rate=16000, max_semitones=1.0, min_semitones=-1.0):
        super(PitchShiftAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.bins_per_octave = 12
        self.max_semitones = max_semitones
        self.min_semitones = min_semitones

    def forward(self, x):

        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D tensor")

        if x.size(1) != 1:
            raise ValueError(f"Expected mono audio, got {x.size(1)} channels")

        # same pitch shift for all samples in the batch
        semitones = np.random.uniform(self.min_semitones, self.max_semitones)
        y = torchaudio.functional.pitch_shift(x, self.sample_rate, semitones,
                                               bins_per_octave=self.bins_per_octave)
        
        return y
