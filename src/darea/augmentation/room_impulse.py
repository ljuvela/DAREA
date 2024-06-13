import torch
from importlib.resources import files

import wget
import zipfile
import os

import torchaudio

from ..utils import get_data_path
from ..audiodataset import AudioDataset

class ConvolutionReverbAugment(torch.nn.Module):

    def __init__(self, dataset, num_workers=0):
        super().__init__()

        self.dataset = dataset
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=1,
            num_workers=num_workers, drop_last=True, 
            )

    def _rir_generator(self):
        while True:
            for x, _ in iter(self.dataloader):
                yield x

    def forward(self, waveform):
        """
        Args:
            waveform, shape = (batch, channels=1, time)

        Returns:
            noisy_waveform, shape (batch, channels=1, time) 
        """

        batch_size, channels, timesteps = waveform.size()

        device = waveform.device
        rir = next(self._rir_generator())

        # align to maximum peak
        offset = 20
        start = torch.maximum(torch.argmax(rir.abs()) - offset, torch.zeros(1))
        stop = torch.minimum(torch.Tensor([rir.size(-1)]), start+timesteps)
        rir = rir[..., int(start):int(stop)]

        rir = rir.to(device)
        # pad from the right
        rir = torch.nn.functional.pad(rir, pad=(0, waveform.size(-1)-rir.size(-1)), mode='constant', value=0.0)
        # normalize
        rir = rir / (rir.norm(2) + 1e-6)
    
        y = torchaudio.functional.fftconvolve(waveform, rir, mode='full')
        return y[..., :timesteps]