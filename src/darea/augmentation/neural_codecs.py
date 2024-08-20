import torch
import dac
import os

from torchaudio.transforms import Resample


class NeuralCodecAugmentation(torch.nn.Module):

    def __init__(self, sample_rate=16000):
        super(NeuralCodecAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.dac_sample_rate = 24000

        model_path = dac.utils.download(model_type="24khz")
        if not os.path.isfile(model_path):
            print(f"Downloading model to {model_path}")
        self.model = dac.DAC.load(model_path)

        self.model.train()

        self.sample_rate = sample_rate
        if self.sample_rate != self.dac_sample_rate:
            self.resampler_to_dac = Resample(orig_freq=self.sample_rate, new_freq=self.dac_sample_rate)
            self.resampler_from_dac = Resample(orig_freq=self.dac_sample_rate, new_freq=self.sample_rate)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)

        if self.sample_rate != self.dac_sample_rate:
            x = self.resampler_to_dac(x)

        out = self.model.forward(x, sample_rate=self.dac_sample_rate)

        x = out["audio"]

        if self.sample_rate != self.dac_sample_rate:
            x = self.resampler_from_dac(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
    

