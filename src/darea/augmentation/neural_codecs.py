import torch
import dac
import os
from encodec import EncodecModel
from torchaudio.transforms import Resample

class DacAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        super(DacAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.dac_sample_rate = 24000

        model_path = dac.utils.download(model_type="24khz")
        if not os.path.isfile(model_path):
            print(f"Downloading model to {model_path}")
        self.model = dac.DAC.load(model_path)

        self.model.eval()
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
    

class EncodecAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000, bandwidth=None):
         # bandwidths [1.5, 3, 6, 12, 24] uses smallest by default
        super(EncodecAugmentation, self).__init__()
        self.sample_rate = sample_rate
        self.bandwidth=bandwidth

        self.encodec_sample_rate = 24_000
        
        self.model = EncodecModel.encodec_model_24khz()

        if bandwidth:
            self.model.set_target_bandwidth(bandwidth)
   
        self.model.eval()

        self.sample_rate = sample_rate
        if self.sample_rate != self.encodec_sample_rate:
            self.resampler_to_encodec = Resample(orig_freq=self.sample_rate, new_freq=self.encodec_sample_rate)
            self.resampler_from_encodec = Resample(orig_freq=self.encodec_sample_rate, new_freq=self.sample_rate)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)
        

        if self.sample_rate != self.encodec_sample_rate:
            x = self.resampler_to_encodec(x)
        
        audio_values = self.model(x)

        x = audio_values

        if self.sample_rate != self.encodec_sample_rate:
            x = self.resampler_from_encodec(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
    