from cleanunet.denoise import CleanUNet

from torchaudio.transforms import Resample
import os 
import json
import torch


class DenoisingAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        super(DenoisingAugmentation, self).__init__()
        self.sample_rate = sample_rate

        path = os.environ.get('CLEANUNET_PATH', None)
        config_path = os.path.join(path, 'configs/DNS-large-full.json')
        ckpt_path= os.path.join(path, 'exp/DNS-large-full/checkpoint/pretrained.pkl')

        with open(config_path) as f:
            data = f.read()

        config = json.loads(data)
        network_config = config["network_config"]
        self.model = CleanUNet(**network_config)
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.model.requires_grad_(False)

        self.model_sample_rate = 16_000 

        self.sample_rate = sample_rate
        if self.sample_rate != self.model_sample_rate:
            self.resampler_to_model = Resample(orig_freq=self.sample_rate, new_freq=self.model_sample_rate)
            self.resampler_from_model = Resample(orig_freq=self.model_sample_rate, new_freq=self.sample_rate)


    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)

        if self.sample_rate != self.model_sample_rate:
            x = self.resampler_to_model(x)
        
        x = self.model(x)

        if self.sample_rate != self.model_sample_rate:
            x = self.resampler_from_model(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x




