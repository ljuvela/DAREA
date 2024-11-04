import torch
import torchaudio

class TimeStretchAugmentation(torch.nn.Module):

    def __init__(self, sample_rate=16000, min_rate=0.90, max_rate=1.10):
        super(TimeStretchAugmentation, self).__init__()

        self.sample_rate = sample_rate
        self.min_rate = min_rate
        self.max_rate = max_rate

    def forward(self, x):

        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D tensor")

        if x.size(1) != 1:
            raise ValueError(f"Expected mono audio, got {x.size(1)} channels")

        rate = torch.rand(1) * (self.max_rate - self.min_rate) + self.min_rate
        # stretch by resampling
        base_sr = torch.tensor(self.sample_rate).float()
        new_sr = base_sr // rate

        y = torchaudio.functional.resample(x, base_sr, new_sr)

        # pad or trim to the original length
        timesteps = x.size(-1)
        if y.size(-1) < timesteps:
            y = torch.nn.functional.pad(y, (0, timesteps - y.size(-1)))
        elif y.size(-1) > timesteps:
            y = y[..., :timesteps]

        return y