import torch
import torch.nn.functional as F

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
        scale_factor = 1/rate
        y = F.interpolate(input=x, size=int(scale_factor*x.size(-1)), mode='linear')

        # negative pad values truncate the signal
        y = torch.nn.functional.pad(y, pad=(x.size(-1)-y.size(-1), 0), mode='constant', value=0.0)

        return y