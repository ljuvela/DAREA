import torch
import torchaudio

def check_cutoff_freq(cutoff_freq, sample_rate):

    if cutoff_freq > sample_rate / 2:
        raise ValueError(f"cutoff_freq must be less than or equal to sample_rate / 2, got {cutoff_freq} > {sample_rate / 2}")
    
    if cutoff_freq < 0:
        raise ValueError(f"cutoff_freq must be greater than 0, got {cutoff_freq}")

def sample_cutoff_freq(cutoff_freq_min, cutoff_freq_max):
    """ Sample a cutoff frequency from a uniform distribution in log space.
    Args: 
        cutoff_freq_min: float, minimum cutoff frequency
        cutoff_freq_max: float, maximum cutoff frequency

    """

    log_fmin = torch.log(torch.tensor(cutoff_freq_min))
    log_fmax = torch.log(torch.tensor(cutoff_freq_max))
    cutoff_log = torch.rand(1) * (log_fmax - log_fmin) + log_fmin
    cutoff = torch.exp(cutoff_log)

    return cutoff


class LowPassFilterAugmentation(torch.nn.Module):

    def __init__(self, sample_rate=16000, cutoff_freq_max=8000, cutoff_freq_min=4000):
        super(LowPassFilterAugmentation, self).__init__()
        self.sample_rate = sample_rate
        self.cutoff_freq_max = cutoff_freq_max
        self.cutoff_freq_min = cutoff_freq_min

        check_cutoff_freq(self.cutoff_freq_max, self.sample_rate)
        check_cutoff_freq(self.cutoff_freq_min, self.sample_rate)

    def forward(self, x):

        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D tensor")

        if x.size(1) != 1:
            raise ValueError(f"Expected mono audio, got {x.size(1)} channels")

        cutoff = sample_cutoff_freq(self.cutoff_freq_min, self.cutoff_freq_max)
        y = torchaudio.functional.lowpass_biquad(x, self.sample_rate, cutoff)

        return y
    
class HighPassFilterAugmentation(torch.nn.Module):

    def __init__(self, sample_rate=16000, cutoff_freq_max=1000, cutoff_freq_min=50):
        super(HighPassFilterAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.cutoff_freq_max = cutoff_freq_max
        self.cutoff_freq_min = cutoff_freq_min

        check_cutoff_freq(self.cutoff_freq_max, self.sample_rate)
        check_cutoff_freq(self.cutoff_freq_min, self.sample_rate)

    def forward(self, x):

        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D tensor")

        if x.size(1) != 1:
            raise ValueError(f"Expected mono audio, got {x.size(1)} channels")

        cutoff = sample_cutoff_freq(self.cutoff_freq_min, self.cutoff_freq_max)
        y = torchaudio.functional.highpass_biquad(x, self.sample_rate, cutoff_freq=cutoff)

        return y
    

class AllPassFilterAugmentation(torch.nn.Module):

    def __init__(self, sample_rate=16000, central_freq=4000, q=0.707):
        super(AllPassFilterAugmentation, self).__init__()
        self.sample_rate = sample_rate
        self.central_freq = central_freq
        self.q = q

        if self.central_freq > self.sample_rate / 2:
            raise ValueError(f"central_freq must be less than or equal to sample_rate / 2, got {self.central_freq} > {self.sample_rate / 2}")
        
        if self.central_freq < 0:
            raise ValueError(f"central_freq must be greater than 0, got {self.central_freq}")

        if self.q <= 0:
            raise ValueError(f"q must be greater than 0, got {self.q}")

    def forward(self, x):

        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D tensor")

        if x.size(1) != 1:
            raise ValueError(f"Expected mono audio, got {x.size(1)} channels")

        y = torchaudio.functional.allpass_biquad(x, self.sample_rate, self.central_freq, self.q)

        return y