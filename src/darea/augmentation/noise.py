import torch
import torchaudio

from ..audiodataset import AudioDataset, get_dataset_filelist

class WhiteNoiseDataset(torch.utils.data.Dataset):

    def __init__(self, num_samples=1, segment_len=16000, sample_rate=16000):
        self.segment_len = segment_len
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.randn(self.segment_len).unsqueeze(0)



class NoiseAugmentation(torch.nn.Module):
    def __init__(self, dataset, batch_size, num_workers=0,
                 min_snr=20, max_snr=35.0):
        super().__init__()
        self.dataset = dataset
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.batch_size = batch_size

        if len(self.dataset) < batch_size:
            raise ValueError(f"Number of samples in the dataset must at least batch size. "
                             f"Got {len(self.dataset)} samples and batch size {batch_size}")

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True, shuffle=True)
        

    def _noise_generator(self):
        """
        Infinite generator that yields noise samples from the dataset.
        
        itertools cycle is not used because cycle caches the values

        """
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

        if waveform.dim() != 3:
            raise ValueError(f"Expected waveform to have 3 dimensions (batch, channels, time). Got {waveform.dim()}")
        batch_size, channels, timesteps = waveform.size()

        noise = next(self._noise_generator())
        noise = noise.to(waveform.device)
        noise = torch.nn.functional.pad(
            noise, pad=(waveform.size(-1)-noise.size(-1), 0),
            mode='constant', value=0.0).to(waveform.device)

        # sample snr from uniform distribution
        snr = torch.empty(batch_size, channels).uniform_(
            self.min_snr, self.max_snr).to(waveform.device)

        noisy_waveform = torchaudio.functional.add_noise(
            waveform=waveform, noise=noise, 
            snr=snr)
        
        return noisy_waveform