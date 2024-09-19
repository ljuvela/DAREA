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

class MIT_RIR_Dataset(AudioDataset):

    # url = "https://www.openslr.org/resources/28/rirs_noises.zip"
    url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
    filelists = {
        'train': files('darea.filelists.mit_rir').joinpath('train_list.txt'),
        'val': files('darea.filelists.mit_rir').joinpath('valid_list.txt'),
        'test': files('darea.filelists.mit_rir').joinpath('test_list.txt'),
    }

    def __init__(self, sampling_rate=16000, segment_size=None, split=True,
                 shuffle=True, n_cache_reuse=1, resample=False, device=None,
                 partition='train', download=False):

        self.data_path = os.path.join(get_data_path(), "mit_rir")

        if partition == 'train':
            filelist = MIT_RIR_Dataset.filelists['train']
        elif partition == 'val':
            filelist = MIT_RIR_Dataset.filelists['val']
        elif partition == 'test':
            filelist = MIT_RIR_Dataset.filelists['test']
        else:
            raise ValueError(
                f"Invalid partition {partition}. Valid partitions are 'train', 'val', 'test'")

        files_text = filelist.read_text()
        files = files_text.split('\n')
        files_full_path = [os.path.join(
            self.data_path, "Audio", f) for f in files]

        super().__init__(files_full_path, sampling_rate, segment_size,
                         split, shuffle, n_cache_reuse, resample, device)

        files_found = self.check_files()
        if not files_found:
            if download:
                self.download()
            else:
                raise FileNotFoundError(
                    f"Data not found at {self.data_path}!"
                    f"Please set the download flag to True to download it from {MIT_RIR_Dataset.url}")

    def __getitem__(self, index):
        return super().__getitem__(index)

    def check_files(self):
        data_path = self.data_path
        audio_path = os.path.join(data_path, "Audio")

        # check if audio files are already present
        if not os.path.exists(audio_path):
            return False

        # check for each file in partition
        missing_files = False
        for filepath in self.audio_files:
            if not os.path.exists(filepath):
                missing_files = True
                break

        return not missing_files

    def download(self):

        data_path = self.data_path
        audio_path = os.path.join(data_path, "Audio")

        # check if the files already exist
        files_found = self.check_files()
        if files_found:
            print(f"Data already present at {audio_path}")
            return

        # Download the data zip archive
        os.makedirs(data_path, exist_ok=True)
        zip_file = os.path.join(data_path, "Audio.zip")
        if os.path.exists(zip_file):
            print(f"Data zip file already present at {zip_file}")
        else:
            print(
                f"Data not found at {zip_file}! Will download it from {MIT_RIR_Dataset.url}")
            wget.download(url=MIT_RIR_Dataset.url, out=zip_file)

        # Extract the zip archive
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)

        return
