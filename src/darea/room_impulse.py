import torch
from importlib.resources import files

from .utils import get_data_path

import wget
import zipfile
import os

from .audiodataset import AudioDataset


class MIT_RIR_Dataset(AudioDataset):

    # url = "https://www.openslr.org/resources/28/rirs_noises.zip"
    url = "https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip"
    filelists = {
        'train': files('darea.filelists.mit_rir').joinpath('train_list.txt'),
        'val': files('darea.filelists.mit_rir').joinpath('valid_list.txt'),
        'test': files('darea.filelists.mit_rir').joinpath('test_list.txt'),
    }
    data_path = os.path.join(get_data_path(), "mit_rir")

    def __init__(self, sampling_rate=16000, segment_size=None, split=True, shuffle=True, n_cache_reuse=1, resample=False, device=None,
                 partition='train'):

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

        super().__init__(files, sampling_rate, segment_size,
                         split, shuffle, n_cache_reuse, resample, device)

    def __getitem__(self, index):
        return super().__getitem__(index)

    def download(self, data_dir=None):
        if data_dir is None:
            data_dir = get_data_path()

        data_path = MIT_RIR_Dataset.data_path

        # check if audio files are already present
        audio_path = os.path.join(data_path, "Audio")
        if os.path.exists(audio_path):
            # check for each file in partition
            missing_files = False
            for file in self.audio_files:
                filepath = os.path.join(audio_path, file)
                if not os.path.exists(filepath):
                    print(
                        f"Data not found at {audio_path}! Will download it from {MIT_RIR_Dataset.url}")
                    missing_files = True
                    break

            if not missing_files:
                print(f"Data already present at {audio_path}")
                return

        os.makedirs(data_path, exist_ok=True)
        zip_file = os.path.join(data_path, "Audio.zip")

        if os.path.exists(zip_file):
            print(f"Data zip file already present at {zip_file}")
        else:
            print(
                f"Data not found at {zip_file}! Will download it from {MIT_RIR_Dataset.url}")
            wget.download(url=MIT_RIR_Dataset.url, out=zip_file)

        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)

        return
