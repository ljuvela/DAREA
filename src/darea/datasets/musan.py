import torch
from importlib.resources import files


import wget
import tarfile 
import os

import torchaudio

from ..utils import get_data_path
from ..audiodataset import AudioDataset

class Musan_Dataset(AudioDataset):

    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    filelists = {
        'train': files('darea.filelists.musan').joinpath('train_list.txt'),
        'val': files('darea.filelists.musan').joinpath('valid_list.txt'),
        'test': files('darea.filelists.musan').joinpath('test_list.txt'),
    }

    def __init__(self, sampling_rate=16000, segment_size=None, split=True,
                 shuffle=True, n_cache_reuse=1, resample=False, device=None,
                 partition='train', download=False):

        self.data_path = os.path.join(get_data_path(), "musan")

        if partition == 'train':
            filelist = Musan_Dataset.filelists['train']
        elif partition == 'val':
            filelist = Musan_Dataset.filelists['val']
        elif partition == 'test':
            filelist = Musan_Dataset.filelists['test']
        else:
            raise ValueError(
                f"Invalid partition {partition}. Valid partitions are 'train', 'val', 'test'")

        files_text = filelist.read_text()
        files = files_text.split('\n')
        files_full_path = [os.path.join(
            self.data_path, "musan", "noise", "free-sound", f) for f in files]

        super().__init__(files_full_path, sampling_rate, segment_size,
                         split, shuffle, n_cache_reuse, resample, device,
                         padding_mode="repeat")

        files_found = self.check_files()
        if not files_found:
            if download:
                self.download()
            else:
                raise FileNotFoundError(
                    f"Data not found at {self.data_path}!"
                    f"Please set the download flag to True to download it from {Musan_Dataset.url}")

    def __getitem__(self, index):
        return super().__getitem__(index)

    def check_files(self):

        data_path = self.data_path
        audio_path = os.path.join(data_path, "musan", "noise", "free-sound")

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
        tar_file = os.path.join(data_path, "musan.tar.gz")
        if os.path.exists(tar_file):
            print(f"Data tar file already present at {tar_file}")
        else:
            print(
                f"Data not found at {tar_file}! Will download it from {Musan_Dataset.url}")
            wget.download(url=Musan_Dataset.url, out=tar_file)

        # Extract the zip archive
        print(f"Extracting data from {tar_file} to {data_path}")
        with tarfile.open(tar_file, 'r') as tar_ref:
            tar_ref.extractall(data_path)
            # only extract the noise folder
            # tar_ref.extract('noise', data_path)

        return
