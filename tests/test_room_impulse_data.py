import os
from darea.utils import get_data_path

from darea.room_impulse import MIT_RIR_Dataset

def test_get_data_path():

    path = get_data_path()
    assert os.path.exists(path)

def test_mit_rir_download():

    data_dir = get_data_path()
    MIT_RIR_Dataset().download(data_dir)


def test_mit_rir_dataset():

    dataset = MIT_RIR_Dataset(sampling_rate=16000, segment_size=16000, partition='train')

