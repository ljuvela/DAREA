import os
from pathlib import Path
import wget


def get_data_path():

    path = os.environ.get('DAREA_DATA_PATH', None)
    if path is None:
        raise RuntimeError("Environment variable DAREA_DATA_PATH is not set! "
                           "Please set it to the path where the data should be stored ")

    path = os.path.realpath(path)
    return path
