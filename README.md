# DAREA
Differentiable augmentation and robustness evaluation for audio


## Setup

Noise and room impulse response samples are stored in `DAREA_DATA_PATH`. The user should set this environment variable to the path where the data is stored. For example, in bash:
```shell
export DAREA_DATA_PATH="<path-to-your-data-directory>"
```

### Musan dataset
Musan noise samples should be located at 
```bash
$DAREA_DATA_PATH/musan/musan/noise/free-sound
```

You can download the dataset by using the following python snippet
```python
from darea.datasets.musan import Musan_Dataset
Musan_Dataset(download=True).download()
```

Check that everything is working correctly by running related tests with pytest
```bash
pytest tests/test_musan_dataset.py
```


### MIT room impulse response dataset


The room impulse response samples should be located at 
```bash
$DAREA_DATA_PATH/mit_rir/Audio
```

You can download the dataset by using the following python snippet
```python
from darea.room_impulse import MIT_RIR_Dataset
MIT_RIR_Dataset(download=True).download()
```

Check that everything is working correctly by running related tests with pytest
```bash
pytest tests/test_mit_rir_dataset.py
```


## Dependencies



### Neural audio codecs

### STE estimator for conventional codecs

This repository uses torch audio wrappers for ffmpeg codecs. In torchaudio 2.3.0, supported ffmpeg major versions are 4, 5, 6

Current ffmpeg version is 7. To install ffmpeg version 6, run the following command
```bash
mamba install -c pytorch -c conda-forge 'ffmpeg<7'
```


### Torchaudio

# Installation

Develop mode
```
pip install -e .
```

