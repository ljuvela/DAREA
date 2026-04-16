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

DAC and SNAC can be installed directly from the official repositories. 

https://github.com/descriptinc/descript-audio-codec
https://github.com/hubertsiuzdak/snac

```bash
pip install descript-audio-codec
pip install snac
```

However, for training time augmentation with other neural codecs you need to clone and install the following modified implementations. 

https://github.com/ollipauna/encodec
https://github.com/ollipauna/speechtokenizer
https://github.com/ollipauna/moshi

Or alternatively run

```bash
pip install git+https://github.com/ollipauna/encodec.git
pip install beartype git+https://github.com/ollipauna/speechtokenizer.git
pip install "git+https://github.com/ollipauna/moshi.git#egg=moshi&subdirectory=moshi"

```

These repositories implement STE based gradient estimation during inference. Note that the moshi needs to compile C/C++ extensions and requires `gcc` or similar compiler on the system.


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

