# DAREA
Differentiable augmentation and robustness evaluation for audio


## Setup

Noise and room impulse response samples are stored in `DAREA_DATA_PATH`. The user should set this environment variable to the path where the data is stored. For example, in bash:
```shell
export DAREA_DATA_PATH='./data'
```

If the data is missing, it will be downloaded automatically.

## Dependencies



### Neural audio codecs

### STE estimator for conventional codecs

### Torchaudio

# Installation

Develop mode
```
pip install -e .
```

## Features

- All modules inherit `torch.nn.Module`

- Composition with `torch.nn.Sequential`

- 

- Random 

## Augmentation types