import torch

from .noise import NoiseAugmentation
from .codecs import CodecAugmentation
from .neural_codecs import NeuralCodecAugmentation
from .room_impulse import ConvolutionReverbAugment
from .filters import LowPassFilterAugmentation, HighPassFilterAugmentation
from .dropout import SampleDropoutAugmentation, StftDropoutAugmentation
from .time_stretch import TimeStretchAugmentation

from ..datasets.mit_rir import MIT_RIR_Dataset
from ..datasets.musan import Musan_Dataset

from typing import List

class DummyAugmentation(torch.nn.Module):
    """Dummy augmentation class for no augmentation
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class AugmentationContainer(torch.nn.Module):

    def __init__(self, augmentations, num_random_choose=1):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(augmentations)
        self.num_random_choose = num_random_choose

    def forward(self, x):

        if self.num_random_choose > len(self.augmentations):
            raise ValueError(
                f"Number of augmentations to choose must be less than or equal to the number of augmentations in the container. Got {self.num_random_choose} and {len(self.augmentations)} augmentations"
            )

        chosen_augmentations = torch.randperm(len(self.augmentations))[
            :self.num_random_choose
        ]

        for idx in chosen_augmentations:
            x = self.augmentations[idx](x)

        return x


class AugmentationContainerKeywords(AugmentationContainer):

    def list_valid_augmentation_keywords(self):
        return [
            "musan_noise",
            "mit_rir_reverb",
            "lowpass_4kHz",
            "highpass_500Hz",
            "sample_dropout",
            "stft_dropout",
            "time_stretch",
            "codec_mp3_8kbps",
            "codec_mp3_16kbps",
            "codec_mp3_32kbps",
            "codec_mp3_64kbps",
            "codec_mp3_92kbps",
            "codec_mp3_128kbps",
            "codec_ogg_vorbis_8kbps",
            "codec_ogg_vorbis_16kbps",
            "codec_ogg_vorbis_32kbps",
            "codec_ogg_vorbis_64kbps",
            "codec_ogg_vorbis_92kbps",
            "codec_ogg_vorbis_128kbps",
            "codec_ogg_vorbis_q-2",
            "codec_ogg_vorbis_q0",
            "codec_ogg_vorbis_q1",
            "codec_ogg_vorbis_q2",
            "codec_ogg_vorbis_q3",
            "codec_ogg_vorbis_q4",
            "codec_ogg_opus_8kbps",
            "codec_ogg_opus_16kbps",
            "codec_ogg_opus_32kbps",
            "codec_ogg_opus_64kbps",
            "codec_ogg_opus_92kbps",
            "codec_ogg_opus_128kbps",
            "codec_g722_48kbps",
            "codec_g722_56kbps",
            "codec_g722_64kbps",
            "codec_g723_1",
            "codec_dac_8kbps",
            "codec_pcm16"
            "nocodec",
        ]

    def __init__(
        self,
        augmentations: List[str],
        sample_rate,
        segment_size=None,
        num_workers=0,
        partition="train",
        resample=True,
        shuffle=True,
        batch_size=1,
        num_random_choose=1,
        grad_clip_norm_level=None,
    ):

        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.num_workers = num_workers
        self.partition = partition
        self.resample = resample
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_random_choose = num_random_choose

        augmentation_modules = []
        for aug in augmentations:
            if aug == "musan_noise":
                dataset = Musan_Dataset(
                    sampling_rate=sample_rate,
                    segment_size=segment_size,
                    partition=partition,
                    resample=resample,
                    shuffle=shuffle,
                )
                augmentation_modules.append(
                    NoiseAugmentation(
                        dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        min_snr=20,
                        max_snr=35.0,
                    )
                )
            elif aug == "mit_rir_reverb":
                dataset = MIT_RIR_Dataset(
                    sampling_rate=sample_rate,
                    segment_size=segment_size,
                    partition=partition,
                    resample=resample,
                    shuffle=shuffle,
                )
                augmentation_modules.append(
                    ConvolutionReverbAugment(dataset, num_workers=num_workers)
                )
            elif aug == "lowpass_4kHz":
                augmentation_modules.append(
                    LowPassFilterAugmentation(cutoff_freq_min=4000, cutoff_freq_max=4000, sample_rate=sample_rate)
                )
            elif aug == "highpass_500Hz":
                augmentation_modules.append(
                    HighPassFilterAugmentation(cutoff_freq_min=500, cutoff_freq_max=500, sample_rate=sample_rate)
                )
            elif aug == 'sample_dropout':
                augmentation_modules.append(
                    SampleDropoutAugmentation(p=0.001)
                )
            elif aug == 'stft_dropout':
                augmentation_modules.append(
                    StftDropoutAugmentation(p=0.001)
                )
            elif aug == "time_stretch":
                augmentation_modules.append(
                    TimeStretchAugmentation(sample_rate=sample_rate,
                                             min_rate=0.90,
                                             max_rate=1.10)
                )
            elif aug == "codec_pcm16":
                augmentation_modules.append(
                    CodecAugmentation(format="pcm16", sample_rate=sample_rate, bitrate=16*sample_rate, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_mp3_8kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=8000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_mp3_16kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=16000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_mp3_32kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=32000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_mp3_64kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=64000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_mp3_92kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=92000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_mp3_128kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=128000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_8kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=8000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_16kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=16000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_32kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=32000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_64kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=64000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_92kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=92000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_128kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=128000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_q-2":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, q_factor=-2, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_q0":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, q_factor=0, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_q1":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, q_factor=1, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_q2":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, q_factor=2, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_q3":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, q_factor=3, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_vorbis_q4":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, q_factor=4, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_opus_8kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=8000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_opus_16kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=16000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_opus_32kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=32000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_opus_64kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=64000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_opus_92kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=92000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_ogg_opus_128kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=128000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_g722_48kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="g722", sample_rate=sample_rate, bitrate=48000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_g722_56kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="g722", sample_rate=sample_rate, bitrate=56000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_g722_64kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="g722", sample_rate=sample_rate, bitrate=64000, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_g723_1":
                augmentation_modules.append(
                    CodecAugmentation(format="g723_1", bitrate=64000, sample_rate=sample_rate, grad_clip_norm_level=grad_clip_norm_level)
                )
            elif aug == "codec_dac_8kbps":
                augmentation_modules.append(
                    NeuralCodecAugmentation(sample_rate=sample_rate)
                )
            elif aug == "nocodec":
                # for no codec, add a dummy module
                augmentation_modules.append(DummyAugmentation())

            else:
                raise ValueError(f"Unknown augmentation '{aug}'")

        super().__init__(augmentation_modules, num_random_choose)

    def forward(self, x):
        return super().forward(x)


class AugmentationContainerAllDarea(AugmentationContainerKeywords):

    def __init__(
        self,
        sample_rate,
        segment_size,
        num_workers=0,
        partition="train",
        resample=True,
        shuffle=True,
        batch_size=1,
        num_random_choose=1,
    ):

        augmentations = self.list_valid_augmentation_keywords()
        super().__init__(
            augmentations=augmentations,
            sample_rate=sample_rate,
            segment_size=segment_size,
            num_workers=num_workers,
            partition=partition,
            resample=resample,
            shuffle=shuffle,
            batch_size=batch_size,
            num_random_choose=num_random_choose
        )

    def forward(self, x):
        return super().forward(x)
