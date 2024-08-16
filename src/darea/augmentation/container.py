import torch

from .noise import NoiseAugmentation
from .codecs import CodecAugmentation
from .room_impulse import ConvolutionReverbAugment

from ..datasets.mit_rir import MIT_RIR_Dataset
from ..datasets.musan import Musan_Dataset

from typing import List

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

    def __init__(
        self,
        augmentations: List[str],
        sample_rate,
        segment_size,
        num_workers=0,
        partition="train",
        resample=True,
        shuffle=True,
        batch_size=1,
        num_random_choose=1,
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
            if aug == "noise":
                dataset = Musan_Dataset(
                    sampling_rate=sample_rate,
                    segment_size=segment_size,
                    partition=partition,
                    resample=resample,
                    shuffle=shuffle,
                )
                augmentation_modules.append(
                    NoiseAugmentation(
                        dataset, batch_size=batch_size, num_workers=num_workers
                    )
                )
            elif aug == "reverb":
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
            elif aug == "codec_mp3_8kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=8000)
                )
            elif aug == "codec_mp3_16kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=16000)
                )
            elif aug == "codec_mp3_32kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=32000)
                )
            elif aug == "codec_mp3_92kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=92000)
                )
            elif aug == "codec_mp3_128kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=128000)
                )
            elif aug == "codec_mp3_64kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=sample_rate, bitrate=128000)
                )
            elif aug == "codec_ogg_vorbis_8kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=8000)
                )
            elif aug == "codec_ogg_vorbis_16kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=16000)
                )
            elif aug == "codec_ogg_vorbis_32kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=32000)
                )
            elif aug == "codec_ogg_vorbis_92kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-vorbis", sample_rate=sample_rate, bitrate=92000)
                )
            elif aug == "codec_ogg_opus_8kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=8000)
                )
            elif aug == "codec_ogg_opus_16kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=16000)
                )
            elif aug == "codec_ogg_opus_32kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=32000)
                )
            elif aug == "codec_ogg_opus_64kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=64000)
                )
            elif aug == "codec_ogg_opus_92kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg-opus", sample_rate=sample_rate, bitrate=92000)
                )
            elif aug == "codec_g722_48kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="g722", sample_rate=sample_rate, bitrate=48000)
                )
            elif aug == "codec_g722_56kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="g722", sample_rate=sample_rate, bitrate=56000)
                )
            elif aug == "codec_g722_64kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="g722", sample_rate=sample_rate, bitrate=64000)
                )
            elif aug == "codec_speex_8kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="speex", sample_rate=sample_rate, bitrate=8000)
                )
            elif aug == "codec_speex_16kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="speex", sample_rate=sample_rate, bitrate=16000)
                )
            elif aug == "codec_speex_32kbps":
                augmentation_modules.append(
                    CodecAugmentation(format="speex", sample_rate=sample_rate, bitrate=32000)
                )
            elif aug == "codec_gsm_fr":
                augmentation_modules.append(
                    CodecAugmentation(format="gsm", sample_rate=sample_rate)
                )
            elif aug == "codec_speex":
                augmentation_modules.append(
                    CodecAugmentation(format="speex", sample_rate=sample_rate)
                )
            else:
                raise ValueError(f"Unknown augmentation {aug}")

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

        augmentations = ["noise", "reverb", "codec_mp3_32kbps", "codec_ogg_vorbis_32kbps"]
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
