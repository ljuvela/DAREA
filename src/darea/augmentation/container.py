import torch

from .noise import NoiseAugmentation
from .codecs import CodecAugmentation
from .room_impulse import ConvolutionReverbAugment

from ..datasets.mit_rir import MIT_RIR_Dataset
from ..datasets.musan import Musan_Dataset

from typing import List

class AugmentationContainer(torch.nn.Module):

    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(augmentations)

    def forward(self, x, num_random_choose=1):

        if num_random_choose > len(self.augmentations):
            raise ValueError(
                f"Number of augmentations to choose must be less than or equal to the number of augmentations in the container. Got {num_random_choose} and {len(self.augmentations)} augmentations"
            )

        chosen_augmentations = torch.randperm(len(self.augmentations))[
            :num_random_choose
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
    ):

        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.num_workers = num_workers
        self.partition = partition
        self.resample = resample
        self.shuffle = shuffle
        self.batch_size = batch_size

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
            elif aug == "codec_mp3":
                augmentation_modules.append(
                    CodecAugmentation(format="mp3", sample_rate=16000)
                )
            elif aug == "codec_ogg":
                augmentation_modules.append(
                    CodecAugmentation(format="ogg", sample_rate=16000)
                )
            else:
                raise ValueError(f"Unknown augmentation {aug}")

        super().__init__(augmentation_modules)

    def forward(self, x, num_random_choose=1):
        return super().forward(x, num_random_choose)


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
    ):

        augmentations = ["noise", "reverb", "codec_mp3", "codec_ogg"]
        super().__init__(
            augmentations,
            sample_rate,
            segment_size,
            num_workers,
            partition,
            resample,
            shuffle,
            batch_size,
        )

    def forward(self, x, num_random_choose=1):
        return super().forward(x, num_random_choose)
