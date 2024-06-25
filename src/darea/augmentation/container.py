import torch

from .noise import NoiseAugmentation
from .codecs import CodecAugmentation
from .room_impulse import ConvolutionReverbAugment

from ..datasets.mit_rir import MIT_RIR_Dataset
from ..datasets.musan import Musan_Dataset

class AugmentationContainer(torch.nn.Module):

    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = torch.nn.ModuleList(augmentations)

    def forward(self, x, num_random_choose=1):

        if num_random_choose > len(self.augmentations):
            raise ValueError(f"Number of augmentations to choose must be less than or equal to the number of augmentations in the container. Got {num_random_choose} and {len(self.augmentations)} augmentations")

        chosen_augmentations = torch.randperm(len(self.augmentations))[:num_random_choose]

        for idx in chosen_augmentations:
            x = self.augmentations[idx](x)

        return x

class AugmentationContainerAllDarea(AugmentationContainer):

    def __init__(self, sample_rate, segment_size, 
                 num_workers=0, partition='train',
                   resample=True, shuffle=True,
                   batch_size=1):

        self.sample_rate = sample_rate
        self.segment_size = segment_size
        self.num_workers = num_workers
        self.partition = partition
        self.resample = resample
        self.shuffle = shuffle
        self.batch_size = batch_size
        rir_dataset = MIT_RIR_Dataset(
            sampling_rate=self.sample_rate,
            segment_size=self.segment_size,
            partition=self.partition,
            resample=resample,
            shuffle=self.resample,
        )
        noise_dataset = Musan_Dataset(
            sampling_rate=self.sample_rate,
            segment_size=self.segment_size,
            partition=self.partition,
            resample=self.resample,
            shuffle=self.shuffle,
        )
        augmentations = [
            NoiseAugmentation(noise_dataset, batch_size=batch_size, num_workers=self.num_workers),
            ConvolutionReverbAugment(rir_dataset, num_workers=self.num_workers),
            CodecAugmentation(format='mp3', sample_rate=16000),
            CodecAugmentation(format='ogg', sample_rate=16000),
        ]

        super().__init__(augmentations)

    def forward(self, x, num_random_choose=1):
        return super().forward(x, num_random_choose)
