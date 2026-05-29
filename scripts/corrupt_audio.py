import argparse

from darea.augmentation import noise
from darea.datasets.musan  import Musan_Dataset
from darea.augmentation.noise import NoiseAugmentation
from darea.datasets.mit_rir import MIT_RIR_Dataset
from darea.augmentation.room_impulse import ConvolutionReverbAugment

import torch
import torchaudio
import os

from glob import glob

def main(args):

    # wavfiles = list(glob(args.input_wav_dir + '/*.wav'))
    wavfiles = list(glob(args.input_wav_dir + '/*.mp3'))

    for f in wavfiles:

        waveform, sample_rate = torchaudio.load(f)
        waveform = waveform.unsqueeze(0)

        # fix random seed
        torch.manual_seed(1)

        dataset = Musan_Dataset(sampling_rate=sample_rate, segment_size=waveform.size(-1), partition='train', resample=True, shuffle=True)
        noise = NoiseAugmentation(dataset, batch_size=1, num_workers=0,
                                  min_snr=40, max_snr=40.0)

        dataset = MIT_RIR_Dataset(sampling_rate=sample_rate, segment_size=waveform.size(-1), partition='train', resample=True, shuffle=True)
        reverb = ConvolutionReverbAugment(dataset, num_workers=0)

        energy = waveform.pow(2).mean()

        x = waveform
        x = reverb(x)
        x = noise(x)
        x = x + torch.randn_like(x) * energy.sqrt() * 0.01
        x = torch.tanh(x / x.abs().max() * 8.99)
        x = reverb(x)

        x = x / x.abs().max()


        os.makedirs(args.output_wav_dir, exist_ok=True)

        torchaudio.save(args.output_wav_dir + '/' + os.path.basename(f), x.squeeze(0), sample_rate)

    print("Augmentation successful")


def parse_args():

    parser = argparse.ArgumentParser(description='Render noise augmentation')
    parser.add_argument('--sample_rate', type=int, default=16000, required=False)
    parser.add_argument('--input_wav_dir', type=str, default=None, required=True)
    parser.add_argument('--output_wav_dir', type=str, default='output', required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    main(args)