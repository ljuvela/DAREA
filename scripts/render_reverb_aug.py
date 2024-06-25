import argparse

from darea.datasets.mit_rir import MIT_RIR_Dataset
from darea.augmentation.room_impulse import ConvolutionReverbAugment

import torchaudio
import os

from glob import glob

def main(args):

    wavfiles = list(glob(args.input_wav_dir + '/*.wav'))

    for f in wavfiles:

        waveform, sample_rate = torchaudio.load(f)
        waveform = waveform.unsqueeze(0)

        dataset = MIT_RIR_Dataset(sampling_rate=sample_rate, segment_size=waveform.size(-1), partition='train', resample=True, shuffle=True)

        reverb = ConvolutionReverbAugment(dataset, num_workers=0)

        noisy_waveform = reverb(waveform)


        os.makedirs(args.output_wav_dir, exist_ok=True)

        torchaudio.save(args.output_wav_dir + '/' + os.path.basename(f), noisy_waveform.squeeze(0), sample_rate)

    print("Reverb augmentation successful")


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