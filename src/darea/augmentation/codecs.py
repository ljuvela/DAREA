import torch
import torchaudio
from torchaudio.io import CodecConfig, AudioEffector
import torchaudio.transforms as T
import os

import soundfile as sf

import tempfile
from subprocess import DEVNULL, STDOUT, check_call
import subprocess

from .estimators import StraightThroughEstimator


class FfMpegCommandLineWrapper():

    def __init__(self, codec, host_sample_rate=16000, codec_sample_rate=16000, bitrate=None):

        self.host_sample_rate = host_sample_rate
        self.codec_sample_rate = codec_sample_rate
        self.bitrate = bitrate
        self.codec = codec

    def apply(self, x, sample_rate):

        # use temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = f"{temp_dir}/temp.wav"
            temp_coded = f"{temp_dir}/temp_coded.wav"
            temp_output = f"{temp_dir}/temp_output.wav"
            # torchaudio.save(temp_input, x, sample_rate)
            sf.write(temp_input, x.numpy().squeeze(), sample_rate)

            if self.bitrate is None:
                cmd = f"ffmpeg -y -i {temp_input} -ar {self.codec_sample_rate} -c:a {self.codec} {temp_coded}"
            else:
                cmd = f"ffmpeg -y -i {temp_input} -ar {self.codec_sample_rate} -b:a {self.bitrate} -c:a {self.codec} {temp_coded}"

            # os.system(cmd)
            os.popen(cmd).read()
            # subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL).wait()

            cmd = f"ffmpeg -y -i {temp_coded} -ar {sample_rate} -c:a pcm_s16le {temp_output}"
            # os.system(cmd)
            os.popen(cmd).read()

            # load the audio file
            x_hat, sr = torchaudio.load(temp_output)
            x_hat = x_hat.T

        return x_hat


class CodecAugmentation(torch.nn.Module):
    def __init__(self, format:str, sample_rate=16000, bitrate=None, q_factor=None, grad_clip_norm_level=None):
        """

        Args: 
            format (str): Codec format. Supported formats are 'mp3', 'ogg-vorbis' and 'ogg-opus'
            sample_rate (int): Sample rate of the audio signal
            q_factor (int): Quality factor of the codec. The quality factor is a number between -2 and 10.
            bitrate (int): Bitrate of the codec. The bitrate is in bits per second. 
                The default value is 32000.

        Choose either q_factor or bitrate. If both are provided, ValueError will be raised.

        More details in 
        https://pytorch.org/audio/2.3.0/generated/torchaudio.io.AudioEffector.html#torchaudio.io.AudioEffector


        """
        super(CodecAugmentation, self).__init__()
        self.sample_rate = sample_rate
        self.bitrate = bitrate

        if q_factor is not None and bitrate is not None:
            raise ValueError("Choose either q_factor or bitrate, not both")
        
        if q_factor is None and bitrate is None:
            raise ValueError("Choose either q_factor or bitrate")

        self.ste = StraightThroughEstimator(grad_clip_norm_level)

        if bitrate is None:
            bitrate = -1 # Default value for bitrate in torchaudio bindings

        if format == "mp3":
            self.codec = AudioEffector(format='mp3', codec_config=CodecConfig(bit_rate=bitrate, qscale=q_factor))
        elif format == "aac":
            raise NotImplementedError("AAC codec is not supported")
            self.codec = AudioEffector(format="aac", codec_config=CodecConfig(bit_rate=bitrate, qscale=q_factor))
        elif format == "ogg-vorbis":
            self.codec = AudioEffector(format="ogg", encoder="vorbis", codec_config=CodecConfig(bit_rate=bitrate, qscale=q_factor))
        elif format == "ogg-opus":
            self.codec = AudioEffector(format="ogg", encoder="opus", codec_config=CodecConfig(bit_rate=bitrate, qscale=q_factor))
        elif format == "g722":
            self.codec = AudioEffector(format="wav", encoder="g722", codec_config=CodecConfig(bit_rate=bitrate, qscale=q_factor))
        elif format == "speex":
            self.codec = AudioEffector(format="ogg", encoder="libspeex", codec_config=CodecConfig(bit_rate=bitrate, qscale=q_factor))
        elif format == "gsm":
            self.codec = AudioEffector(format="gsm", encoder="libgsm", codec_config=CodecConfig(bit_rate=bitrate, qscale=q_factor))
        elif format == "g723_1":
            # self.codec = AudioEffector(format="g723_1", encoder="g723_1", codec_config=CodecConfig(bit_rate=bitrate))
            self.codec = FfMpegCommandLineWrapper(codec='g723_1', host_sample_rate=sample_rate, codec_sample_rate=8000, bitrate=6300)
        elif format == "g726":
            self.codec = AudioEffector(format="g726", encoder="g726", codec_config=CodecConfig(bit_rate=bitrate))
        elif format == "pcm16":
            self.codec = AudioEffector(format="wav", encoder="pcm_s16le")
        else:
            raise ValueError(f"Format '{format}' not supported")


    def forward(self, x: torch.Tensor):
        """
        Args:
            x: torch.Tensor (batch_size, channels, samples)
        """

        input = x

        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D tensor")
        
        if x.size(1) != 1:
            raise ValueError(f"Expected mono audio, got {x.size(1)} channels")

        # TODO: GPU safe operation (move to cpu if implementation is not GPU safe)
        x_hat_list = []
        for x_i in x:
            x_i = x_i.permute(1,0).to('cpu') # Codec inputs must be on CPU
            x_i_hat = self.codec.apply(x_i, sample_rate=self.sample_rate)
            x_i_hat = x_i_hat.permute(1,0).to(input.device)
            x_hat_list.append(
                x_i_hat
            )
        x_hat = torch.stack(x_hat_list, dim=0)

        # cut the input and output to the same length
        timesteps = min(x.size(-1), x_hat.size(-1))
        x = x[..., :timesteps]
        x_hat = x_hat[..., :timesteps]

        # Use straight through estimator to pass gradients when training
        if self.training:
            return self.ste(x, x_hat)
            # return x + x_hat - x.detach()
        else:
            return x_hat
