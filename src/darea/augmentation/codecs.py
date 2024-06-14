import torch
import torchaudio
from torchaudio.io import CodecConfig, AudioEffector

class CodecAugmentation(torch.nn.Module):
    def __init__(self, format:str, sample_rate=16000):
        """

        Args:


        More details in 
        https://pytorch.org/audio/2.3.0/generated/torchaudio.io.AudioEffector.html#torchaudio.io.AudioEffector


        """
        super(CodecAugmentation, self).__init__()
        self.sample_rate = sample_rate

        if format == "wav":
            self.codec = AudioEffector(format='wav')
        elif format == "mp3":
            self.codec = AudioEffector(format='mp3')
        elif format == "mp3-32":
            self.codec = AudioEffector(format='mp3', codec_config=CodecConfig(bit_rate=32000))
        elif format == "mp3-8":
            self.codec = AudioEffector(format='mp3', codec_config=CodecConfig(bit_rate=8000))
        elif format == "ogg":
            self.codec = AudioEffector(format='ogg')
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

        # Use straight through estimator to pass gradients when training
        if self.training:
            return x + x_hat - x.detach()
        else:
            return x_hat
