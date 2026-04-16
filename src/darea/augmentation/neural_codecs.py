import torch
import dac
import os
from encodec import EncodecModel
from torchaudio.transforms import Resample
from speechtokenizer import SpeechTokenizer
from huggingface_hub import hf_hub_download
from moshi.models import loaders
from snac import SNAC
from bigcodec.vq.codec_encoder import CodecEncoder
from bigcodec.vq.codec_decoder import CodecDecoder

class DacAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        # https://github.com/descriptinc/descript-audio-codec
        super(DacAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.dac_sample_rate = 24_000

        model_path = dac.utils.download(model_type="24khz")
        if not os.path.isfile(model_path):
            print(f"Downloading model to {model_path}")
        self.model = dac.DAC.load(model_path)

        self.model.eval()
        self.model.requires_grad_(False)

        self.sample_rate = sample_rate
        if self.sample_rate != self.dac_sample_rate:
            self.resampler_to_dac = Resample(orig_freq=self.sample_rate, new_freq=self.dac_sample_rate)
            self.resampler_from_dac = Resample(orig_freq=self.dac_sample_rate, new_freq=self.sample_rate)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)

        if self.sample_rate != self.dac_sample_rate:
            x = self.resampler_to_dac(x)

        out = self.model.forward(x, sample_rate=self.dac_sample_rate)

        x = out["audio"]

        if self.sample_rate != self.dac_sample_rate:
            x = self.resampler_from_dac(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
    

class EncodecAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000, bandwidth=None):
         # bandwidths [1.5, 3, 6, 12, 24] uses largest by default
         # https://github.com/facebookresearch/encodec
         # https://github.com/ollipauna/encodec

        super(EncodecAugmentation, self).__init__()
        self.sample_rate = sample_rate
        self.bandwidth=bandwidth

        self.encodec_sample_rate = 24_000
        
        self.model = EncodecModel.encodec_model_24khz()

        if bandwidth:
            self.model.set_target_bandwidth(bandwidth)
   
        self.model.train()  # cudnn RNN backward can only be called in training mode
        self.model.requires_grad_(False)

        self.sample_rate = sample_rate
        if self.sample_rate != self.encodec_sample_rate:
            self.resampler_to_encodec = Resample(orig_freq=self.sample_rate, new_freq=self.encodec_sample_rate)
            self.resampler_from_encodec = Resample(orig_freq=self.encodec_sample_rate, new_freq=self.sample_rate)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)
        

        if self.sample_rate != self.encodec_sample_rate:
            x = self.resampler_to_encodec(x)
        
        audio_values = self.model(x)

        x = audio_values

        if self.sample_rate != self.encodec_sample_rate:
            x = self.resampler_from_encodec(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
    

class MimiAugmentation(torch.nn.Module):
    # https://github.com/kyutai-labs/moshi
    # https://github.com/ollipauna/moshi

    def __init__(self, sample_rate=16000, num_codebooks=8):
        super(MimiAugmentation, self).__init__()
        self.sample_rate = sample_rate
        
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        self.model = loaders.get_mimi(mimi_weight, num_codebooks=num_codebooks)
        self.mimi_sample_rate = 24_000

        self.model.eval()
        self.model.requires_grad_(False)

        self.sample_rate = sample_rate
        if self.sample_rate != self.mimi_sample_rate:
            self.resampler_to_mimi = Resample(orig_freq=self.sample_rate, new_freq=self.mimi_sample_rate)
            self.resampler_from_mimi = Resample(orig_freq=self.mimi_sample_rate, new_freq=self.sample_rate)

    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)
        

        if self.sample_rate != self.mimi_sample_rate:
            x = self.resampler_to_mimi(x)
        
        # 1 = input value should be attended to 
        audio_values = self.model(x).x

        x = audio_values

        if self.sample_rate != self.mimi_sample_rate:
            x = self.resampler_from_mimi(x)     

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
    

class SpeechTokenizerAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        # https://github.com/ZhangXInFD/SpeechTokenizer
        # https://github.com/ollipauna/SpeechTokenizer
        super(SpeechTokenizerAugmentation, self).__init__()
        self.sample_rate = sample_rate

        path = os.environ.get('SPEECH_TOKENIZER_PATH', None)

        if path is None:
            raise RuntimeError("Environment variable SPEECH_TOKENIZER_PATH is not set! "
                               "Please set it to the path where the model config should be stored ")


        config_path = 'config.json'
        ckpt_path = 'SpeechTokenizer.pt'

        config_path = os.path.join(path, config_path) 
        ckpt_path = os.path.join(path, ckpt_path) 

        self.model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)

        self.st_sample_rate = self.model.sample_rate # 16000
        self.model.train()  # cudnn RNN backward can only be called in training mode
        self.model.requires_grad_(False)

        self.sample_rate = sample_rate
        if self.sample_rate != self.st_sample_rate:
            self.resampler_to_st = Resample(orig_freq=self.sample_rate, new_freq=self.st_sample_rate)
            self.resampler_from_st = Resample(orig_freq=self.st_sample_rate, new_freq=self.sample_rate)


    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)

        if self.sample_rate != self.st_sample_rate:
            x = self.resampler_to_st(x)
        
        audio_values, _, _ = self.model(x) # codes: (n_q, B, T)
        
        x = audio_values

        if self.sample_rate != self.st_sample_rate:
            x = self.resampler_from_st(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x


class SnacAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        # https://github.com/hubertsiuzdak/snac
        super(SnacAugmentation, self).__init__()
        self.sample_rate = sample_rate


        self.model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self.model.requires_grad_(False)

        self.snac_sample_rate = 24000 

        self.sample_rate = sample_rate
        if self.sample_rate != self.snac_sample_rate:
            self.resampler_to_snac = Resample(orig_freq=self.sample_rate, new_freq=self.snac_sample_rate)
            self.resampler_from_snac = Resample(orig_freq=self.snac_sample_rate, new_freq=self.sample_rate)


    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)

        if self.sample_rate != self.snac_sample_rate:
            x = self.resampler_to_snac(x)
        
        audio_values, _ = self.model(x)
        
        x = audio_values

        if self.sample_rate != self.snac_sample_rate:
            x = self.resampler_from_snac(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
