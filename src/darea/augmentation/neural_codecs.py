import torch
import dac
import os
from transformers import EncodecModel, MimiModel
from torchaudio.transforms import Resample
from speechtokenizer import SpeechTokenizer
from nemo.collections.tts.models import AudioCodecModel


class DacAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        super(DacAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.dac_sample_rate = 24000

        model_path = dac.utils.download(model_type="24khz")
        if not os.path.isfile(model_path):
            print(f"Downloading model to {model_path}")
        self.model = dac.DAC.load(model_path)

        self.model.train()

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
        super(EncodecAugmentation, self).__init__()
        self.sample_rate = sample_rate
        self.bandwidth=bandwidth

        self.encodec_sample_rate = 24000

        model_path = "facebook/encodec_24khz"
        self.model = EncodecModel.from_pretrained(model_path)            

        self.model.train()

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
        
        padding_mask = torch.ones_like(x)
        # 1 = input value should be attended to 
        audio_values = self.model(x, padding_mask, bandwidth=self.bandwidth).audio_values

        x = audio_values

        if self.sample_rate != self.encodec_sample_rate:
            x = self.resampler_from_encodec(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
    

class MimiAugmentation(torch.nn.Module):

    def __init__(self, sample_rate=16000):
        super(MimiAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.mimi_sample_rate = 12_500

        model_path = "kyutai/mimi"
        self.model = MimiModel.from_pretrained(model_path)

        self.model.train()

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
        audio_values = self.model(x).audio_values

        x = audio_values

        if self.sample_rate != self.mimi_sample_rate:
            x = self.resampler_from_mimi(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
    
# To Do
class SpeechTokenizerAugmentation(torch.nn.Module):
    def __init__(self, sample_rate=16000):
        # pip install -U speechtokenizer beartype
        # https://github.com/ZhangXInFD/SpeechTokenizer
        super(SpeechTokenizerAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.st_sample_rate = 16_000

        config_path = '/path/config.json' # TO DO REMAP
        ckpt_path = '/path/SpeechTokenizer.pt'
        self.model = SpeechTokenizer.load_from_checkpoint(config_path, ckpt_path)

        self.model.train()

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
        
        codes = self.model.encode(x) # codes: (n_q, B, T)

        #RVQ_1 = codes[:1, :, :] # Contain content info, can be considered as semantic tokens
        #RVQ_supplement = codes[1:, :, :] # Contain timbre info, complete info lost by the first quantizer

        audio_values = self.model(codes)

        x = audio_values

        if self.sample_rate != self.st_sample_rate:
            x = self.resampler_from_mimi(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x
    


class NeMoAudioCodecAugmentation:
    def __init__(self, sample_rate=16000):
        # module load gcc
        # pip install "nemo_toolkit[tts]"
        # https://huggingface.co/nvidia/audio-codec-44khz
        super(SpeechTokenizerAugmentation, self).__init__()
        self.sample_rate = sample_rate

        self.nemo_sample_rate = 22_000

        model_name = "nvidia/audio-codec-22khz"
        self.model = AudioCodecModel.from_pretrained(model_name)
        self.model.train()

        self.sample_rate = sample_rate
        if self.sample_rate != self.nemo_sample_rate:
            self.resampler_to_nemo = Resample(orig_freq=self.sample_rate, new_freq=self.nemo_sample_rate)
            self.resampler_from_nemo = Resample(orig_freq=self.nemo_sample_rate, new_freq=self.sample_rate)


    def forward(self, x):

        if x.ndim != 3:
            raise ValueError(f"Expected input of shape (batch, channels=1, timestesps), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.shape}")

        timesteps = x.size(-1)
        
        if self.sample_rate != self.nemo_sample_rate:
            x = self.resampler_to_nemo(x)
        
        audio_tensor = x
        audio_len = torch.full((x.size(0),), x.shape(-1))
        
        encoded_tokens, encoded_len = self.nemo_codec_model.encode(audio=audio_tensor, audio_len=audio_len)
        reconstructed_audio, _ = self.nemo_codec_model.decode(tokens=encoded_tokens, tokens_len=encoded_len)

        x = reconstructed_audio

        if self.sample_rate != self.nemo_sample_rate:
            x = self.resampler_from_mimi(x)

        # cut to the original length
        x = x[:, :, :timesteps]

        return x