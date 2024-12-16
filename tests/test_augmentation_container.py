from darea.augmentation.container import AugmentationContainerAllDarea
from darea.augmentation.container import AugmentationContainerKeywords
import torch
import pytest

keywords = [
    # "codec_gsm_fr",
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
    "codec_pcm16",
    # "codec_speex",
    "codec_g723_1",
    "codec_dac_8kbps",
]
@pytest.mark.parametrize('keyword', keywords)
def test_augmentation_container_codecs(keyword):

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    container = AugmentationContainerKeywords(
        augmentations=[keyword],
        sample_rate=22050,
        segment_size=22050,
        num_workers=0,
        partition='train',
        resample=True,
        shuffle=True,
        batch_size=2,
        num_random_choose=1).to(device)

    x = torch.randn(1, 1, 16000).to(device)
    y = container(x)
    assert y.size() == x.size()

def test_augmentation_container_keywords():

    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    container = AugmentationContainerKeywords(
        augmentations=['musan_noise', 'mit_rir_reverb'],
        sample_rate=16000,
        segment_size=16000,
        num_workers=0,
        partition='train',
        resample=True,
        shuffle=True,
        batch_size=2,
        num_random_choose=2).to(device)

    iters = 1
    for _ in range(iters):
        x = torch.randn(2, 1, 16000).to(device)
        y = container(x)
        assert y.size() == x.size()

def test_augmentation_container_all():

    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    container = AugmentationContainerAllDarea(
        sample_rate=16000,
        segment_size=16000,
        num_workers=0,
        partition='train',
        resample=True,
        shuffle=True,
        batch_size=2, 
        num_random_choose=2).to(device)

    iters = 1
    for _ in range(iters):
        x = torch.randn(2, 1, 16000).to(device)
        y = container(x)
        assert y.size() == x.size()
