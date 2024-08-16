from darea.augmentation.container import AugmentationContainerAllDarea
from darea.augmentation.container import AugmentationContainerKeywords
import torch

def test_augmentation_container_codecs():

    keywords = ['codec_mp3_8kbps',
                'codec_mp3_16kbps',
                'codec_mp3_32kbps',
                'codec_mp3_92kbps',
                'codec_mp3_128kbps',
                'codec_mp3_256kbps',
                'codec_ogg_vorbis_8kbps',
                'codec_ogg_vorbis_16kbps',
                'codec_ogg_vorbis_32kbps',
                'codec_ogg_vorbis_64kbps',
                'codec_ogg_vorbis_92kbps',
                'codec_ogg_opus_8kbps',
                'codec_ogg_opus_16kbps',
                'codec_ogg_opus_16kbps',
                'codec_ogg_opus_32kbps',
                'codec_ogg_opus_92kbps',
                'codec_g722_48kbps'
                'codec_g722_56kbps'
                'codec_g722_64kbps',
                'codec_speex',
                'codec_gsm']
                
    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for keyword in keywords:

        container = AugmentationContainerKeywords(
            augmentations=[keyword],
            sample_rate=16000,
            segment_size=16000,
            num_workers=0,
            partition='train',
            resample=True,
            shuffle=True,
            batch_size=2,
            num_random_choose=1).to(device)

        iters = 10
        for _ in range(iters):
            x = torch.randn(2, 1, 16000).to(device)
            y = container(x)
            assert y.size() == x.size()

def test_augmentation_container_keywords():

    torch.manual_seed(0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    container = AugmentationContainerKeywords(
        augmentations=['noise', 'reverb'],
        sample_rate=16000,
        segment_size=16000,
        num_workers=0,
        partition='train',
        resample=True,
        shuffle=True,
        batch_size=2,
        num_random_choose=2).to(device)

    iters = 10
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

    iters = 10
    for _ in range(iters):
        x = torch.randn(2, 1, 16000).to(device)
        y = container(x)
        assert y.size() == x.size()

