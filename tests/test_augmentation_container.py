from darea.augmentation.container import AugmentationContainerAllDarea
from darea.augmentation.container import AugmentationContainerKeywords
import torch

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
        batch_size=2).to(device)
    
    iters = 10
    for _ in range(iters):
        x = torch.randn(2, 1, 16000).to(device)
        y = container(x, num_random_choose=2)
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
        batch_size=2).to(device)
    
    iters = 10
    for _ in range(iters):
        x = torch.randn(2, 1, 16000).to(device)
        y = container(x, num_random_choose=2)
        assert y.size() == x.size()

