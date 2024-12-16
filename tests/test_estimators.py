import torch
from darea.augmentation.estimators import StraightThroughEstimator

def test_round_no_grad():

    torch.manual_seed(0)

    x = torch.randn(2, 1, 16000)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = torch.round(x)

    # no ste, assert that gradient is none
    z = x_hat
    assert z.size() == x.size()
    assert torch.allclose(z, x_hat)
    loss = z.sum()
    loss.backward()
    assert x.grad is not None

def test_ste():

    torch.manual_seed(0)

    x = torch.randn(2, 1, 16000)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = torch.round(x)

    ste = StraightThroughEstimator(clip_norm_level=None)

    z = ste(x, x_hat)
    z.retain_grad()

    assert z.size() == x.size()
    
    assert torch.allclose(z, x_hat)
    
    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.allclose(x.grad, z.grad)

    assert torch.norm(x.grad, dim=-1).max() > 1.0



def test_ste_norm():

    # fix random seed for reproducibility
    x = torch.randn(2, 1, 16000)
    x = torch.nn.Parameter(x, requires_grad=True)

    x_hat = torch.round(x)

    # check that gradient norm is clipped to 1
    ste = StraightThroughEstimator(clip_norm_level=1.0)
    z = ste(x, x_hat)
    z.retain_grad()

    assert z.size() == x.size()
    assert torch.allclose(z, x_hat)

    loss = z.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.norm(x.grad, dim=-1).max() <= 1.0 + 1e-5
