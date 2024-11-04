import torch


class SampleDropoutAugmentation(torch.nn.Module):

    def __init__(self, p=0.0001):
        super(SampleDropoutAugmentation, self).__init__()

        self.dropout = torch.nn.Dropout(p=p)

    def forward(self, x):

        x = self.dropout(x)
        return x


class StftDropoutAugmentation(torch.nn.Module):

    def __init__(self, p=0.001, hop_length=512, win_length=2048, n_fft=2048):
        super(StftDropoutAugmentation, self).__init__()

        self.dropout = torch.nn.Dropout(p=p)
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft

        self.register_buffer('window', torch.hann_window(win_length))

    def forward(self, x):

        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor, got {x.dim()}D tensor")

        if x.size(1) != 1:
            raise ValueError(f"Expected mono audio, got {x.size(1)} channels")
    
        length = x.size(-1)

        x = x.squeeze(1)

        X = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            return_complex=True,
            onesided=True,
            center=True,
            window=self.window
        )

        # mask magnitudes
        mask = self.dropout(torch.ones_like(X.abs()))
        X = X * mask

        x = torch.istft(
            X,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            onesided=True,
            center=True,
            window=self.window,
            length=length
        )

        x = x.unsqueeze(1)

        return x
