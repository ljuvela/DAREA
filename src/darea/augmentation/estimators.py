import torch


class StraightThroughEstimatorFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, x_hat, clip_norm_level=0.0):
        ctx.clip_norm_level = clip_norm_level
        return x_hat

    @staticmethod
    def backward(ctx, grad_output):

        grad_input = grad_output.clone()
        # normalize the gradient
        if ctx.clip_norm_level > 0:
            norm = torch.norm(grad_input, dim=-1, keepdim=True)
            mask = norm > ctx.clip_norm_level
            mask = mask.expand_as(grad_input)
            grad_input_scaled = grad_input / norm * ctx.clip_norm_level
            grad_input = torch.where(mask, grad_input_scaled, grad_input)

        return grad_input, grad_input, None
    

class StraightThroughEstimator(torch.nn.Module):

    def __init__(self, clip_norm_level=None):
        super(StraightThroughEstimator, self).__init__()
        if clip_norm_level is None:
            clip_norm_level = 0.0
        else:
            clip_norm_level = float(clip_norm_level)
        self.clip_norm_level = clip_norm_level

    def forward(self, x, x_hat):
        return StraightThroughEstimatorFunction.apply(x, x_hat, self.clip_norm_level)
    