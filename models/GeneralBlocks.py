import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class PreEmphasisFilter(nn.Module):
    def __init__(self, device):
        super(PreEmphasisFilter, self).__init__()
        self.alpha = torch.Tensor([0.97]).to(device)
        self.alpha.requires_grad = False

    def forward(self, x):
        output = torch.cat((x[:, :, 0].view(x.shape[0], x.shape[1], 1), x[:, :, 1:] - self.alpha * x[:, :, :-1]), dim=2)
        return output


class NormConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        super(NormConv1d, self).__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                          stride=stride, padding=padding, dilation=dilation, bias=bias))

    def forward(self, x):
        output = self.conv(x)
        return output


class ConvBlock(nn.Sequential):
    def __init__(self, params, in_channels, out_channels, dilation=1, filter_size=None, mask=None):
        super(ConvBlock, self).__init__()
        if filter_size is None:
            filter_size = params.filter_size
        if mask is not None:
            self.mask_in = mask
            self.rf = int((params.filter_size - 1) * dilation)
            self.mask_out = torch.zeros(self.mask_in.shape)
            self.mask_out[0] = int(mask[0] - self.rf)
            self.mask_out[1] = int(mask[1])
        else:
            self.mask_out = None
        self.conv = NormConv1d(in_channels, out_channels, filter_size, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, use_mask=False):
        out_conv = self.conv(x)
        if use_mask:
            tmp = torch.cat((out_conv[:, :, :int(self.mask_out[0])], out_conv[:, :, int(self.mask_out[1] + 1):]), dim=2)
            tmp_norm = self.norm(tmp)
            out_norm = out_conv
            out_norm[:, :, :int(self.mask_out[0])] = tmp_norm[:, :, :int(self.mask_out[0])]
            out_norm[:, :, int(self.mask_out[1] + 1):] = tmp_norm[:, :, int(self.mask_out[0]):]
        else:
            out_norm = self.norm(out_conv)
        return self.activation(out_norm)
