from models.GeneralBlocks import *


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.head = ConvBlock(params, 1, params.hidden_channels, params.dilation_factors[0])
        self.body = nn.Sequential()
        self.Fs = params.current_fs
        for i in range(params.num_layers - 2):
            block = ConvBlock(params, params.hidden_channels, params.hidden_channels, params.dilation_factors[i + 1])
            self.body.add_module('block%d' % (i + 1), block)
        self.tail = nn.Sequential()
        self.tail.add_module('tail0',
                             NormConv1d(in_channels=params.hidden_channels, out_channels=params.hidden_channels,
                                        kernel_size=params.filter_size,
                                        dilation=params.dilation_factors[-1]))
        self.filter = nn.Sequential(
            NormConv1d(in_channels=params.hidden_channels, out_channels=params.hidden_channels,
                       kernel_size=params.filter_size, padding=int((params.filter_size - 1) / 2)),
            nn.Tanh()
        )
        self.gate = nn.Sequential(
            NormConv1d(in_channels=params.hidden_channels, out_channels=params.hidden_channels,
                       kernel_size=params.filter_size, padding=int((params.filter_size - 1) / 2)),
            nn.Sigmoid()
        )
        self.out_conv = NormConv1d(params.hidden_channels, 1, kernel_size=1)
        self.pe_filter = PreEmphasisFilter(params.device)

    def forward(self, noise_plus_sig, prev_sig):
        out_head = self.head(noise_plus_sig)
        out_body = self.body(out_head)
        out_tail = self.tail(out_body)
        filter = self.filter(out_tail)
        gate = self.gate(out_tail)
        out_tail = filter * gate
        out_tail = self.out_conv(out_tail)
        out_filt = self.pe_filter(out_tail)
        ind = int((prev_sig.shape[2] - out_filt.shape[2]) / 2)
        prev_sig = prev_sig[:, :, ind:(prev_sig.shape[2] - ind)]
        output = out_filt + prev_sig
        return output


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        if params.run_mode == 'inpainting':
            mask = params.current_holes
        else:
            mask = None
        self.head = ConvBlock(params, 1, params.hidden_channels, params.dilation_factors[0], mask=mask)
        mask = self.head.mask_out
        self.body = nn.ModuleList()
        for i in range(params.num_layers - 2):
            block = ConvBlock(params, params.hidden_channels, params.hidden_channels,
                              params.dilation_factors[i + 1], mask=mask)
            mask = block.mask_out
            self.body.add_module('block%d' % (i + 1), block)
        self.mask_out = mask
        self.tail = NormConv1d(params.hidden_channels, 1, kernel_size=params.filter_size,
                               dilation=params.dilation_factors[-1])
        self.pe_filter = PreEmphasisFilter(params.device)

    def forward(self, sig, use_mask=False):
        out_head = self.head(sig, use_mask)
        out_body = out_head
        for b in self.body:
            out_body = b(out_body, use_mask)
        out_tail = self.tail(out_body)
        output = self.pe_filter(out_tail)
        return output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('ConvBlock') == -1 and hasattr(m, 'weight'):
        if m.weight.numel() > 1 and m.weight.requires_grad:  # scalar blocks are initiailized upon creation
            m.weight.data.normal_(0.0, 0.02)

    elif classname.find('Norm') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
