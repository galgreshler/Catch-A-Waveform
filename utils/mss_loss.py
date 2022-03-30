import torch
import numpy as np


def stft(sig, n_fft, hop_length, window_size):
    s = torch.stft(sig, n_fft, hop_length, win_length=window_size,
                   window=torch.hann_window(window_size, device=sig.device), return_complex=False)
    return s


def spec(x, n_fft, hop_length, window_size):
    s = stft(x, n_fft, hop_length, window_size)
    n = torch.norm(s, p=2, dim=-1)
    return n


def norm(x):
    return (x.view(x.shape[0], -1) ** 2).sum(dim=-1).sqrt()


def squeeze(x):
    if len(x.shape) == 3:
        assert x.shape[-1] in [1, 2]
        x = torch.mean(x, -1)
    if len(x.shape) != 2:
        raise ValueError(f'Unknown input shape {x.shape}')
    return x


def multi_scale_spectrogram_loss(params, x_in, x_out):
    losses = []
    args = [params.multispec_loss_n_fft,
            params.multispec_loss_hop_length,
            params.multispec_loss_window_size]
    for n_fft, hop_length, window_size in zip(*args):
        if window_size == -1:
            window_size = x_in.shape[1]
            hop_length = window_size + 1
            n_fft = int(2 ** np.ceil(np.log2(window_size)))
        if params.run_mode == 'inpainting':
            x_in_cp = x_in.clone()
            x_out_cp = x_out.clone()

            not_valid_idx_start = [idx[0] for idx in params.current_holes]
            not_valid_idx_end = [idx[1] for idx in params.current_holes]

            x_in = x_in_cp[:, :int(not_valid_idx_start[0])]
            x_out = x_out_cp[:, :int(not_valid_idx_start[0])]

            spec_in = spec(squeeze(x_in.float()), n_fft, hop_length, window_size)
            spec_out = spec(squeeze(x_out.float()), n_fft, hop_length, window_size)

            if len(params.current_holes) > 1:
                for i in range(len(params.current_holes) - 1):
                    spec_in = torch.cat((spec_in,
                                spec(squeeze(x_in_cp[:, not_valid_idx_end[i] + 1:not_valid_idx_start[i+1]].float()),
                                n_fft, hop_length, window_size)), dim=2)
                    spec_out = torch.cat((spec_out,
                                spec(squeeze(x_out_cp[:, not_valid_idx_end[i] + 1:not_valid_idx_start[i+1]].float()),
                                n_fft, hop_length, window_size)), dim=2)
                spec_in = torch.cat((spec_in,
                            spec(squeeze(x_in_cp[:, not_valid_idx_end[-1] + 1:].float()),
                            n_fft, hop_length, window_size)), dim=2)
                spec_out = torch.cat((spec_out,
                            spec(squeeze(x_out_cp[:, not_valid_idx_end[-1] + 1:].float()),
                            n_fft, hop_length, window_size)), dim=2)

        else:
            spec_in = spec(squeeze(x_in.float()), n_fft, hop_length, window_size)
            spec_out = spec(squeeze(x_out.float()), n_fft, hop_length, window_size)
        losses.append(norm(spec_in - spec_out))
    return sum(losses) / len(losses)
