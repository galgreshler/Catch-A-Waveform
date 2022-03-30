import os
import numpy as np
import soundfile as sf
import glob
from numpy.fft import fft, ifft
from utils.resize_right import ResizeLayer
from params import Params
import torch
import torch.nn as nn
import librosa
from models import CAW
from scipy import interpolate


def get_noise(params, shape):
    return torch.randn(shape, device=params.device)


def stitch_signals(real_signal, signal_to_stitch, frame_idcs, window_size=2 ** 14 - 1):
    naive_stitched_signal = np.copy(real_signal)
    for idx in frame_idcs:
        naive_stitched_signal[idx] = signal_to_stitch[idx]
    # overlap add between real and generated signals
    ola_stitched_signal = np.copy(naive_stitched_signal)
    for i, win_size in enumerate(window_size):
        if win_size % 2 == 0:
            win_size -= 1
        window = np.hanning(win_size)
        transition_in_idcs = range(frame_idcs[i][0] - (win_size + 1) // 2, frame_idcs[i][0])
        in_window = window[:(win_size + 1) // 2]
        out_window = window[(win_size + 1) // 2 - 1:]
        transition_out_idcs = range(frame_idcs[i][-1], frame_idcs[i][-1] + win_size // 2 + 1)
        ola_stitched_signal[transition_in_idcs] = in_window * signal_to_stitch[transition_in_idcs] + out_window * \
                                              real_signal[transition_in_idcs]
        ola_stitched_signal[transition_out_idcs] = in_window * real_signal[transition_out_idcs] + out_window * \
                                               signal_to_stitch[transition_out_idcs]
    return ola_stitched_signal


def calc_snr(est, real):
    min_len = min(len(est), len(real))
    real = real[:min_len]
    est = est[:min_len]
    real_fit = real
    est_fit = est
    snr = 10 * np.log10(sum(real_fit ** 2) / sum((est_fit - real_fit) ** 2))

    return snr


def calc_lsd(est, real, eps=1e-15):
    WIN_SIZE = 2048
    min_length = min(len(est), len(real))
    assert abs(len(real) - len(est)) / min_length < 0.2, 'Mismatch in length between 2 signals'
    real = real[:min_length]
    est = est[:min_length]
    X = abs(librosa.stft(est, n_fft=WIN_SIZE, hop_length=WIN_SIZE)) ** 2
    X[X < eps] = eps
    X = np.log(X)
    Y = abs(librosa.stft(real, n_fft=WIN_SIZE, hop_length=WIN_SIZE)) ** 2
    Y[Y < eps] = eps
    Y = np.log(Y)
    Z = (X - Y) ** 2
    lsd = np.sqrt(Z.mean(0)).mean()
    return lsd


def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model


def calc_gradient_penalty(params, netD, real_data, fake_data, LAMBDA, alpha=None, _grad_outputs=None, mask_ratio=None):
    # Gradient penalty method for WGAN
    if alpha is None:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        if torch.cuda.is_available():
            alpha = alpha.cuda(real_data.get_device())  # gpu) #if use_cuda else alpha
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True)
    if params.run_mode == 'inpainting':
        use_mask = True
    else:
        use_mask = False
        mask_ratio = 1
    disc_interpolates = netD(interpolates, use_mask)
    if params.run_mode == 'inpainting':
        disc_interpolates_cp = disc_interpolates.clone()
        disc_interpolates = disc_interpolates_cp[:, :, :params.not_valid_idx_start[0]]
        if len(params.current_holes) > 1:
            for i in range(len(params.current_holes) - 1):
                disc_interpolates = torch.cat((disc_interpolates, disc_interpolates_cp[:, :, params.not_valid_idx_end[i] + 1:params.not_valid_idx_start[i+1]]), dim=2)
        disc_interpolates = torch.cat((disc_interpolates, disc_interpolates_cp[:, :, params.not_valid_idx_end[-1] + 1:]), dim=2)
    if _grad_outputs is None:
        _grad_outputs = torch.ones(disc_interpolates.size())
        if torch.cuda.is_available():
            _grad_outputs = _grad_outputs.cuda(real_data.get_device())
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=_grad_outputs,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((mask_ratio * gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    del gradients, interpolates, _grad_outputs, disc_interpolates
    return gradient_penalty


def create_input_signals(params, input_signal, Fs):
    # Performs downscaling for desired scales and outputs list of signals
    signals_list = []
    fs_list = []
    n_scales = len(params.scales)
    set_first_scale = False
    rf = calc_receptive_field(params.filter_size, params.dilation_factors)
    for k in range(n_scales):
        downsample = params.scales[k]
        fs = int(Fs / downsample)
        if downsample == 1:
            coarse_sig = input_signal
        else:
            coarse_sig = torch.Tensor(librosa.resample(input_signal.squeeze().numpy(), Fs, fs))
        if params.run_mode == 'inpainting':
            holes_sum = 0
            for hole_idx in params.inpainting_indices:
                holes_sum += hole_idx[1] - hole_idx[0] + 2*rf
            if (holes_sum) / params.Fs * fs > len(coarse_sig):
                    continue
        if params.speech and fs < 500:
            continue
        if params.set_first_scale_by_energy and not params.speech:
            e = (coarse_sig ** 2).mean()
            if e < params.min_energy_th and not set_first_scale:
                continue
        set_first_scale = True
        signals_list.append(coarse_sig)
        assert np.mod(fs, 1) == 0, 'Sampling rate is not integer'
        fs_list.append(int(fs))

        # Write downsampled real sound
        filename = 'real@%dHz.wav' % fs
        write_signal(os.path.join(params.output_folder, filename), coarse_sig.cpu(), fs)

    return signals_list, fs_list


def calc_pad_size(params, dilation_factors=None, filter_size=None):
    if dilation_factors is None:
        dilation_factors = params.dilation_factors
    if filter_size is None:
        filter_size = params.filter_size
    return int(np.ceil(sum(dilation_factors) * (filter_size - 1) / 2))


def calc_receptive_field(filter_size, dilation_factors, Fs=None):
    if Fs is None:
        # in samples
        return (filter_size * dilation_factors[0] + sum(dilation_factors[1:]) * (filter_size - 1))
    else:
        # in [ms]
        return (filter_size * dilation_factors[0] + sum(dilation_factors[1:]) * (filter_size - 1)) / Fs * 1e3


def resample_sig(params, input_signal, orig_fs=None, target_fs=None):
    if not hasattr(params, 'resamplers') or type(params.resamplers) == str:
        params.resamplers = {}
    if (orig_fs, target_fs) in params.resamplers.keys() and params.resamplers[(orig_fs, target_fs)].in_shape[2] == \
            input_signal.shape[2]:
        resampler = params.resamplers[(orig_fs, target_fs)]
    else:
        in_shape = input_signal.shape
        scale_factors = (1, 1, target_fs / orig_fs)
        resampler = ResizeLayer(in_shape, scale_factors=scale_factors, device=params.device)
        params.resamplers[(orig_fs, target_fs)] = resampler
    new_sig = resampler(input_signal)

    return new_sig


def get_input_signal(params):
    file_name = params.input_file.split('.')
    if len(file_name) < 2:
        params.input_file = '.'.join([params.input_file, 'wav'])
    output_folder = file_name[0].replace(' ', '_')
    if len(params.segments_to_train) == 0:
        samples, Fs = librosa.load(os.path.join('inputs', params.input_file), sr=None,
                                   offset=params.start_time, duration=2 * params.max_length)
    else:
        if len(params.segments_to_train) % 2 == 1:
            raise Exception('Please provide valid segments, in the form of: start1, end1, start2, end2, ... in [sec]')
        params.max_length = 1e3  # dummy
        params.min_length = 0
        for idx in range(0, len(params.segments_to_train), 2):
            if idx == 0:
                samples, Fs = librosa.load(os.path.join('inputs', params.input_file), sr=None,
                                           offset=params.segments_to_train[idx],
                                           duration=params.segments_to_train[idx + 1] - params.segments_to_train[idx])
            else:
                _samples, _ = librosa.load(os.path.join('inputs', params.input_path), sr=None,
                                           offset=params.segments_to_train[idx],
                                           duration=params.segments_to_train[idx + 1] - params.segments_to_train[
                                               idx])
                samples = np.concatenate((samples, _samples))

    if samples.shape[0] / Fs > params.max_length:
        n_samples = int(params.max_length * Fs)
        samples = samples[:n_samples]

    params.output_folder = output_folder
    params.output_folder = os.path.join('outputs', params.output_folder)
    params.Fs = Fs
    if params.init_sample_rate < Fs:
        hr_samples = samples.copy()
        samples = librosa.resample(hr_samples, Fs, params.init_sample_rate)
        params.Fs = params.init_sample_rate
    params.norm_factor = max(abs(samples.reshape(-1)))
    samples = samples / params.norm_factor
    return samples


def draw_signal(params, generators_list, signals_lengths_list, fs_list, noise_amp_list, reconstruction_noise_list=None,
                condition=None, output_all_scales=False):
    # Draws a signal up to current scale, using learned generators
    pad_size = calc_pad_size(params)
    if output_all_scales:
        signals_all_scales = []
    for scale_idx, (netG, noise_amp) in enumerate(zip(generators_list, noise_amp_list)):
        signal_padder = nn.ConstantPad1d(pad_size, 0)
        if condition is None:
            n_samples = signals_lengths_list[scale_idx]
            if reconstruction_noise_list is not None:
                noise_signal = reconstruction_noise_list[scale_idx]
            else:
                noise_signal = get_noise(params, (1, 1, n_samples))
                noise_signal = noise_signal * noise_amp

            if scale_idx == 0:
                prev_sig = torch.full(noise_signal.shape, 0, device=params.device, dtype=noise_signal.dtype)
            else:
                prev_sig = signal_padder(prev_sig)

            # pad noise with zeros, to match signal after filtering
            if reconstruction_noise_list is None:
                # reconstruction_noise is already padded
                noise_signal = signal_padder(noise_signal)
                if scale_idx == 0:
                    prev_sig = signal_padder(prev_sig)
        else:
            if scale_idx < condition["condition_scale_idx"]:
                continue
            elif scale_idx == condition["condition_scale_idx"]:
                prev_sig = resample_sig(params, condition["condition_signal"], condition['condition_fs'],
                                        params.fs_list[scale_idx]).expand(1, 1, -1)
            noise_signal = get_noise(params, prev_sig.shape[2]).expand(1, 1, -1)
            noise_signal = signal_padder(noise_signal)
            noise_signal = noise_signal * noise_amp
            prev_sig = signal_padder(prev_sig)

        # Generate this scale signal
        cur_sig = netG((noise_signal + prev_sig).detach(), prev_sig)

        if output_all_scales:
            signals_all_scales.append(torch.squeeze(cur_sig).detach().cpu().numpy())

        # Upsample for next scale
        if scale_idx < len(fs_list) - 1:
            up_sig = resample_sig(params, cur_sig, orig_fs=fs_list[scale_idx], target_fs=fs_list[scale_idx + 1])
            if up_sig.shape[2] > signals_lengths_list[scale_idx + 1]:
                assert abs(
                    up_sig.shape[2] > signals_lengths_list[scale_idx + 1]) < 20, 'Should not happen, check this!'
                up_sig = up_sig[:, :, :signals_lengths_list[scale_idx + 1]]
            elif up_sig.shape[2] < signals_lengths_list[scale_idx + 1]:
                assert abs(
                    up_sig.shape[2] < signals_lengths_list[scale_idx + 1]) < 20, 'Should not happen, check this!'
                up_sig = torch.cat(
                    (up_sig, up_sig.new_zeros(1, 1, signals_lengths_list[scale_idx + 1] - up_sig.shape[2])),
                    dim=2)
        else:
            up_sig = cur_sig
        prev_sig = up_sig
        prev_sig = prev_sig.detach()

        del up_sig, cur_sig, noise_signal, netG

    if output_all_scales:
        return signals_all_scales
    else:
        return prev_sig


def cast_general(x):
    if x.isdigit():  # int
        return (int(x))
    else:
        try:
            ret = float(x)  # float
            if ret % 1 == 0:
                ret = int(ret)  # int
            return ret
        except ValueError:  # str or bool
            if x == 'True':
                return True
            elif x == 'False':
                return False
            else:
                if x[0] == "'" and x[-1] == "'":
                    x = x[1:-1]
                return x


def params_from_log(path, gpu_num=0):
    fId = open(path, 'r')
    line = fId.readline()
    params = Params()
    while not line[:2] == '\n' and not line == '':
        if not '=' in line:
            line = fId.readline()
            continue
        if line.startswith('file_name'):
            args = line.split('=')
            file_name = args[1].strip('\n')[1:]
            params.file_name = file_name
            line = fId.readline()
            continue
        args = line.split()
        if len(args) < 3:
            setattr(params, args[0], '')
        elif len(args) > 3 or args[2][0] == '[':  # it's a list
            tmp = line.split('[')
            try:
                tmp2 = tmp[1].split(']')
                setattr(params, args[0], [cast_general(a) for a in tmp2[0].split(', ')])
            except:
                pass
        else:
            setattr(params, args[0], cast_general(args[2]))
        line = fId.readline()
    fId.close()
    params.is_cuda = True if torch.cuda.is_available() else False
    if params.is_cuda:
        torch.cuda.set_device(gpu_num)
        params.gpu_num = gpu_num
        params.device = torch.device("cuda:%d" % gpu_num)
    else:
        params.device = torch.device("cpu")
    params.noise_amp_list = noise_amp_list_from_log(path)
    try:
        params.dilation_factors = [int(i) for i in params.dilation_factors]
    except:
        params.dilation_factors = [2 ** i for i in range(params.num_layers)]
    params.fs_list = [int(i) for i in params.fs_list]
    params.inputs_lengths = [int(s) for s in params.inputs_lengths]
    return params


def noise_amp_list_from_log(path):
    fId = open(path, 'r')
    line = fId.readline()
    noise_amp_list = []
    while line:
        if line.startswith('noise_amp') and not line.startswith('noise_amp_factor'):
            args = line.split()
            noise_amp_list.append(float(args[1]))
        line = fId.readline()
    fId.close()
    return noise_amp_list


def override_params(params, params_override):
    for key in vars(params_override):
        setattr(params, key, getattr(params_override, key))
    return params


def generators_list_from_folder(params):
    generators_list = []
    n_generators = len(params.scales)
    for scale_idx in range(n_generators):
        params.hidden_channels = params.hidden_channels_init if scale_idx == 0 else int(
            params.hidden_channels_init * params.growing_hidden_channels_factor)
        params.current_fs = params.fs_list[scale_idx]
        netG = CAW.Generator(params).to(params.device)
        try:
            netG.load_state_dict(
                torch.load('%s/netGScale%d.pth' % (params.output_folder, scale_idx), map_location=params.device))
            netG = reset_grads(netG, False)
            netG.eval()
            generators_list.append(netG)
        except:
            netG = CAW.Generator(params).to(params.device)
            continue
    return generators_list


def write_signal(path, signal, fs, overwrite=False, subtype='PCM_16'):
    if signal is None:
        return
    if torch.is_tensor(signal):
        signal = signal.squeeze().detach().cpu().numpy()
    if not path.endswith('.wav'):
        path = path + '.wav'
    if not overwrite:
        if os.path.exists(path):
            files = glob.glob(path[:-4].replace('[Hz]', '[[]Hz[]]') + '*')
            path = path[:-4] + '_' + str(len(files)) + path[-4:]
    maxAmp = max(abs(signal.reshape(-1)))
    if maxAmp > 1:
        signal = signal / maxAmp  # normalize to avoid clipping
    sf.write(path, signal, fs, subtype=subtype)


def time_freq_stitch_by_fft(low_signal, high_signal, low_Fs, high_Fs, filt_file=None):
    factor = int(high_Fs / low_Fs)
    nFFT = len(high_signal)
    nFFT_low = len(low_signal)
    nFFT_orig = nFFT
    if nFFT / factor < nFFT_low:
        nFFT = nFFT_low * 4

    if not filt_file is None:
        f_id = open(filt_file)
        real_data = np.array([float(n) for n in f_id.readline().strip('\n').split()])
        imag_data = np.array([float(n) for n in f_id.readline().strip('\n').split()])
        f_id.close()
        Hlib = real_data + 1j * imag_data
        f = interpolate.interp1d(np.array([i / len(Hlib) for i in range(len(Hlib))]), Hlib, fill_value="extrapolate")
        H = f(np.array([i / nFFT for i in range(nFFT)]))
    else:
        H = 1 / factor

    padded_low = np.zeros(len(low_signal) * int(high_Fs / low_Fs))
    padded_low[::factor] = low_signal
    high_fft = fft(high_signal)
    # low_fft = fft(padded_low) * factor

    low_fft = fft(padded_low) / H

    stitch_idx = int(np.ceil(nFFT_low / 2))
    filt_half_len = int(nFFT / high_Fs * 200)
    stitch_filt = np.array([i / filt_half_len / 2 for i in range(filt_half_len * 2, -1, -1)])
    tmp = np.zeros((nFFT // 2,), dtype=complex)
    tmp[:stitch_idx - 2 * filt_half_len] = low_fft[:stitch_idx - 2 * filt_half_len]
    tmp[stitch_idx:] = high_fft[stitch_idx:nFFT // 2]
    tmp[stitch_idx - 2 * filt_half_len:stitch_idx + 1] = stitch_filt * low_fft[
                                                                       stitch_idx - 2 * filt_half_len:stitch_idx + 1] + np.flip(
        stitch_filt) * high_fft[stitch_idx - 2 * filt_half_len:stitch_idx + 1]
    R = np.concatenate((np.real(tmp), np.array([np.real(tmp[-1])]), np.flipud(np.real(tmp[1:]))))
    I = np.concatenate((np.imag(tmp), np.zeros(1, ), -np.flipud(np.imag(tmp[1:]))))
    out_fft = R + 1j * I
    out = np.real(ifft(out_fft))
    if nFFT_orig != nFFT:
        out = out[:nFFT_orig]
        print('Dimentions mismatch!')
    return out
