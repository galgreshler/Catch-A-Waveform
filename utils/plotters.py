from bokeh.plotting import figure, output_file, show
from bokeh.palettes import Category20
from bokeh.layouts import column
import numpy as np
import numpy.fft as fft
import os
import torch


FIG_WIDTH = 1200


def plot(x, y=None, labels=None):
    p = figure(width=FIG_WIDTH)
    if y is None:
        y = x.copy()
        x = range(y.shape[0])
    if len(y.shape) > 1:
        for i in range(y.shape[1]):
            if labels is None:
                p.line(x, y[:, i], color=Category20[20][i % 20])
            else:
                p.line(x, y[:, i], color=Category20[20][i % 20], legend_label=labels[i])
            # p.scatter(x, y[:, i], color=Category20[20][i % 20])
        p.legend.click_policy = 'hide'
    else:
        p.line(x, y, color=Category20[20][0])
        # p.scatter(x, y, color=Category20[20][0])

    show(p)


def plot_losses(params, loss_vectors):
    # Plot losses in each scale
    output_file(os.path.join(params.output_folder, '../figures', 'losses.html'))
    p_vec = []
    for losses, fs in zip(loss_vectors, params.fs_list):
        p = figure(title='Losses @ %dHz' % fs)
        p.title.align = "center"
        p.width = FIG_WIDTH
        p.xaxis.axis_label = 'Epoch#'
        p.line(range(params.num_epoches), -losses['v_err_real'], legend_label='D(real)', color=Category20[20][0])
        p.line(range(params.num_epoches), losses['v_err_fake'], legend_label='D(fake)', color=Category20[20][1])
        p.line(range(params.num_epoches), losses['v_gp'], legend_label='Gradient Penalty', color=Category20[20][2])
        p.line(range(params.num_epoches), losses['v_rec_loss'], legend_label='Rec. Loss', color=Category20[20][3])
        p.legend.click_policy = "hide"
        p_vec.append(p)
    show(column(p_vec))


def plot_signal_time_freq(*args, Fs=16000, labels=None):
    if np.isscalar(Fs):
        Fs = np.ones(len(args)) * Fs
    p_time = figure(title="Signal in Time", width=FIG_WIDTH)
    p_freq = figure(title="Signal in Freq", width=FIG_WIDTH)
    for idx, signal in enumerate(args):
        if torch.is_tensor(signal):
            signal = np.array(signal.tolist())
        if signal.ndim > 1:
            signal = np.squeeze(signal)
        if signal.ndim > 1:
            n_signals = signal.shape[0]
            for idx_2 in range(n_signals):
                cur_signal = signal[idx_2, :]
                N = len(cur_signal)
                t_vec = [i / Fs[idx] for i in range(N)]
                fft_size = int(2 ** np.ceil(np.log2(len(cur_signal) / 2)))
                freq_grid = [f / fft_size * Fs[idx] / 2 for f in range(fft_size)]
                S = fft.rfft(cur_signal, (fft_size - 1) * 2)
                legned_str = 'sig' + str(idx_2) if labels is None else labels[idx_2]
                p_time.scatter(t_vec, cur_signal, color=Category20[20][idx_2 % 20], legend_label=legned_str)
                p_freq.scatter(freq_grid, 20 * np.log10(abs(S)), color=Category20[20][idx_2 % 20], legend_label=legned_str)
        else:
            N = len(signal)
            t_vec = [i / Fs[idx] for i in range(N)]
            fft_size = int(2 ** np.ceil(np.log2(len(signal) / 2)))
            freq_grid = [f / fft_size * Fs[idx] / 2 for f in range(fft_size)]
            S = fft.rfft(signal, (fft_size - 1) * 2)
            legned_str = 'sig'+str(idx) if labels is None else labels[idx]
            p_time.scatter(t_vec, signal, color=Category20[20][idx], legend_label=legned_str)
            p_freq.scatter(freq_grid, 20*np.log10(abs(S)), color=Category20[20][idx], legend_label=legned_str)
    p_time.legend.click_policy = 'hide'
    p_freq.legend.click_policy = 'hide'

    show(column([p_time, p_freq]))