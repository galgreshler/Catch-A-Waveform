import argparse
import os
import librosa
from utils.utils import calc_snr, calc_lsd
from generating import AudioGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Folder of trained model', type=str, required=True)
    parser.add_argument('--lr_signal', help='Name of signal to perform bandwidth extension on', type=str, required=True)
    parser.add_argument('--filter_file',
                        help='Text file describing the anti-aliasing filter frequency response used for downsampling',
                        type=str, default=None)

    args = parser.parse_args()

    file_name = args.lr_signal.split('.')
    audio_generator = AudioGenerator(os.path.join('outputs', args.input_folder))
    if len(file_name) < 2:
        args.lr_signal = '.'.join([args.lr_signal, 'wav'])

    lr_signal, condition_fs = librosa.load(os.path.join('inputs', args.lr_signal), sr=None)
    norm_factor = abs(lr_signal).max()
    lr_signal = lr_signal / norm_factor

    condition = {'condition_signal': lr_signal, 'condition_fs': condition_fs, 'name': args.lr_signal.split('.')[0]}
    filter_file = None if args.filter_file is None else os.path.join('inputs', args.filter_file + '.txt')
    extended_signal = audio_generator.extend(condition, filter_file)
    # If high-resolution signal exist, use it to calculate snr and lsd of extended signal
    if os.path.exists(os.path.join('inputs', args.lr_signal.replace('_lr', '_hr'))):
        hr_signal, hr_fs = librosa.load(os.path.join('inputs', args.lr_signal.replace('_lr', '_hr')),
                                        sr=audio_generator.params.Fs)
        # The model is working on normalized signals, so we normalize the ground truth as well for snr calculation,
        # You may instead multiply extended_signal by norm_factor, in order to return to the original amplitudes.
        hr_signal = hr_signal / norm_factor
        snr = calc_snr(extended_signal, hr_signal)
        lsd = calc_lsd(extended_signal, hr_signal)
        print('SNR: %.2f[dB], LSD: %.2f\n' % (snr, lsd))
