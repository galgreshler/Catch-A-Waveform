import argparse

from generating import AudioGenerator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', help='Folder of trained model', type=str, required=True)
    parser.add_argument('--new', help='Use new noise for inpainting', default=False, action='store_true')

    args = parser.parse_args()

    audio_generator = AudioGenerator(args['input_folder'])
    audio_generator.inpaint(new_noise=args['new'])
