import torch

class Params(object):
    def __init__(self):
        ######################
        # Running Parameters #
        ######################
        self.start_time = 0
        self.segments_to_train = []
        self.min_length = 20
        self.max_length = 25
        self.plot_signals = False
        self.manual_random_seed = -1 # -1 for no setting
        self.plot_losses = False
        self.init_sample_rate = 16000
        self.fs_list = [320, 400, 500, 640, 800, 1000, 1280, 1600, 2000, 2500, 4000, 8000, 10000, 12000, 14400, 16000]
        self.run_mode = 'normal'
        self.speech = False
        self.set_first_scale_by_energy = True
        self.add_cond_noise = True
        self.min_energy_th = 0.0025  # minimum mean energy for first scale
        self.is_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.initial_noise_amp = 1
        self.noise_amp_factor = 0.01

        #####################
        # Losses Parameters #
        #####################
        self.lambda_grad = 0.01
        self.alpha1 = 10
        self.alpha2 = 1e-4
        self.multispec_loss_n_fft = (2048, 1024, 512)
        self.multispec_loss_hop_length = (240, 120, 50)
        self.multispec_loss_window_size = (1200, 600, 240)

        ###########################
        # Optimization Parameters #
        ###########################
        self.num_epochs = 2000
        self.learning_rate = 0.0015
        self.scheduler_lr_decay = 0.1
        self.beta1 = 0.5

        ####################
        # Model Parameters #
        ####################
        self.filter_size = 9
        self.num_layers = 8
        self.hidden_channels_init = 16
        self.growing_hidden_channels_factor = 6
