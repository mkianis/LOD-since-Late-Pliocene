########################################################################################################################
########################################################################################################################
## loading the required packages

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import scipy
import matplotlib.pyplot as plt
from ismember import ismember

from tqdm import tqdm
import math
import os

# These are for reproducibility
np.random.seed(seed=2024)
torch.manual_seed(2024)

########################################################################################################################
########################################################################################################################

"""
This is the source code for PIDM algorithm. It is used to derive ∆LOD since Late Pliocene (3.6 Myr ago).
"""
########################################################################################################################
########################################################################################################################

def linear_beta_schedule(timesteps):
    """
    Creates a linear schedule for beta_t from 1e-4 to 0.02 over the number of timesteps.

    Args:
        timesteps (int): Number of diffusion steps.

    Returns:
        torch.Tensor: Tensor of betas.
    """

    return torch.linspace(1e-4, 0.02, timesteps)


########################################################################################################################
########################################################################################################################

def get_alpha_bar(betas):
    """
    Computes the cumulative product of (1 - beta_t) to get alpha_bar.

    Args:
        betas (torch.Tensor): Beta schedule tensor.

    Returns:
        torch.Tensor: Cumulative product of alphas.
    """

    alphas = 1. - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar


########################################################################################################################
########################################################################################################################

def extract(a, t, x_shape):
    """
    Extracts the relevant coefficients for the given timestep t and reshapes them.

    Args:
        a (torch.Tensor): Tensor of coefficients.
        t (torch.Tensor): Current timestep.
        x_shape (torch.Size): Shape of the input data.

    Returns:
        torch.Tensor: Extracted and reshaped coefficients.
    """

    out = a.gather(-1, t).float()
    return out.reshape(-1, *([1] * (len(x_shape) - 1)))


########################################################################################################################
########################################################################################################################

class TimeSeriesDiffusionModel(nn.Module):
    """
    A simple neural network for time series diffusion models.
    This model takes in a noisy time series and the current timestep and outputs the predicted noise.
    """

    def init(self, input_dim, hidden_dim, time_emb_dim):
        """
        Initializes the TimeSeriesDiffusionModel.

            Args:
                input_dim (int): Number of features in the input time series.
                hidden_dim (int): Number of hidden units in the network.
                time_emb_dim (int): Dimension of the timestep embedding.
            """

        super(TimeSeriesDiffusionModel, self).__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.Tanh(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Tanh(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Tanh(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Tanh(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.Tanh(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Noisy input time series of shape (batch_size, input_dim).
            t (torch.Tensor): Current timestep of shape (batch_size,).

        Returns:
            torch.Tensor: Predicted noise of shape (batch_size, input_dim).
        """
        t = t.unsqueeze(-1) / 1000.  # Normalize timestep
        time_emb = self.time_mlp(t)  # (batch_size, time_emb_dim)
        x_input = torch.cat([x, time_emb], dim=1)  # Concatenate along feature dimension
        return self.net(x_input)


########################################################################################################################
########################################################################################################################

class ConvLSTMCell(nn.Module):
    """
    the convolutional LSTM cell
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):
        """

        Args:
            in_channels: number of input channels. In our problem it is 1
            out_channels: number of output channels. In our problem it is 1
            kernel_size: the kernel size
            padding: whether to use padding or not
            activation: which activation function to use: either relu or tanh
            frame_size: what is the shape of the input images (such as (181, 361))
        """

        super(ConvLSTMCell, self).__init__()

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels,
            out_channels=4 * out_channels,
            kernel_size=kernel_size,
            padding=padding)

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.zeros(out_channels, *frame_size))
        self.W_co = nn.Parameter(torch.zeros(out_channels, *frame_size))
        self.W_cf = nn.Parameter(torch.zeros(out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev)
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev)

        # Current Cell output
        C = forget_gate * C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C)

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C


class ConvLSTM(nn.Module):
    """
    the convolutional LSTM implementation
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, activation, frame_size):
        """

        Args:
            in_channels: number of input channels. In our problem it is 1
            out_channels: number of output channels. In our problem it is 1
            kernel_size: the kernel size
            padding: whether to use padding or not
            activation: which activation function to use: either relu or tanh
            frame_size: what is the shape of the input images (such as (181, 361))
        """
        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convLSTMcell = ConvLSTMCell(in_channels, out_channels,
                                         kernel_size, padding, activation, frame_size)

    def forward(self, X):
        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, self.out_channels, seq_len,
                             height, width, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels,
                        height, width, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize Cell Input
        C = torch.zeros(batch_size, self.out_channels,
                        height, width, device='cuda' if torch.cuda.is_available() else 'cpu')

        # Unroll over time steps
        for time_step in range(seq_len):
            H, C = self.convLSTMcell(X[:, :, time_step], H, C)

            output[:, :, time_step] = H

        return output


class Seq2Seq(nn.Module):
    """
    sequence-to-sequence model to capture the temporal patterns in the data
    """

    def __init__(self, num_channels, num_kernels, kernel_size, padding,
                 activation, frame_size, num_layers, input_sequence_length, output_sequence_length):
        """

        Args:
            num_channels: number of input channels. In our problem it is 1
            num_kernels: how many kernels to use
            kernel_size: the size of each kernel
            padding: whether to use padding or not
            activation: which activation function to use (relu or tanh)
            frame_size: the size of the input images (such as (181, 361))
            num_layers: how many convLSTM layers to use
            input_sequence_length: the input sequence length
            output_sequence_length: how many time-steps to model (in our case 1)
        """
        super(Seq2Seq, self).__init__()

        self.sequential_mu = nn.Sequential()
        self.sequential_sigma = nn.Sequential()

        # Add First layer (Different in_channels than the rest)
        self.sequential_mu.add_module(
            "convlstm1_mu", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        self.sequential_sigma.add_module(
            "convlstm1_sigma", ConvLSTM(
                in_channels=num_channels, out_channels=num_kernels,
                kernel_size=kernel_size, padding=padding,
                activation=activation, frame_size=frame_size)
        )

        self.sequential_mu.add_module(
            "batchnorm1_mu", nn.BatchNorm3d(num_features=num_kernels)
        )

        self.sequential_sigma.add_module(
            "batchnorm1", nn.BatchNorm3d(num_features=num_kernels)
        )

        # Add rest of the layers
        for l in range(2, num_layers + 1):
            self.sequential_mu.add_module(
                f"convlstm{l}_mu", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size)
            )

            self.sequential_sigma.add_module(
                f"convlstm{l}_sigma", ConvLSTM(
                    in_channels=num_kernels, out_channels=num_kernels,
                    kernel_size=kernel_size, padding=padding,
                    activation=activation, frame_size=frame_size)
            )

            self.sequential_mu.add_module(
                f"batchnorm{l}_mu", nn.BatchNorm3d(num_features=num_kernels)
            )

            self.sequential_sigma.add_module(
                f"batchnorm{l}_sigma", nn.BatchNorm3d(num_features=num_kernels)
            )

        # Add Convolutional Layer to predict output frame
        self.conv_mu = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

        self.conv_sigma = nn.Conv2d(
            in_channels=num_kernels, out_channels=num_channels,
            kernel_size=kernel_size, padding=padding)

        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        self.relu_activation = nn.ReLU()

    def forward(self, X):
        # Forward propagation through all the layers
        mu = self.sequential_mu(X)
        # Return only the last output frame
        mu = self.conv_mu(mu[:, :, -1])
        mu = mu.repeat(self.output_sequence_length, 1, 1, 1, 1)

        # Forward propagation through all the layers
        sigma = self.sequential_sigma(X)
        # Return only the last output frame
        sigma = self.conv_sigma(sigma[:, :, -1])
        sigma = sigma.repeat(self.output_sequence_length, 1, 1, 1, 1)

        return self.relu_activation(mu), self.relu_activation(sigma)


########################################################################################################################
########################################################################################################################

class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time series data.
    """

    def init(self, data):
        """
        Initializes the dataset with the given time series data.

            Args:
                data (np.ndarray or torch.Tensor): Time series data of shape (num_samples, input_dim).
            """

        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()

        self.data = data

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.data.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the sample at the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            torch.Tensor: Time series sample.
        """
        return self.data[idx]


########################################################################################################################
########################################################################################################################

def normalize_uncertainties(u, a, b):
    """
    this function normalizes the prior information regarding uncertainty

    Args:
        u: unnormalized uncertainties
        a: lower bound of the normalized uncertainties
        b: upper bound of the normalized uncertainties

        note that 0<a<b<1

    Returns: u_norm: normalized uncertainties

    """

    if a <= 0 or b >= 1 or a >= b or b <= 0 or a >= 1:
        print("error! the normalized interval (a,b) should satisfy 0<a<b<1")
        u_norm = []
    else:
        x1 = min(u)
        x2 = max(u)
        u_norm = ((b - a) / (x2 - x1)) * u + (a * x2 - b * x1) / (x2 - x1)

    return u_norm


########################################################################################################################
########################################################################################################################

def compute_k2L(k_2L_tmp, time):
    """

    Args:
        k_2L_tmp: the time-dependent second-degree load Love numbers
        time: time

    Returns: k_2L, the second-degree load Love number at 'time'

    """
    T = k_2L_tmp[:, 0]
    time = -time * 1000
    k_2L = np.interp(time, T, k_2L_tmp[:, 1])
    return k_2L


########################################################################################################################
########################################################################################################################

def compute_L02(t, L):
    """

    Args:
        t: co-latitude
        L: loading function

    Returns: L20, the zonal coefficient of the spherical expansion of the loading function

    """
    t *= np.pi / 180
    Y20 = np.sqrt(5) / 2 * (3 * np.cos(t) ** 2 - 1)
    L20 = 1 / (4 * np.pi) * (np.pi / 180) ** 2 * np.sum(np.multiply(L, Y20, np.sin(t)))
    return L20


########################################################################################################################
########################################################################################################################

def compute_convolution(A, B):
    """

    Args:
        A: input 1
        B: input 2

    Returns: out, the convolution of input 1 and input 2

    """
    out = np.empty((B.shape[0], B.shape[1]))
    for i in range(B.shape[1]):
        out[:, i] = np.convolve(A, B[:, i], mode="same")
    return out


########################################################################################################################
########################################################################################################################

def compute_ocean_function(L, Earth_topography, cf, of):
    """

    Args:
        L: the loading function
        Earth_topography: the topography of the Earth at the respective time

    Returns: out, the time-dependent ocean function. Implementation from Kendall et al. 2005:
     https://doi.org/10.1111/j.1365-246X.2005.02553.x

    """

    if L.shape[0] == Earth_topography.shape[0] and L.shape[1] == Earth_topography.shape[1]:
        Earth_topography += L
    else:
        Earth_topography = L
    out = -0.5 * np.sign(Earth_topography) + 0.5
    out = 0.5 * np.sign(out - 0.6) + 0.5
    return cf, of


########################################################################################################################
########################################################################################################################
def cosd(x):
    """

    Args:
        x: argument in degrees

    Returns: the cos of the argument x

    """
    return np.cos(x * np.pi / 180)


def sind(x):
    """

    Args:
        x:  x: argument in degrees

    Returns: the sin of the argument x

    """
    return np.sin(x * np.pi / 180)


########################################################################################################################
########################################################################################################################
def plate_motion(PMM, cf, time):
    """

    Args:
        PMM: plate motion model, with rates in terms of mm/year. It has the same shape as cf
        cf: present day continental function
        time: time in year

    Returns: the cf_t, the continental function at time t in the past

    """
    cf_t = cf - PMM * time
    return cf_t


########################################################################################################################
########################################################################################################################

def load_deformation_coefficients(load_love_number_path):
    """

    Args:
        load_love_number_path: path of the data to read load Love numbers

    Returns: second-degree load Love numbers hprimel and kprimel (h'_L, k'_L)

    """
    hprimel = np.loadtxt(load_love_number_path, skiprows=14)
    hprimel = hprimel[:, 1:2]

    kprimel = -np.array(
        [0, 1, 0.310, 0.197, 0.133, 0.104, 0.090, 0.082, 0.076, 0.072, 0.069, 0.066, 0.064, 0.062, 0.060,
         0.058, 0.056, 0.055, 0.054, 0.052, 0.051, 0.050, 0.049, 0.048, 0.047, 0.046, 0.045, 0.044, 0.043,
         0.042, 0.041, 0.040, 0.040, 0.039, 0.038, 0.037, 0.037, 0.036, 0.035, 0.034, 0.034])

    return hprimel, kprimel


########################################################################################################################
########################################################################################################################

def compute_Gphi_Gu_legendre_kiani_fast(x, kprimel, hprimel, truncation_order):
    """

    Args:
        x:
        kprimel: second-degree load Love numbers kprimel (k'_L)
        hprimel: second-degree load Love numbers hprimel (k'_L)
        truncation_order: at which order to cut the Green's function expansion

    Returns: Gphi, GU, the non-dimensional Green’s functions

    """
    y0 = np.ones(x.shape);
    y1 = x
    Gphi = 0
    GU = 0
    for n in range(np.max(kprimel.shape) - 1):
        print(f"load Love number {n} / {truncation_order}")
        match n:
            case 0:
                Gphi = Gphi + (1 + kprimel[n]) * y0
                GU = GU + (hprimel[n]) * y0
            case 1:
                Gphi = Gphi + (1 + kprimel[n]) * y1
                GU = GU + (hprimel[n]) * y1
            case _:
                L_n = (2 * n - 1) / (n) * np.multiply(x, y1) - (n - 1) / (n) * y0
                y0 = y1
                y1 = L_n

                Gphi = Gphi + (1 + kprimel[n]) * L_n
                GU = GU + (hprimel[n]) * L_n

    for n in range(np.max(kprimel.shape), truncation_order - 1):
        print(f"load Love number {n} / {truncation_order}")
        L_n = (2 * n - 1) / (n) * np.multiply(x, y1) - (n - 1) / (n) * y0
        y0 = y1
        y1 = L_n

        GU = GU + (hprimel[n + 1]) * L_n

    return Gphi, GU

########################################################################################################################
########################################################################################################################

def compute_mean_std(m, s):

    data = np.loadtxt("data/calc_sl/lslp.txt")
    ME = np.mean(m, axis=1)
    ST = np.std(m, axis=1)**2+np.mean(s, axis=1)**2

    return data


########################################################################################################################
########################################################################################################################

def compute_legendre_coeff(f, phi, lamb, grid_resolution=1):
    """

    Args:
        f: spatio-temporal function
        phi: latitude
        lamb: longitude
        grid_resolution: the resolution of the fundamental grid

    Returns: the coefficients of the spherical harmonic expansion of the function f: L211, L212, L201. See the
     definitions in Adhikari et al. 2016 https://doi.org/10.5194/gmd-9-1087-2016

    """
    theta = 90 - phi
    l = 2
    m = 1

    normalizing_coeff = np.sqrt(2 * (2 * l + 1) * math.factorial(l - m) / math.factorial(l + m))
    P, _ = scipy.special.lpmn(2, 2, cosd(theta))
    Plm = P[:, 2, :]

    C = cosd(m * lamb)
    S = sind(m * lamb)

    dSurface = (grid_resolution ** 2) * sind(theta) * (np.pi / 180) ** 2

    Y211 = normalizing_coeff * np.multiply(Plm, C)
    Y212 = normalizing_coeff * np.multiply(Plm, S)

    L211 = (1 / (4 * np.pi)) * np.nansum(np.multiply(f, Y211, dSurface))
    L212 = (1 / (4 * np.pi)) * np.nansum(np.multiply(f, Y212, dSurface))
    l = 2
    m = 0

    normalizing_coeff = np.sqrt((2 - 1) * (2 * l + 1) * math.factorial(l - m) / math.factorial(l + m))

    Plm = P[:, 0, :]

    C = cosd(m * lamb);

    dSurface = (grid_resolution ** 2) * sind(theta) * (np.pi / 180) ** 2

    Y201 = normalizing_coeff * np.multiply(Plm, C)

    L201 = (1 / (4 * np.pi)) * np.sum(np.multiply(f, Y201, dSurface))

    return L211, L212, L201


########################################################################################################################
########################################################################################################################

def compute_Y2mn(phi, lamb, m, n):
    """

    Args:
        phi: latitude
        lamb: longitude
        m: degree
        n: order

    Returns: Y2mn, the degree m, order n, spherical harmonic function

    """
    theta = 90 - phi
    l = 2
    normalizing_coeff = np.sqrt(2 * (2 * l + 1) * math.factorial(l - m) / math.factorial(l + m));
    P, _ = scipy.special.lpmn(2, 2, cosd(theta))
    Plm = P[:, m, :]
    C = cosd(m * lamb)
    S = sind(m * lamb)
    if n == 1:
        n1 = 1
        n2 = 0
    elif n == 2:
        n2 = 1
        n1 = 0
    else:
        n1 = 0
        n2 = 0
    Y2mn = normalizing_coeff * np.multiply(Plm, (n1 * C + n2 * S))
    return Y2mn


########################################################################################################################
########################################################################################################################

def compute_Lambda_Ulambda(phi, lamb, m1, m2, m3, R_e, Omega, k2, h2, g):
    """

    Args:
        phi: latitude
        lamb: longitude
        m1: xp polar motion
        m2: -yp polar motion
        m3: LOD
        R_e: mean radius of the Earth
        Omega: mean rotation rate of the Earth
        k2: second degree Love numbers k2
        h2: second degree Love numbers h2
        g: Earth's gravitational constant

    Returns: Lambda, Ulambda: rotational potential and radial displacements of the solid Earth due to
     rotational potential

    """
    coeffs = []
    coeffs.append((1 / (6 * np.sqrt(5))) * Omega ** 2 * R_e ** 2 * (m1 ** 2 + m2 ** 2 - 2 * m3 ** 2 - 4 * m3))
    coeffs.append([0.])
    coeffs.append(-(1 / (np.sqrt(15))) * Omega ** 2 * R_e ** 2 * (m1 * (1 + m3)))
    coeffs.append(-(1 / (np.sqrt(15))) * Omega ** 2 * R_e ** 2 * (m2 * (1 + m3)))
    coeffs.append(-(1 / (np.sqrt(60))) * Omega ** 2 * R_e ** 2 * (m1 ** 2 - m2 ** 2))
    coeffs.append(-(1 / (np.sqrt(60))) * Omega ** 2 * R_e ** 2 * (2 * m1 * m2))

    coeffs = np.array(coeffs)

    Lambda = 0
    Ulambda = 0
    k = 1
    for m in range(2 + 1):
        for n in range(1, 2 + 1):
            Y2mn = compute_Y2mn(phi, lamb, m, n)
            Lambda = Lambda + (1 + k2) * coeffs[k] * Y2mn
            Ulambda = Ulambda + (h2 / g) * coeffs[k] * Y2mn
            k = k + 1

    return Lambda, Ulambda


########################################################################################################################
########################################################################################################################

def compute_Phi_UPhi(phi, lamb, g, rho_E, L, grid_resolution, Greens_function):
    """

    Args:
        phi: latitude
        lamb: longitude
        g: Earth's gravitational constant
        rho_E: mean density of the Earth
        L: loading function
        grid_resolution: the resolution of the fundamental grid for computations
        Greens_function: the Green's functions

    Returns: PHI, UPHI: gravitational potential and radial displacements of the solid Earth due to
     gravitational potential

    """
    theta = 90 - phi
    dSurface = (grid_resolution ** 2) * sind(phi) * (np.pi / 180) ** 2

    tmp_theta = np.reshape(theta, np.prod(theta.shape), 1)
    tmp_lambda = np.reshape(lamb, np.prod(lamb.shape), 1)

    r, _ = ismember(Greens_function[:, 1:2], tmp_theta)
    Greens_function = Greens_function[r, :]
    PHI = np.empty((phi.shape[0], phi.shape[1]))
    PHI[:] = np.nan
    UPHI = np.empty((phi.shape[0], phi.shape[1]))
    UPHI[:] = np.nan
    for i in range(phi.shape[0]):
        r, _ = ismember(Greens_function[:, 1:2], theta[i, 1])
        tmp_Green = Greens_function[r, :]

        for j in range(phi.shape[2]):

            match j:
                case 0:
                    x = tmp_lambda - lamb[i, j]
                    x[x < 0] = 360 + x[x < 0]
                    x = np.reshape(np.reshape(x, theta.shape[0], []).T, x.shape)
                    _, c = ismember(np.concatenate(tmp_Green(ismember(tmp_Green[:, 2:3], x)), x), tmp_Green[:, :3])
                case 2:
                    x = tmp_lambda - lamb(i, j)
                    x[x < 0] = 360 + x[x < 0]
                    x = np.reshape(np.reshape(x, theta.shape[0], []).T, x.shape)
                    _, c = ismember(np.concatenate(tmp_Green(ismember(tmp_Green[:, 2:3], x)), x), tmp_Green[:, :3])

            tmp_Green = tmp_Green[c, :]

            Gphi = np.reshape(tmp_Green[:, 3:4], theta.shape[1], []).T
            GU = np.reshape(tmp_Green[:, 4:5], theta.shpe[1], []).T

            PHI[i, j] = (3 * g / (4 * np.pi * rho_E)) * np.nansum(np.multiply(Gphi, L, dSurface))
            UPHI[i, j] = (3 / (4 * np.pi * rho_E)) * np.nansum(np.multiply(GU, L, dSurface))

    return PHI, UPHI


########################################################################################################################
########################################################################################################################

def compute_sea_level(phi, lamb, H, L, m1, m2, m3, R_e, Omega, k2, h2, g, rho_E, rho_O, rho_I, A_O, ocean_mask,
                      grid_resolution, Greens_function):
    """

    Args:
        phi: latitude
        lamb: longitude
        H: continental mass change
        L: loading function
        m1: xp polar motion
        m2: -yp polar motion
        m3: LOD
        R_e: mean radius of the Earth
        Omega: mean rotation rate of the Earth
        k2: second-degree Love numbers k_2
        h2: second-degree Love numbers h_2
        g: Earth's gravitational constant
        rho_E: mean density of the Earth
        rho_O: density of ocean water
        rho_I: density of ice
        A_O: surface area of the Earth's oceans
        ocean_mask: mask of the oceans
        grid_resolution: the resolution of the fundamental grid for computations
        Greens_function: Green's function

    Returns: S: the sea-level change function

    """
    theta = 90 - phi
    dSurface = (grid_resolution ** 2) * sind(theta) * (np.pi / 180) ** 2

    Lambda, Ulambda = compute_Lambda_Ulambda(phi, lamb, m1, m2, m3, R_e, Omega, k2, h2, g)

    PHI, UPHI = compute_Phi_UPhi(phi, lamb, g, rho_E, L, grid_resolution, Greens_function)

    U = UPHI + Ulambda

    Cbar = PHI + Lambda - g * U

    E = -((R_e ** 2) / (g * rho_O * A_O)) * np.nansum(g * rho_I * np.multiply(H, dSurface))
    C = -((R_e ** 2) / (g * rho_O * A_O)) * np.nansum(rho_O * Cbar * np.multiply(ocean_mask, dSurface))

    N = (1 / g) * (PHI + Lambda) + E + C

    S = N - U

    return S


########################################################################################################################
########################################################################################################################

def solve_SLE(lat, lon, H, S, cf, of, N_iter=100, truncation_order=10000,
              load_love_number_path="load_Love_numbers.txt"):
    """

    Args:
        lat: latitude
        lon: longitude
        H: continental mass change
        S: sea-level change
        cf: continental function
        of: ocean function
        N_iter: number of iterations to achieve the goal
        truncation_order: oder of truncation for spherical harmonic expansions
        load_love_number_path: path to read the load Love numbers from

    Returns: LOD

    """
    hprimel, kprimel = load_deformation_coefficients(load_love_number_path=load_love_number_path)
    counter = 0
    cos_alpha = []
    coord = []

    for t in np.arange(90 - max(max(lat)), 90 - min(min(lat)) + 1):
        print(f"generating grids for latitude {t}")
        for tp in np.arange(90 - max(max(lat)), 90 - min(min(lat)) + 1):
            for ld in np.arange(0, 360 + 1):
                cos_alpha.append(cosd(t) * cosd(tp) + sind(t) * sind(tp) * cosd(ld))
                coord.append([t, tp, ld])

    cos_alpha = np.array(cos_alpha)
    coord = np.array(coord)
    cos_alpha[cos_alpha > 1] = np.floor(cos_alpha[cos_alpha > 1])
    cos_alpha[cos_alpha < -1] = np.ceil(cos_alpha[cos_alpha < -1])

    Gphi, GU = compute_Gphi_Gu_legendre_kiani_fast(cos_alpha, kprimel, hprimel, truncation_order)
    Greens = np.concatenate((coord, Gphi, GU), axis=1)
    eps = 1e-4

    iter = 1
    Omega = 7.2921e-5
    g = 9.81
    A_O = 361e12
    init_m1 = 1e-20
    init_m2 = 1e-20
    init_m3 = 1e-20
    L = np.multiply(H, cf) + np.multiply(S, cf)
    while iter < N_iter:
        print(f"computations for the iteration {iter} / {N_iter}")
        L211, L212, L201 = compute_legendre_coeff(L, lat, lon, grid_resolution=1)
        R_e = 6371e3
        dL13 = -4 * np.pi * R_e ** 4 / np.sqrt(15) * L211
        dL23 = -4 * np.pi * R_e ** 4 / np.sqrt(15) * L212
        dL33 = -4 * np.pi * R_e ** 4 / np.sqrt(15) * 2 * L201 / np.sqrt(3)
        C_A = 2.6068e35
        C_earth = 8.0345e37
        rho_ice = 900.
        rho_water = 1000.
        rho_E = 5512.
        h2 = 0.6149
        k2 = 0.3055
        k2prime = -0.3
        ks = 0.942
        to_mas = 180 / np.pi * 3600 * 1000.
        to_ms = 86400 * 1000.
        coeff_solid_earth_deformation = ks / (ks - k2) * (1 + k2prime);
        LOD_coeff_solid_earth_deformation = (1 + k2prime) / (1 + (4 / 3) * (C_A / C_earth) * (k2 / ks));

        m1 = coeff_solid_earth_deformation * dL13 / C_A
        m2 = coeff_solid_earth_deformation * dL23 / C_A
        m3 = -LOD_coeff_solid_earth_deformation * dL33 / C_earth

        if abs(m1 - init_m1) / abs(init_m1) < eps and abs(m2 - init_m2) / abs(init_m2) < eps and abs(
                m3 - init_m3) / abs(init_m3) < eps:
            break
        else:
            init_m1 = m1
            init_m2 = m2
            init_m3 = m3
        grid_resolution = 1
        S = compute_sea_level(lat, lon, H, L, m1, m2, m3, R_e, Omega, k2, h2, g, rho_E, rho_water, rho_ice, A_O, of,
                              grid_resolution, Greens)
        iter = iter + 1

    return m3 * to_ms


########################################################################################################################
########################################################################################################################

def compute_LOD(SL_path, LN_path, cf_path, of_path, Earth_topography):
    """

    Args:
        SL_path: the path of the sea-level data
        LN_path: the path of the load Love numbers
        cf_path: the path to read the continental function from
        of_path: the path to read the ocean function from
        Earth_topography: the present-day topography of the Earth

    Returns: LOD and its uncertainty

    """
    GMSL = np.loadtxt(SL_path, skiprows=10)
    time = GMSL[:, 0:1]
    Q = abs(GMSL[:, 3:4] - GMSL[:, 2:3])
    GMSL = GMSL[:, 1:2] - GMSL[-1, 1:2]
    k_2L_tmp = np.loadtxt(LN_path, delimiter=',', skiprows=1)
    k_2L_tmp[:, 1:2] = k_2L_tmp[:, 1:2] - 0.07
    k_2L = compute_k2L(k_2L_tmp, time)
    of = np.loadtxt(of_path)
    cf = np.loadtxt(cf_path)
    of[of == 0] = np.nan
    cf[cf == 0] = np.nan
    t = np.linspace(0, 180, 181)
    l = np.linspace(-180, 180, 361)
    t, l = np.meshgrid(t, l)
    t, l = t.T, l.T
    rho_w = 1e3
    N = 100  ##number of simulations
    e = GMSL.shape[0]
    time = time[:e, :]
    L20 = np.empty((e, N))
    L20[:] = np.nan
    k_2L = k_2L[:e, :]
    for epoch in range(e):
        print(f"epoch {epoch} / {e}")
        for i in range(N):
            L = np.random.normal(loc=0., scale=Q[epoch], size=t.shape)
            H = np.multiply(L, cf)
            H = (H - np.nanmean(H) - GMSL[epoch]) * rho_w
            S = np.multiply(L, of)
            S = (S - np.nanmean(S) + GMSL[epoch]) * rho_w
            H[np.isnan(H)] = 0
            S[np.isnan(S)] = 0
            L = H + S
            L20[epoch, i] = compute_L02(t, L)
            cf1, of1 = compute_ocean_function(L, Earth_topography, cf, of)
            if epoch > e:
                cf = cf1
                of = of1

    R_e = 6371e3
    k2 = 0.3055
    k2prime = -0.3
    ks = 0.942
    to_ms = 86400 * 1e3
    C_A = 2.6068e35
    C_earth = 8.0345e37
    LOD_coeff_solid_earth_deformation = (1 + k2prime) / (1 + (4 / 3) * (C_A / C_earth) * (k2 / ks))
    LOD = solve_SLE(90 - t, l, H, S, cf, of, N_iter=100, truncation_order=10000, load_love_number_path=LN_path)
    LOD = - 8 * np.pi * R_e ** 4 / (C_earth * np.sqrt(45)) * LOD_coeff_solid_earth_deformation * L20 * to_ms
    L33 = -8 * np.pi * R_e ** 4 / (np.sqrt(45)) * L20
    k_2L_conv_L33 = compute_convolution(k_2L, L33)
    LOD = -1 / C_earth * (L33 + k_2L_conv_L33) * to_ms
    m = LOD.mean(axis=1)
    s = LOD.std(axis=1)

    return m, s


########################################################################################################################
########################################################################################################################
@torch.no_grad()
def sample_diffusion_model(model, betas, alpha_bar, device, input_shape, num_samples=100):
    """
    Samples from the trained diffusion model to generate multiple reconstructions for uncertainty quantification.

    Args:
        model (nn.Module): Trained diffusion model.
        betas (torch.Tensor): Beta schedule tensor.
        alpha_bar (torch.Tensor): Cumulative product of alphas.
        device (torch.device): Device to run the sampling on.
        input_shape (tuple): Shape of the input data (input_dim,).
        num_samples (int): Number of samples to generate.

    Returns:
        torch.Tensor: Generated samples of shape (num_samples, input_dim).
    """

    model.eval()
    betas = betas.to(device)
    alpha_bar = alpha_bar.to(device)
    T = len(betas)

    # Initialize with random noise
    x = torch.randn((num_samples, input_shape[0]), device=device)

    for t in reversed(range(1, T)):
        #print(t)
        t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
        a_bar = extract(alpha_bar, t_batch, x.shape)
        beta = betas[t]

        # Predict noise
        pred_noise = model(x, t_batch)

        # Compute the mean of q(x_{t-1} | x_t, x_0)
        x0_pred = (x - torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a_bar)

        mean = torch.sqrt(a_bar_prev := extract(alpha_bar, t_batch - 1, x.shape)) * x0_pred + \
               torch.sqrt(1 - a_bar_prev - (a_bar - a_bar_prev)) * pred_noise

        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(betas[t]) * noise
        else:
            x = mean

    return x


########################################################################################################################
########################################################################################################################

def compute_physical_loss(samples, t, SL_path, LN_path, cf_path, of_path, Earth_topography):
    """

    Args:
        samples: generated samples
        t: co-latitude
        SL_path: the path of the sea-level data
        LN_path: the path of the load Love numbers
        cf_path: the path to read the continental function from
        of_path: the path to read the ocean function from
        Earth_topography: the present-day topography of the Earth

    Returns: the physical loss function for the optimization of PIDM

    """
    # the formulation is viscoelastic, so we need to compute the whole thing
    m, s = compute_LOD(SL_path=SL_path, LN_path=LN_path, cf_path=cf_path, of_path=of_path,
                       Earth_topography=Earth_topography)

    loss = (m[t] - samples[0]) ** 2
    return loss


########################################################################################################################
########################################################################################################################

def train_diffusion_model(model, dataloader, betas, alpha_bar, device, SL_path, LN_path, cf_path, of_path,
                          Earth_topography, epochs=100, lr=1e-3):
    """
    Trains the diffusion model using the provided data.

    Args:
        model (nn.Module): The diffusion model to train.
        dataloader (DataLoader): DataLoader for the training data.
        betas (torch.Tensor): Beta schedule tensor.
        alpha_bar (torch.Tensor): Cumulative product of alphas.
        device (torch.device): Device to run the training on.
        epochs (int): Number of training epochs.
        lr (float): Learning rate for the optimizer.

    Returns:
        nn.Module: Trained model.
    """

    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    model.to(device)
    betas = betas.to(device)
    alpha_bar = alpha_bar.to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            batch = batch.to(device)
            batch_size, input_dim = batch.shape

            # Sample random timesteps for each sample in the batch
            t = torch.randint(0, len(betas), (batch_size,), device=device).long()

            # Get alpha_bar for current timesteps
            a_bar = extract(alpha_bar, t, batch.shape)
            #a_bar_prev = extract(alpha_bar, t - 1, batch.shape)

            # Sample noise
            noise = torch.randn_like(batch)

            # Compute the noisy sample
            noisy = torch.sqrt(a_bar) * batch + torch.sqrt(1 - a_bar) * noise

            # Predict the noise using the model
            pred_noise = model(noisy, t)

            # Compute physical loss
            samples = sample_diffusion_model(model, betas, alpha_bar, device, input_shape=(input_dim,),
                                             num_samples=1)
            loss_physics = compute_physical_loss(samples, t, SL_path=SL_path, LN_path=LN_path, cf_path=cf_path,
                                                 of_path=of_path,
                                                 Earth_topography=Earth_topography)

            # Compute loss
            loss = mse(pred_noise, noise) + loss_physics

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model


########################################################################################################################
########################################################################################################################

def diffusion_model_pipeline(time_series_data,
                             input_dim,
                             Earth_topography,
                             SL_path="data/calc_sl_a1/gm.txt",
                             LN_path="k2_love_numbers.txt",
                             of_path="of.txt",
                             cf_path="cf.txt",
                             hidden_dim=128,
                             time_emb_dim=32,
                             batch_size=64,
                             epochs=100,
                             lr=1e-3,
                             num_samples=100,
                             device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Complete pipeline to train and sample from a diffusion model on time series data.

    Args:
        time_series_data (np.ndarray or torch.Tensor): Time series data of shape (num_samples, input_dim).
        input_dim (int): Number of features in the input time series.
        hidden_dim (int, optional): Number of hidden units in the model. Defaults to 128.
        time_emb_dim (int, optional): Dimension of the timestep embedding. Defaults to 32.
        batch_size (int, optional): Batch size for training. Defaults to 64.
        epochs (int, optional): Number of training epochs. Defaults to 100.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        num_samples (int, optional): Number of samples to generate for uncertainty. Defaults to 100.
        device (str, optional): Device to use ('cuda' or 'cpu'). Defaults to 'cuda' if available.

    Returns:
        torch.Tensor: Generated samples of shape (num_samples, input_dim).
        nn.Module: Trained diffusion model.
    """
    m, s = compute_LOD(SL_path=SL_path, LN_path=LN_path, cf_path=cf_path, of_path=of_path,
                       Earth_topography=Earth_topography)
    # Prepare dataset and dataloader
    dataset = TimeSeriesDataset()
    dataset.init(time_series_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define diffusion parameters
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)
    alpha_bar = get_alpha_bar(betas)

    # Initialize model
    model = TimeSeriesDiffusionModel()
    model.init(input_dim=input_dim, hidden_dim=hidden_dim, time_emb_dim=time_emb_dim)

    # Train the model
    trained_model = train_diffusion_model(model, dataloader, betas, alpha_bar, device, epochs, lr,
                                          SL_path=SL_path, LN_path=LN_path, cf_path=cf_path, of_path=of_path,
                                          Earth_topography=Earth_topography)

    # Sample from the model
    samples = sample_diffusion_model(trained_model, betas, alpha_bar, device, input_shape=(input_dim,),
                                     num_samples=num_samples)

    samples.append(m)
    samples.append(s)
    return samples, trained_model


########################################################################################################################
########################################################################################################################
# running the code
# configurations
data = np.loadtxt("data/calc_sl/lslp.txt")
std = np.abs(data[:, -1])
Y = data[:, 1]
n = Y.shape[0]
#
num_samples = 50000
data = []
for _ in range(num_samples):
    noise = np.random.normal(0, std, size=n)
    series = Y + noise
    data.append(series)
data = np.array(data)  # Shape: (num_samples, time_steps)
Earth_topography = np.loadtxt("topo_bed.txt")  # Earth's topography, for the computation of ocean function
# Parameters
input_dim = data.shape[1]  # Number of time steps
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Run the diffusion model pipeline
samples, model = diffusion_model_pipeline(
    time_series_data=data,
    input_dim=input_dim,
    hidden_dim=32,
    time_emb_dim=32,
    batch_size=256,
    epochs=5000,  # epochs to train
    lr=5e-4,
    num_samples=50000,  # Generate num_samples samples for uncertainty
    device=device,
    Earth_topography=Earth_topography,
    SL_path="data/calc_sl_a1/gm.txt",
    LN_path="k2_love_numbers.txt",
    of_path="of.txt",
    cf_path="cf.txt"
)

m, s = samples[-2], samples[-1]

LOD_PIDM = compute_mean_std(m, s)
########################################################################################################################
########################################################################################################################
## save the results

folder_path = "output"

# Check if the folder exists
if not os.path.exists(folder_path):
    # Create the folder
    os.makedirs(folder_path)

np.savetxt("output/LOD_PIDM.txt", LOD_PIDM)

########################################################################################################################
########################################################################################################################

print("code successfully completed!")
