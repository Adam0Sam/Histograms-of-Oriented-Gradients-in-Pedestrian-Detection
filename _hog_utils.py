import numpy as np
from skimage.feature._hog import _hog_channel_gradient

def _central_hog_channel_gradient(channel):
    return _hog_channel_gradient(channel)

def _holistic_hog_channel_gradient(channel):
    g_row = np.zeros(channel.shape, dtype=channel.dtype)
    g_col = np.zeros(channel.shape, dtype=channel.dtype)
    # forward difference
    g_row[0, :] = channel[1, :] - channel[0, :]
    g_col[:, 0] = channel[:, 1] - channel[:, 0]
    # backward difference
    g_row[-1, :] = channel[-1, :] - channel[-2, :]
    g_col[:, -1] = channel[:, -1] - channel[:, -2]
    # central difference
    g_row[1:-1, :] = (channel[2:, :] - channel[:-2, :])
    g_col[:, 1:-1] = (channel[:, 2:] - channel[:, :-2])

    return g_row, g_col