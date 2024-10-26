import numpy as np
from skimage.feature import _hoghistogram
from skimage.feature._hog import _hog_normalize_block, _hog_channel_gradient
from skimage._shared import utils
from tqdm import tqdm

from parameters import HOG_Parameters


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


def hog(
        image,
        hog_parameters: HOG_Parameters
):
    image = np.atleast_2d(image)
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if image.ndim != 2:
        raise ValueError(
            'Only images with two spatial dimensions are supported.'
        )

    g_row, g_col = _holistic_hog_channel_gradient(
        image) if hog_parameters.holistic_derivative_mask else _central_hog_channel_gradient(
        image)

    s_row, s_col = image.shape[:2]
    c_row, c_col = hog_parameters.pixels_per_cell
    b_row, b_col = hog_parameters.cells_per_block
    b_row_stride, b_col_stride = hog_parameters.block_stride

    n_cells_row = int(s_row // c_row)
    n_cells_col = int(s_col // c_col)

    orientation_histogram = np.zeros(
        (n_cells_row, n_cells_col, hog_parameters.orientations), dtype=float
    )
    g_row = g_row.astype(float, copy=False)
    g_col = g_col.astype(float, copy=False)

    _hoghistogram.hog_histograms(
        g_col,
        g_row,
        c_col,
        c_row,
        s_col,
        s_row,
        n_cells_col,
        n_cells_row,
        hog_parameters.orientations,
        orientation_histogram,
    )

    # now compute the histogram for each cell
    hog_image = None

    if hog_parameters.visualize:
        from .. import draw

        radius = min(c_row, c_col) // 2 - 1
        orientations_arr = np.arange(hog_parameters.orientations)
        # set dr_arr, dc_arr to correspond to midpoints of orientation bins
        orientation_bin_midpoints = np.pi * (orientations_arr + 0.5) / hog_parameters.orientations
        dr_arr = radius * np.sin(orientation_bin_midpoints)
        dc_arr = radius * np.cos(orientation_bin_midpoints)
        hog_image = np.zeros((s_row, s_col), dtype=float_dtype)
        for r in range(n_cells_row):
            for c in range(n_cells_col):
                for o, dr, dc in zip(orientations_arr, dr_arr, dc_arr):
                    centre = tuple([r * c_row + c_row // 2, c * c_col + c_col // 2])
                    rr, cc = draw.line(
                        int(centre[0] - dc),
                        int(centre[1] + dr),
                        int(centre[0] + dc),
                        int(centre[1] - dr),
                    )
                    hog_image[rr, cc] += orientation_histogram[r, c, o]

    n_blocks_row = (n_cells_row - b_row) + 1
    n_blocks_col = (n_cells_col - b_col) + 1
    if n_blocks_col <= 0 or n_blocks_row <= 0:
        min_row = b_row * c_row
        min_col = b_col * c_col
        raise ValueError(
            'The input image is too small given the values of '
            'pixels_per_cell and cells_per_block. '
            'It should have at least: '
            f'{min_row} rows and {min_col} cols.'
        )
    normalized_blocks = np.zeros(
        (n_blocks_row, n_blocks_col, b_row, b_col, hog_parameters.orientations), dtype=float_dtype
    )

    for r in range(0, n_blocks_row, b_row_stride):
        for c in range(0, n_blocks_col, b_col_stride):
            block = orientation_histogram[r: r + b_row, c: c + b_col, :]
            normalized_blocks[r, c, :] = _hog_normalize_block(block, method=hog_parameters.block_norm)
    normalized_blocks = normalized_blocks.ravel()

    if hog_parameters.visualize:
        return normalized_blocks, hog_image
    else:
        return normalized_blocks


def m_hog(
        image,
        hog_parameters: HOG_Parameters
):
    """
    Notes:
    ------
    - Even though this implementation is more theoretically accurate to the original HOG paper, as the block strides directly determine the dimensionality of the output feature vector, as opposed to the hog() function above, where block strides simply result in some vectors having zero values, the hog() function, on average, is more accuracte than m_hog() in pedestrian detection. I have no idea why.
    """
    image = np.atleast_2d(image)
    float_dtype = utils._supported_float_type(image.dtype)
    image = image.astype(float_dtype, copy=False)

    if image.ndim != 2:
        raise ValueError(
            'Only images with two spatial dimensions are supported.'
        )

    g_row, g_col = _holistic_hog_channel_gradient(
        image) if hog_parameters.holistic_derivative_mask else _central_hog_channel_gradient(
        image)

    s_row, s_col = image.shape[:2]
    c_row, c_col = hog_parameters.pixels_per_cell
    b_row, b_col = hog_parameters.cells_per_block
    b_row_stride, b_col_stride = hog_parameters.block_stride

    n_cells_row = int(s_row // c_row)
    n_cells_col = int(s_col // c_col)

    orientation_histogram = np.zeros(
        (n_cells_row, n_cells_col, hog_parameters.orientations), dtype=float
    )
    g_row = g_row.astype(float, copy=False)
    g_col = g_col.astype(float, copy=False)

    _hoghistogram.hog_histograms(
        g_col,
        g_row,
        c_col,
        c_row,
        s_col,
        s_row,
        n_cells_col,
        n_cells_row,
        hog_parameters.orientations,
        orientation_histogram,
    )

    n_blocks_row = (s_row - (b_row + 1) * c_row) // (b_row_stride * c_row)
    n_blocks_col = (s_col - (b_col + 1) * c_col) // (b_col_stride * c_col)
    if n_blocks_col <= 0 or n_blocks_row <= 0:
        min_row = b_row * c_row
        min_col = b_col * c_col
        raise ValueError(
            'The input image is too small given the values of '
             'pixels_per_cell and cells_per_block. '
            'It should have at least: '
            f'{min_row} rows and {min_col} cols.'
        )
    normalized_blocks = np.zeros(
        (n_blocks_row, n_blocks_col, b_row, b_col, hog_parameters.orientations), dtype=float_dtype
    )

    for r in range(0, n_blocks_row):
        for c in range(0, n_blocks_col):
            block = orientation_histogram[
                    r * b_row_stride: r * b_row_stride + b_row,
                    c * b_col_stride: c * b_col_stride + b_col,
                    :
                    ]
            normalized_blocks[r, c, :] = _hog_normalize_block(block, method=hog_parameters.block_norm)
    normalized_blocks = normalized_blocks.ravel()

    return normalized_blocks