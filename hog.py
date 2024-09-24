import numpy as np
from skimage.feature import _hoghistogram
from skimage.feature._hog import _hog_normalize_block
from _hog_utils import _central_hog_channel_gradient, _holistic_hog_channel_gradient
from skimage._shared import utils

class HOG_Parameters:
    def __init__(self,
                 orientations,
                 pixels_per_cell,
                 cells_per_block,
                 block_stride,
                 block_norm,
                 holistic_derivative_mask,
                 visualize=False):
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_stride = block_stride
        self.block_norm = block_norm
        self.holistic_derivative_mask = holistic_derivative_mask
        self.visualize = visualize
    def get_hog_name(self):
        return "orientations_" + str(self.orientations) + "_pixels_per_cell_" + str(self.pixels_per_cell) + "_cells_per_block_" + str(self.cells_per_block) + "_block_stride_" + str(self.block_stride) + "_block_norm_" + self.block_norm + "_holistic_derivative_mask_" + str(self.holistic_derivative_mask)


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

    g_row, g_col = _holistic_hog_channel_gradient(image) if hog_parameters.holistic_derivative_mask else _central_hog_channel_gradient(
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
