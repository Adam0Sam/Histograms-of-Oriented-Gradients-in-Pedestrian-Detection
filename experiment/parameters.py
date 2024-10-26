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
        return "orientations_" + str(self.orientations) + "_pixels_per_cell_" + str(
            self.pixels_per_cell) + "_cells_per_block_" + str(self.cells_per_block) + "_block_stride_" + str(
            self.block_stride) + "_block_norm_" + self.block_norm + "_holistic_derivative_mask_" + str(
            self.holistic_derivative_mask)


class SVM_Parameters:
    def __init__(self, hog_parameters: HOG_Parameters, window_size):
        self.hog_parameters = hog_parameters
        self.window_size = window_size
    def get_svm_name(self):
        return "svm_" + self.hog_parameters.get_hog_name() + "_window_" + str(self.window_size)
