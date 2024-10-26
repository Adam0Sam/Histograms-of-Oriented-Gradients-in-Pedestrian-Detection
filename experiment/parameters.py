class HOG_Parameters:
    def __init__(self,
                 orientations,
                 pixels_per_cell,
                 cells_per_block,
                 block_stride,
                 holistic_derivative_mask,
                 block_norm='L2-Hys',
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

from parameters import SVM_Parameters, HOG_Parameters

window_sizes = [(100, 50), (128, 96), (128, 64), (112, 48)]
orientations = [9, 13, 18]
pixels_per_cell_list = [(4,4), (6,6), (8,8), (10,10)]
cells_per_block_list = [(1,1),(2,2), (3,3), (4,4)]
block_strides = [(1,1), (2,2), (3,3)]
holistic_derivative_masks = [True, False]

def iterate_model_parameters(cheap=None):
    def get_parameter_list():
        parameter_list = []
        for window_size in window_sizes:
            for orientation in orientations:
                for cell_size in pixels_per_cell_list:
                    for block_size in cells_per_block_list:
                        for block_stride in block_strides:
                            if block_stride[0] > block_size[0]:
                                continue
                            if cheap is not None:
                                if cheap and cell_size[0] >= block_stride[0] * 4:
                                    continue
                                if not cheap and cell_size[0] < block_stride[0] * 4:
                                    continue
                            for holistic_derivative_mask in holistic_derivative_masks:
                                hog_parameters = HOG_Parameters(
                                    orientations=orientation,
                                    pixels_per_cell=cell_size,
                                    cells_per_block=block_size,
                                    block_stride=block_stride,
                                    holistic_derivative_mask=holistic_derivative_mask,
                                    block_norm='L2-Hys'
                                )
                                svm_parameters = SVM_Parameters(
                                    window_size=window_size,
                                    hog_parameters=hog_parameters
                                )
                                parameter_list.append(svm_parameters)
        return parameter_list
    
    for svm_parameters in get_parameter_list():
        yield svm_parameters
    
        

def get_model_count(condition_callback=None,cheap=None):
    count = 0
    def increment_counter(svm_parameters):
        if callable(condition_callback) and not condition_callback(svm_parameters):
            return
        nonlocal count
        count += 1
    for svm_parameters in iterate_model_parameters(cheap):
        increment_counter(svm_parameters)
    return count

