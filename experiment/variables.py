from parameters import SVM_Parameters, HOG_Parameters

window_sizes = [(100, 50), (128, 96), (128, 64), (112, 48)]
orientations = [9, 13, 18]
pixels_per_cell_list = [(4,4), (6,6), (8,8), (10,10)]
cells_per_block_list = [(1,1),(2,2), (3,3), (4,4)]
block_strides = [(1,1), (2,2), (3,3)]
holistic_derivative_masks = [True, False]

def iterate_model_parameters(callback: callable,cheap=None):
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
                            callback(svm_parameters)

def get_model_count(condition_callback=None,cheap=None):
    count = 0
    def increment_counter(svm_parameters):
        if callable(condition_callback) and not condition_callback(svm_parameters):
            return
        nonlocal count
        count += 1
    iterate_model_parameters(increment_counter,cheap)
    return count

