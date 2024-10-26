import datetime
import json
import os
import tracemalloc

import numpy as np

from dataset import get_dataset_path
from evaluate import get_dimensions
from parameters import SVM_Parameters
from transform import hog_transform, grayscale_transform
from variables import iterate_model_parameters


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy data types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_measurements(time_map, memory_map, base_path='measurements'):
    """
    Save measurement results to files with timestamp

    Args:
        time_map: Dictionary of timing measurements
        memory_map: Dictionary of memory measurements
        base_path: Base directory for saving measurements

    Returns:
        tuple: Paths to saved time and memory files
    """
    # Create measurements directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Generate timestamp for unique filenames
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save time measurements
    time_path = os.path.join(base_path, f'time_measurements.json')
    with open(time_path, 'w') as f:
        json.dump(time_map, f, cls=NumpyEncoder, indent=2)

    # Save memory measurements
    memory_path = os.path.join(base_path, f'memory_measurements.json')
    with open(memory_path, 'w') as f:
        json.dump(memory_map, f, cls=NumpyEncoder, indent=2)

    return time_path, memory_path


def load_measurements(time_path, memory_path):
    """
    Load measurement results from files

    Args:
        time_path: Path to time measurements file
        memory_path: Path to memory measurements file

    Returns:
        tuple: (time_map, memory_map)
    """
    with open(time_path, 'r') as f:
        time_map = json.load(f)

    with open(memory_path, 'r') as f:
        memory_map = json.load(f)

    return time_map, memory_map


def get_latest_measurements(base_path='measurements'):
    """
    Load the most recent measurement files from the specified directory

    Args:
        base_path: Directory containing measurement files

    Returns:
        tuple: (time_map, memory_map) or (None, None) if no files found
    """
    if not os.path.exists(base_path):
        return None, None

    # Get all measurement files
    time_files = [f for f in os.listdir(base_path) if f.startswith('time_measurements_')]
    memory_files = [f for f in os.listdir(base_path) if f.startswith('memory_measurements_')]

    if not time_files or not memory_files:
        return None, None

    # Get most recent files
    latest_time = max(time_files)
    latest_memory = max(memory_files)

    return load_measurements(
        os.path.join(base_path, latest_time),
        os.path.join(base_path, latest_memory)
    )


def get_hog_computation_intensity():
    import time
    X_test = np.load(get_dataset_path(
        (128, 64),
        'test',
        'point',
        'INRIA'
    ))[:100]
    X_test_gray = grayscale_transform(X_test)
    dimension_time_map = {}
    dimension_memory_map = {}

    tracemalloc.start()

    def compute_intensity(svm_parameters: SVM_Parameters):
        nonlocal X_test_gray
        nonlocal dimension_time_map
        nonlocal dimension_memory_map

        if svm_parameters.window_size != (128, 64):
            return

        import gc
        import psutil
        gc.collect()

        start_time = time.perf_counter()
        hog_transform(X_test_gray, svm_parameters.hog_parameters)
        end_time = time.perf_counter()

        current, peak = tracemalloc.get_traced_memory()
        process = psutil.Process()

        dimensions = get_dimensions(svm_parameters.hog_parameters, (128, 64))
        dimension_time_map[dimensions] = end_time - start_time
        dimension_memory_map[dimensions] = {
            'peak_memory': peak / 1024 / 1024,
            'current_memory': current / 1024 / 1024,
            'ram_usage': process.memory_info().rss / 1024 / 1024,
            'ram_percent': process.memory_percent()
        }
    iterate_model_parameters(compute_intensity)

    tracemalloc.stop()
    save_measurements(dimension_time_map, dimension_memory_map)
    return dimension_time_map, dimension_memory_map