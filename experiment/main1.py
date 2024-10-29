from matplotlib import pyplot as plt
from tqdm import tqdm
from evaluate import compute_score_table, get_k_rows
from plot import output_metrics_table
from utils import get_detectors_by_prop, get_short_svm_name
from variables import window_sizes, pixels_per_cell_list
from dataset import datasets, dataset_name_map
from scipy.ndimage import uniform_filter1d  
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

hdm_label_map = {
    True: 'With HDM',
    False: 'Without HDM'
}
if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 6))
    average_score_map = {}
    for pixels_per_cell in pixels_per_cell_list:
        with_parameter = get_detectors_by_prop('pixels_per_cell', pixels_per_cell)
        mcc_scores = get_k_rows('mcc', 'PnPLO', with_parameter, -1, sort=False)   
        average_score_map[pixels_per_cell] = sum([m[1] for m in mcc_scores]) / len(mcc_scores)
        smoothed_mcc_scores = uniform_filter1d([m[1] for m in mcc_scores], size=5)
        max_score_idx, max_score_val = max(enumerate([m[1] for m in mcc_scores]), key=lambda x: x[1])
        ax.plot([m[1] for m in mcc_scores], label=f'Cell Size {pixels_per_cell}', color=colors[pixels_per_cell_list.index(pixels_per_cell)], alpha=0.2)
        ax.plot(smoothed_mcc_scores, color=colors[pixels_per_cell_list.index(pixels_per_cell)], linestyle='--')
        # Plot the max score wit the score value denoted in the plot
        ax.plot(
            max_score_idx, 
            max_score_val, 
            'o', 
            color=colors[pixels_per_cell_list.index(pixels_per_cell)], 
            markersize=4
        )
        ax.text(
            max_score_idx+1, 
            max_score_val, 
            f'{max_score_val:.2f}', 
            fontsize=9, 
            # color=colors[window_sizes.index(window)]
        )
        # ax.plot(max_score_idx, max_score_val, 'o', color=colors[window_sizes.index(window)], markersize=4,)
    ax.legend(loc='lower right')
    ax.set_xlabel('Model Pair Index')
    ax.set_ylabel('MCC Score')
    ax.set_title(f'General MCC Scores for Different Window Size')
    plt.savefig(f'/Users/adamsam/repos/ee/Pedestrian-Detection/code/experiment/mcc_cell_size_PnPLO.png', dpi=300)
    print(average_score_map)

    
