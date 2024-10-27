from matplotlib import pyplot as plt
from tqdm import tqdm
from evaluate import compute_score_table, get_k_rows
from plot import output_metrics_table
from utils import get_detectors_by_prop, get_short_svm_name
from variables import window_sizes, cells_per_block_list
from scipy.ndimage import uniform_filter1d  
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
# if __name__ == '__main__':
#     fig, ax = plt.subplots(figsize=(10, 6))
#     for block_size in cells_per_block_list:
#         with_block_size = get_detectors_by_prop('cells_per_block', block_size)
#         mcc_scores = get_k_rows('mcc', 'total', with_block_size, -1, sort=False)   
#         smoothed_mcc_scores = uniform_filter1d([m[1] for m in mcc_scores], size=5)
#         max_score_idx, max_score_val = max(enumerate([m[1] for m in mcc_scores]), key=lambda x: x[1])
#         ax.plot([m[1] for m in mcc_scores], label=f'Block Size: {block_size}', color=colors[cells_per_block_list.index(block_size)], alpha=0.3)
#         ax.plot(smoothed_mcc_scores, color=colors[cells_per_block_list.index(block_size)], linestyle='--')
#         # Plot the max score wit the score value denoted in the plot
#         ax.plot(
#             max_score_idx, 
#             max_score_val, 
#             'o', 
#             color=colors[cells_per_block_list.index(block_size)], 
#             markersize=4
#         )
#         ax.text(
#             max_score_idx+1, 
#             max_score_val, 
#             f'{max_score_val:.2f}', 
#             fontsize=9, 
#             # color=colors[window_sizes.index(window)]
#         )
#         # ax.plot(max_score_idx, max_score_val, 'o', color=colors[window_sizes.index(window)], markersize=4,)
    
#     ax.legend(loc='lower right')
#     ax.set_xlabel('Model Pair Index')
#     ax.set_ylabel('MCC Score')
#     ax.set_title('MCC Scores for Different Block Sizes')
#     plt.savefig('/Users/adamsam/repos/ee/Pedestrian-Detection/code/experiment/compare_windows_total.png', dpi=300)
#     plt.close()

    
if __name__ == '__main__':
    from variables import window_sizes
    dataset = 'caltech_30'
    for window_size in window_sizes:
        print(f"\nsTable for {dataset} with window size {window_size}")
        table = compute_score_table(dataset, window_size=window_size)
        
        for it in tqdm(range(0, len(table), 40)):
            sub_table = {get_short_svm_name(key): table[key] for key in list(table.keys())[it:it+40]}
            output_metrics_table(sub_table, custom_detector_names=True)
            plt.savefig(f'/Users/adamsam/repos/ee/Pedestrian-Detection/tables/{dataset}/{dataset}_{window_size}_{it}.png', dpi=300, bbox_inches='tight', transparent="True", pad_inches=0)
            plt.close()