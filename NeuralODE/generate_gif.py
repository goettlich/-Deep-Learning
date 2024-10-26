import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
from tqdm import tqdm

def generate_gif(result, model_names, log_every, exp_dir, gif_filename="prediction_progress.gif"):

    # convert tensors to np.arrays
    ground_truth = np.array(result['ground_truth'])
    for mname in model_names:
        result[mname] = [np.array(r) for r in result[mname]]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), sharex=True, sharey=True)

    # axis limits set by ground truth of larges sample in test batch, result (x,y)
    min_value = np.min(ground_truth, axis=(0,1))
    max_value = np.max(ground_truth, axis=(0,1))
    writer = PillowWriter(fps=5)
    
    num_frames = len(result[model_names[0]])
    
    with writer.saving(fig, os.path.join(exp_dir, gif_filename), dpi=80):
        with tqdm(total=num_frames) as pbar:
            
            for i in range(num_frames):

                pbar.set_description(f'processing frame {i}/{num_frames} (iteration {log_every * i}) ...')

                for j, ax in enumerate(axs.flatten()):
                    ax.clear()
                    for mname in model_names:
                        ax.plot(*(np.array(result[mname][i][j, :, :]).T), label=mname)
                    ax.plot(*(np.array(ground_truth[j, :, :]).T), label='GT')
                    ax.set_xlim(min_value[0], max_value[0])
                    ax.set_ylim(min_value[1], max_value[1])

                # Create a figure-wide legend
                handles, labels = axs[0,0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', ncol=2, fontsize=28)

                # Add iteration info as a title
                fig.suptitle(f"Iteration {(i + 1)*log_every}", fontsize=30, x=0.1, ha='left')
                
                # Save frame
                writer.grab_frame()

                pbar.update()

    plt.close(fig)
    print(f"GIF saved to {os.path.join(exp_dir, gif_filename)}")