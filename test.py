import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import seaborn as sns

from utils.data_utils import DataProcessTest
from utils.transformer_model import TransformerModel
from download_data import make_dir

if __name__ == "__main__":

    # parser to get data path
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='../data/videos/v_ArmFlapping_09.mp4', help='path to video file to be analysed')
    args = parser.parse_args()

    # get data path from parser
    video_path = args.video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # initialise the data processing class
    data_precessor = DataProcessTest()

    # read the video, preprocess it and get the features for the transformer model
    features, starting_timestamps, ending_timestamps = data_precessor.preprocess(video_path)

    # load the transformer model
    model = TransformerModel()
    model.load('./model_utils/transformer_weights.hdf5')

    # make predictions
    y_pred = model.predict(features)
    y_pred = np.argmax(y_pred, -1)

    # processing the windows to make non-overlapping windows
    new_y_pred = [y_pred[0]]
    new_starting_timestamps = [starting_timestamps[0]]

    # Iterate over windows
    for i in range(1, len(y_pred)):
        # If y_pred changes, calculate the mid-point of the current window 
        # and append to new arrays
        if y_pred[i] != y_pred[i-1]:
            new_start = (starting_timestamps[i] + ending_timestamps[i]) / 2
            new_starting_timestamps.append(new_start)
            new_y_pred.append(y_pred[i])
            
    new_starting_timestamps.append(ending_timestamps[-1])

    # Convert lists to numpy arrays
    new_y_pred = np.array(new_y_pred)
    new_starting_timestamps = np.array(new_starting_timestamps)

    # Filter out small windows
    durations = np.diff(new_starting_timestamps)
    valid_bins = np.logical_or(np.logical_and(durations >= 5.0, new_y_pred == 0), np.logical_and(durations >= 1.0, new_y_pred != 0))

    # Apply the filter
    filtered_y_pred = new_y_pred[valid_bins]
    filtered_starting_timestamps = new_starting_timestamps[:-1][valid_bins]
    filtered_starting_timestamps[0] = 0

    # plot the results

    # color palette
    cp = sns.color_palette("pastel")

    ending_timestamp = ending_timestamps[-1]

    # Color and label mapping
    colors = {1: cp[0], 2: cp[1], 3: cp[2]}
    labels = {1: 'Arm Flapping', 2: 'Head Banging', 3: 'Spinning'}

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate through each bin and plot a bar
    for i, (start, y) in enumerate(zip(filtered_starting_timestamps, filtered_y_pred)):
        # Determine the end of the bin
        if i < len(filtered_starting_timestamps) - 1:
            end = filtered_starting_timestamps[i + 1]
        else:
            end = ending_timestamp
        
        # Only plot bars for y_pred 1, 2, or 3
        if y in {1, 2, 3}:
            ax.barh(y=0.5, width=end-start, left=start, height=1, color=colors[y])
            
    # Add legend, labels, title, etc.
    legend_patches = [mpatches.Patch(color=colors[y], label=labels[y]) for y in colors.keys()]

    # Add legend, labels, title, etc.
    ax.legend(handles=legend_patches, fontsize=15)
    ax.set_ylim([0, 1.4])
    ax.set_xlim([0, ending_timestamp])
    ax.set_xlabel('Time (seconds)', fontsize=15)
    ax.set_title(f"Video Title: {video_name}", fontsize=15)

    # Remove y-ticks and y-tick labels
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Remove left, top, and right spines
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # path to save the plot
    plots_path = "./plots/"
    make_dir(plots_path)

    # save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, video_name + '.png'), format='png', dpi=300)