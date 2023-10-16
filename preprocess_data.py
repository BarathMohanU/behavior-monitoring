import argparse
from glob import glob
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json
import numpy as np

from utils.data_utils import DataProcessTrain, extract_windows
from utils.logger import LoggerManager

# intialise logger
logger = LoggerManager.get_logger(log_file='log_files/preprocess_data.log')

if __name__ == "__main__":

    # parser to get data path
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data', help='path to store dataset')
    args = parser.parse_args()

    # get data path from parser
    data_path = args.data_path

    # obtain list of all video paths available
    video_paths = glob(os.path.join(data_path, 'videos', '*.mp4'))

    # initialise the data processing class
    data_processing = DataProcessTrain(batch_size=32)

    # iterate over all video paths and save the features and labels

    all_latents = []
    all_frame_labels = []
    all_full_labels = []
    all_file_names = []

    for video_path in tqdm(video_paths):

        file_name = os.path.basename(video_path)

        # read the video
        data_processing.read_video(video_path)

        # compute the features
        latent = data_processing.compute_raw_features()

        # get the labels
        frame_labels, full_label = data_processing.get_labels()
        data_processing.reset()

        # save the features
        all_latents.append(latent)
        all_frame_labels.append(frame_labels)
        all_full_labels.append(full_label)
        all_file_names.append(file_name)

        logger.info(f"Processed {file_name}")

    # train test split
    train_latents, test_latents, train_frame_labels, test_frame_labels, train_full_labels, test_full_labels, train_file_names, test_file_names = train_test_split(
        all_latents, all_frame_labels, all_full_labels, all_file_names, test_size=0.2, stratify=all_full_labels, random_state=42
    )

    # train val split
    train_latents, val_latents, train_frame_labels, val_frame_labels, train_full_labels, val_full_labels, train_file_names, val_file_names = train_test_split(
        train_latents, train_frame_labels, train_full_labels, train_file_names, test_size=0.2, stratify=train_full_labels, random_state=42
    )

    # save the data as a pickle file
    with open(os.path.join(data_path, 'processed_frame_data.pkl'), 'wb') as f:
        pickle.dump(
            {
                'train_latents': train_latents,
                'val_latents': val_latents,
                'test_latents': test_latents,
                'train_frame_labels': train_frame_labels,
                'val_frame_labels': val_frame_labels,
                'test_frame_labels': test_frame_labels,
                'train_full_labels': train_full_labels,
                'val_full_labels': val_full_labels,
                'test_full_labels': test_full_labels,
                'train_file_names': train_file_names,
                'val_file_names': val_file_names,
                'test_file_names': test_file_names,
            },
            f,
        )
              
    # save the train and test file names separately as a json
    with open('./jsons/train_val_test_split.json', 'w', encoding='utf-8') as f:
        json.dump(
            {
                'train_file_names': sorted(train_file_names),
                'val_file_names': sorted(val_file_names),
                'test_file_names': sorted(test_file_names),
            },
            f,
            ensure_ascii=False,
            indent=4
        )

    # save the input shapes in parameters json
    with open('./jsons/parameters.json', 'r') as f:
        params = json.load(f)

    params['transformer_hyperparameters']['input_shape'] = train_latents[0].shape[1]

    with open('./jsons/parameters.json', 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

    # compute sliding windows
    frame_window = params['transformer_hyperparameters']['frame_window']
    frame_stride = params['transformer_hyperparameters']['frame_stride']

    train_latents = extract_windows(train_latents, type='features', window_size=frame_window, stride=frame_stride)
    val_latents = extract_windows(val_latents, type='features', window_size=frame_window, stride=frame_stride)
    test_latents = extract_windows(test_latents, type='features', window_size=frame_window, stride=frame_stride)

    train_frame_labels = extract_windows(train_frame_labels, type='labels', window_size=frame_window, stride=frame_stride)
    val_frame_labels = extract_windows(val_frame_labels, type='labels', window_size=frame_window, stride=frame_stride)
    test_frame_labels = extract_windows(test_frame_labels, type='labels', window_size=frame_window, stride=frame_stride)

    # save the data as a pickle file
    with open(os.path.join(data_path, 'processed_training_data.pkl'), 'wb') as f:
        pickle.dump(
            {
                'train_latents': train_latents,
                'val_latents': val_latents,
                'test_latents': test_latents,
                'train_frame_labels': train_frame_labels,
                'val_frame_labels': val_frame_labels,
                'test_frame_labels': test_frame_labels,
                'train_full_labels': train_full_labels,
                'val_full_labels': val_full_labels,
                'test_full_labels': test_full_labels,
                'train_file_names': train_file_names,
                'val_file_names': val_file_names,
                'test_file_names': test_file_names,
            },
            f,
        )