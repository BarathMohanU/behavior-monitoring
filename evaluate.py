import argparse
import pickle
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from utils.transformer_model import TransformerModel
from utils.data_utils import extract_windows_features
from utils.logger import LoggerManager
from download_data import make_dir

# intialise logger
logger = LoggerManager.get_logger(log_file='log_files/evaluate.log')


def numpy_mode_ignore_zeros(arr):
    """
    Compute the mode of a non-zero elements in a NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.generic or np.signedinteger or np.floating
        The mode of the non-zero elements of the array, returned 
        as a NumPy scalar (int or float, depending on the dtype of the array).
    """
    try:
        arr = arr[arr != 0]
        values, counts = np.unique(arr, return_counts=True)
        max_count_index = np.argmax(counts)
        return values[max_count_index]
    except:
        return 0


def plot_confusion_matrix(y_true, y_pred, class_names, path):
    """
    Plot and save the confusion matrix for classification results.

    Parameters
    ----------
    y_true : list or np.ndarray
        True labels.
    y_pred : list or np.ndarray
        Predicted labels.
    class_names : list of str
        Names of the classes, used for labeling the x and y axes.
    path : str
        Path where the plot will be saved.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the plot
    plt.savefig(path, format='png', dpi=300)

    logger.info(f"The confusion matrix has been saved to {path}")


def save_classification_report(y_true, y_pred, target_names, path):
    """
    Generate and save a classification report to a specified path.

    Parameters
    ----------
    y_true : array-like
        True labels of the data. Should be one-dimensional where 
        each value corresponds to the true label of a sample.
    y_pred : array-like
        Predicted labels of the data. Should be one-dimensional and 
        the same length as `y_true`.
    target_names : list of str
        Display names for the labels in the report. The order should 
        correspond to that used in `y_true` and `y_pred`.
    path : str
        Path where the classification report will be saved. If the 
        directory does not exist, it will be created.
    """
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=target_names)
    
    # Display the classification report
    logger.info("Classification Report:")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Save the classification report to file
    with open(path, 'w') as f:
        f.write(report)
    
    logger.info(f"The classification report has been saved to {path}")


if __name__ == "__main__":

    # parser to get data path
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data', help='path to store dataset')
    args = parser.parse_args()

    # get data path from parser
    data_path = args.data_path

    # load the dataset
    with open(os.path.join(data_path, 'processed_frame_data.pkl'), 'rb') as f:
        data = pickle.load(f)

    # load the parameters for the model
    with open('./jsons/parameters.json', 'r') as f:
        params = json.load(f)

    frame_window = params['transformer_hyperparameters']['frame_window']
    frame_stride = params['transformer_hyperparameters']['frame_stride']

    # instantiate the transformer model and load weights
    model = TransformerModel()
    model.load('./model_utils/transformer_weights.hdf5')

    # preprocess the video and make predictions
    y_preds_full_video = []

    for record in data['test_latents']:
        window_data = extract_windows_features(record, window_size=frame_window, stride=frame_stride)
        pred = model.predict(window_data)
        pred = np.argmax(pred, axis=-1)
        y_preds_full_video.append(numpy_mode_ignore_zeros(pred))

    y_preds_full_video = np.array(y_preds_full_video)
    y_true_full_video = np.array(data['test_full_labels'])

    # load the processed window data
    with open(os.path.join(data_path, 'processed_training_data.pkl'), 'rb') as f:
        data_2s = pickle.load(f)

    # make predictions on the 2s window data
    y_preds_2s_window = model.predict(data_2s['test_latents'])
    y_preds_2s_window = np.argmax(y_preds_2s_window, -1)
    y_true_2s_window = data_2s['test_frame_labels']

    # compute and print the accuracies
    accuracy_full_video = np.mean(y_preds_full_video == y_true_full_video)
    accuracy_2s_window = np.mean(y_preds_2s_window == y_true_2s_window)

    logger.info(f"Accuracy on full videos: {accuracy_full_video}")
    logger.info(f"Accuracy on 2s windows: {accuracy_2s_window}")

    # path to save the results
    reults_path = './results/'
    make_dir(reults_path)

    # plot and save the confusion matrix
    try:
        plot_confusion_matrix(y_true_full_video, y_preds_full_video, ['None', 'Arm Flapping', 'Head Banging', 'Spinning'], os.path.join(reults_path, 'confusion_matrix_full_video.png'))
    except:
        plot_confusion_matrix(y_true_full_video, y_preds_full_video, ['Arm Flapping', 'Head Banging', 'Spinning'], os.path.join(reults_path, 'confusion_matrix_full_video.png'))
    plot_confusion_matrix(y_true_2s_window, y_preds_2s_window, ['None', 'Arm Flapping', 'Head Banging', 'Spinning'], os.path.join(reults_path, 'confusion_matrix_2s_window.png'))

    # save the classification report
    try:
        save_classification_report(y_true_full_video, y_preds_full_video, ['None', 'Arm Flapping', 'Head Banging', 'Spinning'], os.path.join(reults_path, 'classification_report_full_video.txt'))
    except:
        save_classification_report(y_true_full_video, y_preds_full_video, ['Arm Flapping', 'Head Banging', 'Spinning'], os.path.join(reults_path, 'classification_report_full_video.txt'))
    save_classification_report(y_true_2s_window, y_preds_2s_window, ['None', 'Arm Flapping', 'Head Banging', 'Spinning'], os.path.join(reults_path, 'classification_report_2s_window.txt'))