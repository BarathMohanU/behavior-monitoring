import torch
import clip
import json
import os
from PIL import Image
import cv2
import numpy as np
import warnings

warnings.filterwarnings("ignore")

class DataProcessTrain:
    """
    Class to process video data using CLIP model features.

    The `DataProcessing` class facilitates the pre-processing and feature extraction
    of video data using the OpenAI's CLIP model. The class allows reading videos, 
    computing image and latent features, assigning categories based on certain 
    keywords, and obtaining labels from annotations.
    
    Attributes
    ----------
    clip_variant : str
        Variant of the CLIP model to be used.
    data_path : str
        Path to the data directory.
    batch_size : int
        Batch size for processing frames.
    device : str
        Computational device - either "cuda" or "cpu".
    model : clip model
        Loaded CLIP model variant.
    preprocess : clip preprocess
        Preprocessing step for the CLIP model.
    annotations : dict
        Parsed annotations from the annotations.json file.
    text_features : tensor
        Encoded text features using the CLIP model.
    category_mapping : dict
        Mapping of category strings to numerical identifiers.
    state : State
        Inner `State` class instance holding video-specific information.
    """

    class State:
        """
        Inner class to hold state of the video being processed.

        Holds video-specific attributes such as the frames, fps, and video_path. 
        This class is utilized to encapsulate information about the video, and 
        instances of it can be reset for subsequent videos to be processed.
        
        Attributes
        ----------
        all_frames : tensor or None
            Tensor holding all preprocessed frames of the video, or None if no video is read.
        fps : int or None
            Frames per second of the read video, or None if no video is read.
        video_path : str or None
            Path of the read video, or None if no video is read.
        """
        def __init__(self):
            self.all_frames = None
            self.fps = None
            self.video_path = None


    def __init__(self, batch_size=256):
        """
        Initialize a `DataProcessing` instance.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for processing frames, by default 256.
        """

        self.batch_size = batch_size

        # load parameters from json
        with open('./jsons/parameters.json', 'r') as f:
            parameters = json.load(f)

        self.clip_variant = parameters['clip_variant']

        # select gpu if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load the model
        self.model, self.preprocess = clip.load(self.clip_variant, device=self.device)

        # load the json annotations
        with open('./jsons/annotations.json', 'r') as f:
            self.annotations = json.load(f)

        # load the list of prompts from the json file
        with open('./jsons/text_prompts.json', 'r') as f:
            self.text_list = json.load(f)

        # tokenize the text and compute text features
        text = clip.tokenize(self.text_list).to(self.device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(text)

        # define category mapping
        self.category_mapping = {"armflapping": 1, "headbanging": 2, "spinning": 3}

        # initialise state
        self.state = self.State()


    def read_video(self, video_path):
        """
        Read and preprocess frames from a video.

        Parameters
        ----------
        video_path : str
            Path to the video file to be read and processed.
        """

        self.state.video_path = video_path

        cap = cv2.VideoCapture(self.state.video_path)

        frames = []

        self.state.fps = cap.get(cv2.CAP_PROP_FPS)

        # Collect all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break  
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self.preprocess(Image.fromarray(frame_rgb)).unsqueeze(0)
            frames.append(image)

        self.state.all_frames = torch.cat(frames).to(self.device)

        cap.release()


    @staticmethod
    def min_max_norm(latent):
        """
        Normalize the input latent values using Min-Max normalization.

        The normalized values are scaled to the range [-1, 1].

        Parameters
        ----------
        latent : np.ndarray
            Input array to be normalized.

        Returns
        -------
        np.ndarray
            The normalized array.
        """
        max_ = np.max(latent)
        min_ = np.min(latent)
        norm_latent = (latent - min_) / (max_ - min_)
        norm_latent = norm_latent * 2 - 1
        
        return norm_latent


    def compute_raw_features(self):
        """
        Compute raw features from image frames using a pre-trained model.

        The method processes frames in batches to compute image features, and subsequently,
        obtains probabilities by matrix multiplication with text features.

        Returns
        -------
        np.ndarray
            Array of computed latent values, stacked vertically (along axis 0).
        """
        all_latents = []

        # Ensure computations are not tracked by autograd
        with torch.no_grad():
            num_batches = int(np.ceil(len(self.state.all_frames) / self.batch_size))
            
            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size
                
                # Extract a batch of frames
                batch_frames = self.state.all_frames[start_idx:end_idx]
                
                # Compute image features
                image_feature = self.model.encode_image(batch_frames)
                
                # Compute probabilities by performing a matrix multiplication
                logits_per_image = image_feature @ self.text_features.T
                latent = logits_per_image.cpu().numpy()
                
                # Store the computed probabilities
                all_latents.append(latent)

        # Stack all computed probability arrays vertically
        all_latents = np.vstack(all_latents)

        return all_latents


    def assign_category(self, input_string):
        """
        Assign a category based on the occurrence of a keyword in the input string.

        Parameters
        ----------
        input_string : str
            String to be checked for category keywords.

        Returns
        -------
        int
            Numerical identifier of the found category, 0 if not found.
        """
        for key, category in self.category_mapping.items():
            if key in input_string.lower():
                return category


    def get_timestamps(self):
        """
        Generate timestamps for all frames based on the frames per second (fps).

        Calculates timestamps by creating an array of sequential integers from 1 to
        the total number of frames, inclusive. This array is then divided by the
        frames per second (fps) value from the state, generating timestamps in seconds.

        Returns
        -------
        np.ndarray
            1D array of timestamps, each corresponding to a frame.
        """
        return np.arange(1, len(self.state.all_frames) + 1) / self.state.fps


    def get_labels(self):
        """
        Obtain labels for each frame and the full label of the video.

        Returns
        -------
        frame_labels : np.ndarray
            NumPy array containing a label per frame, derived from time-stamped annotations.
        full_label : int
            The full label assigned to the video based on its name.
        """

        # get the time stamp of each frame
        time_stamps = self.get_timestamps()

        # get the file name
        name = os.path.splitext(os.path.basename(self.state.video_path))[0]

        # initiliase frame labels
        frame_labels = np.zeros_like(time_stamps)

        data = self.annotations[name]['behaviours']['list']

        # iterate through annotation list and assign labels
        for item in data:
            category_value = self.category_mapping.get(item["category"], 0)
            times = item['time'].split(':') if ':' in item['time'] else item['time'].split('-')
            times = [float(item) for item in times]
            frame_labels[np.logical_and(time_stamps >= times[0], time_stamps <= times[1])] = category_value

        # assign the label for the whole video
        full_label = self.assign_category(name)

        return frame_labels, full_label
    

    def reset(self):
        """
        Reset the `state` attribute to a new instance of the inner `State` class.
        """
        self.state = self.State()


def extract_windows_features(array, window_size=48, stride=1):
    """
    Extract windowed features from a 2D array with a specified size and stride.

    Parameters
    ----------
    array : np.ndarray
        2D array from which windows of features are extracted.
    window_size : int, optional
        The size of each extracted window, by default 48.
    stride : int, optional
        The number of indices between the start of each window, by default 1.

    Returns
    -------
    np.ndarray
        3D array of windows with shape (num_windows, window_size, array.shape[1]).
    """
    windows = []
    for start in range(0, array.shape[0] - window_size + 1, stride):
        window = array[start:start + window_size, :]
        windows.append(window)
    return np.array(windows)


def numpy_mode(arr):
    """
    Compute the mode of a 1D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        1D array for which the mode is computed.

    Returns
    -------
    scalar
        The element that appears most frequently in `arr`.
    """
    values, counts = np.unique(arr, return_counts=True)
    max_count_index = np.argmax(counts)
    return values[max_count_index]


def extract_windows_and_find_mode(array, window_size=48, stride=1):
    """
    Extract windows of data and compute mode for each window.

    Parameters
    ----------
    array : np.ndarray
        1D array from which windows are extracted and mode is computed.
    window_size : int, optional
        The size of each window, by default 48.
    stride : int, optional
        The number of indices between the start of each window, by default 1.

    Returns
    -------
    np.ndarray
        1D array of modes for each window.
    """
    modes = []
    for start in range(0, len(array) - window_size + 1, stride):
        window = array[start:start + window_size]
        window_mode = numpy_mode(window)
        modes.append(window_mode)
    return np.array(modes)


def extract_windows_timestamps(time_stamps, window_size=48, stride=1):
    """
    Extract the starting and ending timestamps of windows from an array.

    Parameters
    ----------
    time_stamps : np.ndarray
        1D array from which window timestamps are extracted.
    window_size : int, optional
        The size of each window, by default 48.
    stride : int, optional
        The number of indices between the start of each window, by default 1.

    Returns
    -------
    tuple of np.ndarray
        Two 1D arrays containing starting and ending timestamps of windows.
    """
    starting_times = []
    ending_times = []
    for start in range(0, time_stamps.shape[0] - window_size + 1, stride):
        window = time_stamps[start:start + window_size]
        starting_times.append(window[0])
        ending_times.append(window[-1])
    return np.array(starting_times), np.array(ending_times)


def extract_windows(arrays, type='features', window_size=48, stride=1):
    """
    Extract windows from arrays, treating them as features or labels.

    Parameters
    ----------
    arrays : list of np.ndarray
        List of arrays from which windows are extracted.
    type : str, optional
        A string indicating whether to treat windows as 'features' or 'labels', by default 'features'.
    window_size : int, optional
        The size of each window, by default 48.
    stride : int, optional
        The number of indices between the start of each window, by default 1.

    Returns
    -------
    np.ndarray
        Concatenated windows extracted from input arrays.
    """
    windows = []
    for array in arrays:
        if type == 'features':
            windows.append(extract_windows_features(array, window_size=window_size, stride=stride))
        elif type == 'labels':
            windows.append(extract_windows_and_find_mode(array, window_size=window_size, stride=stride))
    return np.concatenate(windows, 0)


class DataProcessTest():
    """
    A class used to test data processing functionalities.

    ...

    Attributes
    ----------
    data_processing : DataProcessTrain
        An instance of the DataProcessTrain class for processing training data.
    params : dict
        Transformer hyperparameters loaded from a JSON file.

    Methods
    -------
    preprocess(video_path):
        Preprocesses video data and extracts latent windows and timestamps.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the DataProcessTest object.
        """
        self.data_processing = DataProcessTrain()

        with open('./jsons/parameters.json', 'r') as f:
            self.params = json.load(f)["transformer_hyperparameters"]

    def preprocess(self, video_path):
        """
        Preprocesses video data and extracts latent windows and timestamps.

        Parameters
        ----------
        video_path : str
            Path to the video file to be processed.

        Returns
        -------
        latent_windows : np.ndarray
            Windows of latent features extracted from the video.
        starting_times : np.ndarray
            Starting timestamps of each latent window.
        ending_times : np.ndarray
            Ending timestamps of each latent window.
        """
        # Read video and compute raw features using methods from `DataProcessTrain`
        self.data_processing.read_video(video_path)
        latents = self.data_processing.compute_raw_features()
        time_stamps = self.data_processing.get_timestamps()

        # Reset the state in the data_processing instance
        self.data_processing.reset()

        # Extract features and timestamps
        latent_windows = extract_windows_features(
            latents, 
            window_size=self.params['frame_window'], 
            stride=self.params['frame_stride']
        )

        starting_times, ending_times = extract_windows_timestamps(
            time_stamps, 
            window_size=self.params['frame_window'], 
            stride=self.params['frame_stride']
        )

        return latent_windows, starting_times, ending_times