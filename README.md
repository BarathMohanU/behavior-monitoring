
# Behavior Monitoring and Analysis

This repo contains code to download and analyse the [SSBD](https://rolandgoecke.net/research/datasets/ssbd/#:~:text=Dataset%20Files%3A%C2%A0ssbd%2Drelease%20(ZIP%2C%201.7MB)) dataset, by leveraging the [CLIP](https://github.com/openai/CLIP) model. A custom transformer architecture is trained to perform action classification on videos in this dataset. 

### Setting Up
First, clone this repository

```
git clone https://github.com/BarathMohanU/behavior-monitoring.git
cd behavior-monitoring
```
Next, setup miniconda or anaconda on your device and create an environment using the `yml` file provided. After creating the environment activate it.

```
conda env create -f environment.yml --name [env_name]
conda activate [env_name]
```

### Downloading the Dataset

Optionally download the dataset (provided in google drive with restricted access) and unzip it into any directory.

If you have not downloaded the data from drive (or do not have access), then you will need to run the script which will download it for you. You will need to provide a path for downloading the data into; the default for this is `../data/`.

```
python download_data.py --data_path "../data"
```

The list of videos that are age restricted will be stored in `./jsons/age_restricted_videos.json`, along with their dataset names. These will have to be manually downloaded and added to `../data/videos/`. This script will also jsonify the annotations provided in the dataset and store them in `./jsons/annotations`.

### Data Preparation

The data preparation can be run by using the following command:
```
python preprocess_data.py --data_path "../data/"
```
The dataset is prepared in the following manner:

1. The videos are read from one by one.
2. The videos are processed by passing each frame to the CLIP model:
  a. The `ViT-B/32` variant of clip is used due to its light-weight nature. The variant to be used is defined in `./jsons/parameters.json` and can be altered.
  b. The text prompts used are stored `./jsons/text_prompts.json`. A total of 26 prompts are used which could derive meaningful features for predicting arm flapping, head banging, or spinning.
  c. The dot product of the image features and text features is taken from the CLIP model. A min-max scaling is applied to this output rather than a softmax as the text prompts are not necessarily mutually exclusive.
3. The features of the videos are then split into 2 second sliding windows with a stide of 1 frame. Each window forms a single sample for the downstream network. The activity which takes place in the majority of the window is considered the label for the window.
4. Data preparation predominantly makes use of classes and functions defined in `./utils/data_utils.py`.
5. The processed data file, which is ready for training is stored in `../data/processed_training_data.pkl`.
6. The list of names of videos in the train-val-test split is stored in `./jsons/train_val_test_split.json`.

### Model Training

The idea here is that the CLIP model provides specific features about the static frames (like arms up, head down, etc). The transformer model leverages this static information and relates them across time-points to predict the action in motion.

The transformer model can be trained by using this command:

```
python train.py --data_path "../data/"
```


1. The transformer model is defined in `./utils/transformer_model.py` and its hyperparameters of the model are defined in `./jsons/parameters.json`.

2. The inputs to the model are of form `[batch_size, time, features]`. The output of the model is a softmax over four classes: `["No Action", "Arm Flapping", "Head Banging", "Spinning"]` denoting the probabilities of actions predominantly present in the 2s window.

3. The model uses basic transformer encoder blocks with self-attention to process the inputs. Then it applies a series of dense layers before flattening the time dimension to reduce dimensionality. Then it again applies a series of dense layers after flattening the latents. Then finally an output dense layer with 4 units predicts the labels.

4. The hyperparameters of this network are not tuned (due to time and hardware constraints) and are set using intuition. Batch normalization and dropout are used along with the dense layers. The activation function of choice is `ELU`.

5. The [focal crossentropy loss](https://arxiv.org/pdf/1708.02002.pdf) is used to combat class imbalance in the dataset without needing to undersample or oversample.

6. The weights of the model in best epoch in terms of loss on the validation set are selected. The best weights are stored in `./model_utils/transformer_weights.hdf5`.

### Evaluation

The model can be evaluated by running:

```
python evaluate.py --data_path "../data/"
```

Evaluation of the model can be done in two ways. The model can be used to predict the majority action in the entire video or in 2 sec windows of a video. The model is evaluated in both ways separately and the results (confusion matrix and classification report) are stored in `./results/`.

### Running on a new video

The model can be run on a new video to predict the action in it by running

```
python test.py --video_path [path_to_video]
```

This will save a `PNG` plot to `./plots/` with the same name as the video file showing which actions were recognised at what time-points.

An example is already saved using `v_ArmFlapping_09.mp4` at `./plots/v_ArmFlapping_09.png`.

Note that in its current state, the code only works with `MP4` video formats.

### Logs

Logs will be saved here: `./log_files/`
