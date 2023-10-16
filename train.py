import argparse
import pickle
import os

from utils.transformer_model import TransformerModel

if __name__ == "__main__":

    # parser to get data path
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data', help='path to store dataset')
    args = parser.parse_args()

    # get data path from parser
    data_path = args.data_path

    # load the dataset
    with open(os.path.join(data_path, 'processed_training_data.pkl'), 'rb') as f:
        data = pickle.load(f)

    # instantiate the transformer model
    model = TransformerModel()

    # train the model
    model.train(data['train_latents'], data['train_frame_labels'],
            data['val_latents'], data['val_frame_labels'], path='./model_utils/transformer_weights.hdf5')