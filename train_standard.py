import gc
import os
import pickle
from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from keras.optimizers.legacy import Adam
from keras.src.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim, mean_squared_error
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split, KFold
from tensorflow.python.ops.numpy_ops import np_config
from tqdm import tqdm

from models_standard import *

to_client = []  # List of file names that can be transferred to client


def load_data(topomaps_folder: str, labels_folder: str, test_size, anomaly_detection, random_state):
    x, y = _create_dataset(topomaps_folder, labels_folder)

    print(f"Splitting data set into {1 - test_size} training set and {test_size} test set "
          f"{'for latent space analysis' if not anomaly_detection else 'for anomaly detection'}")

    if anomaly_detection:
        # Training set only contains images whose label is 0
        train_indices = np.where(y == 0)[0]
        x_train = x[train_indices]
        y_train = y[train_indices]

        # Split the remaining data into testing sets
        remaining_indices = np.where(y != 0)[0]
        x_remaining = x[remaining_indices]
        y_remaining = y[remaining_indices]
        _, x_test, _, y_test = train_test_split(x_remaining, y_remaining, test_size=test_size,
                                                random_state=random_state)

        # Check dataset for anomaly detection task
        y_train_only_contains_label_0 = all(y_train) == 0
        y_test_only_contains_label_1_and_2 = all(label in [0, 1, 2] for label in y_test)
        if not y_train_only_contains_label_0 or not y_test_only_contains_label_1_and_2:
            raise Exception("Data was not loaded successfully")
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    return x_train, x_test, y_train, y_test


def _create_dataset(topomaps_folder, labels_folder):
    topomaps_files = os.listdir(topomaps_folder)
    labels_files = os.listdir(labels_folder)

    topomaps_files.sort()
    labels_files.sort()

    x = []
    y = []

    n_files = len(topomaps_files)

    for topomaps_file, labels_file in tqdm(zip(topomaps_files, labels_files), total=n_files,
                                           desc=f"Loading data set from {topomaps_folder} and {labels_folder} folders"):
        if topomaps_file.endswith('.DS_Store'):
            continue
        topomaps_array = np.load(f"{topomaps_folder}/{topomaps_file}", allow_pickle=True)
        labels_array = np.load(f"{labels_folder}/{labels_file}", allow_pickle=True)
        if topomaps_array.shape[0] != labels_array.shape[0]:
            raise Exception("Shapes must be equal")
        for i in range(topomaps_array.shape[0]):
            x.append(topomaps_array[i])
            y.append(labels_array[i])

    x = np.array(x)
    y = np.array(y)

    return x, y


def scaled_ssim(original, reconstructed):
    # data_range=1 requires the data to be normalized between 0 and 1
    original = normalize(original)
    reconstructed = normalize(reconstructed)
    score = ssim(original, reconstructed, data_range=1, channel_axis=-1)
    return (score + 1) / 2  # The reference paper deals with ssim between 0 and 1 instead of -1 and 1


def reduce_size(x, y, new_size):
    return x[:new_size], y[:new_size]


def expand(x):
    return np.expand_dims(x, -1).astype("float32")


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def custom_grid_search(x_train, latent_dimension):
    param_grid = {
        'epochs': [2500],
        'l_rate': [10 ** -5],  # Con gli AE ho notato che questo è il valore ottimale sempre
        'batch_size': [32],  # Come sopra
        'patience': [30]
    }

    print("\nlatent_dimension:", latent_dimension)
    print("Custom grid search with param_grid:", param_grid)

    grid_search = CustomGridSearchCV(param_grid)
    grid_search.fit(x_train, latent_dimension)

    return grid_search


def refit(fitted_grid, x_train, y_train, latent_dimension):
    print("\nRefitting based on best params:", fitted_grid.best_params_)

    best_epochs = fitted_grid.best_params_["epochs"]
    best_l_rate = fitted_grid.best_params_["l_rate"]
    best_batch_size = fitted_grid.best_params_["batch_size"]
    best_patience = fitted_grid.best_params_["patience"]

    val_size = 0.2
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size)
    print(f"validation data is {val_size} of training data")
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)

    encoder = EncoderStandard(latent_dimension)
    decoder = DecoderStandard()
    autoencoder = Autoencoder(encoder, decoder, best_epochs, best_l_rate, best_batch_size, best_patience)
    autoencoder.compile(Adam(best_l_rate))

    early_stopping = EarlyStopping("val_loss", patience=best_patience, verbose=1)

    history = autoencoder.fit(x_train, x_train, best_batch_size, best_epochs,
                              validation_data=(x_val, x_val), callbacks=[early_stopping], verbose=1)

    # dbg
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    ssim_scores = []
    mse_scores = []
    for train_idx, val_idx in cv.split(x_train):
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        predicted = autoencoder.predict(x_val_fold)
        ssim = scaled_ssim(x_val_fold, predicted)
        mse = mean_squared_error(x_val_fold, predicted)
        ssim_scores.append(ssim)
        mse_scores.append(mse)
    avg_ssim = np.mean(ssim_scores)
    avg_mse = np.mean(mse_scores)
    avg_score = avg_ssim / (avg_mse + 1)
    print(f"[dbg] avg_ssim for best combination after fit: {avg_ssim:.4f}")
    print(f"[dbg] avg_mse for best combination after fit: {avg_mse:.4f}")
    print(f"[dbg] avg_score for best combination after fit: {avg_score:.4f}")

    return history, autoencoder


class CustomGridSearchCV:
    def __init__(self, param_grid):
        self.param_grid = param_grid
        self.best_params_ = {}
        self.best_score_ = None
        self.grid_ = []

    def fit(self, x_train, latent_dimension):
        param_combinations = list(product(*self.param_grid.values()))
        n_combinations = len(param_combinations)

        n_splits = 5
        print("n_splits:", n_splits)
        print(f"scorers: {scaled_ssim} and {mean_squared_error}")

        for params in tqdm(param_combinations, total=n_combinations, desc="Combination", unit="combination"):
            params_dict = dict(zip(self.param_grid.keys(), params))

            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            ssim_scores = []
            mse_scores = []

            for train_idx, val_idx in cv.split(x_train):
                x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]

                encoder = EncoderStandard(latent_dimension)
                decoder = DecoderStandard()
                autoencoder = Autoencoder(encoder, decoder, params_dict['epochs'], params_dict['l_rate'],
                                          params_dict['batch_size'])
                autoencoder.compile(Adam(params_dict['l_rate']))

                early_stopping = EarlyStopping("val_loss", patience=params_dict['patience'])
                autoencoder.fit(x_train_fold, x_train_fold, params_dict['batch_size'], params_dict['epochs'],
                                validation_data=(x_val_fold, x_val_fold), callbacks=[early_stopping], verbose=0)

                predicted = autoencoder.predict(x_val_fold)
                ssim = scaled_ssim(x_val_fold, predicted)
                mse = mean_squared_error(x_val_fold, predicted)
                ssim_scores.append(ssim)
                mse_scores.append(mse)

                # Clear the TensorFlow session to free GPU memory
                # https://stackoverflow.com/a/52354943/17082611
                tf.keras.backend.clear_session()
                del encoder, decoder, autoencoder
                gc.collect()

            avg_ssim = np.mean(ssim_scores)
            avg_mse = np.mean(mse_scores)

            ssim_std = np.std(ssim_scores)
            mse_std = np.std(mse_scores)

            avg_score = avg_ssim / (avg_mse + 1)
            score_std = np.std(ssim_scores) / (np.std(mse_scores) + 1)

            params_dict['avg_score'] = avg_score
            params_dict['score_std'] = score_std
            params_dict['ssim'] = avg_ssim
            params_dict['mse'] = avg_mse
            params_dict['ssim_std'] = ssim_std
            params_dict['mse_std'] = mse_std
            self.grid_.append(params_dict)

            print(f"avg_ssim for current combination: {avg_score:.4f} ± {ssim_std:.4f}")
            print(f"avg_mse for current combination: {avg_mse:.4f} ± {mse_std:.4f}")
            print(f"avg_score for current combination: {avg_score:.4f} ± {score_std:.4f}")

            # Update the best hyperparameters based on the highest avg_score
            if self.best_score_ is None or avg_score > self.best_score_:
                self.best_score_ = avg_score
                self.best_params_ = params_dict

        return self


def save(history, subject):
    file_name = f'history_{subject}_standard.pickle'
    with open(file_name, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    to_client.append(file_name)


if __name__ == '__main__':
    print("TensorFlow GPU usage:", tf.config.list_physical_devices('GPU'))

    # Dati ridotti al solo intorno del blink
    subject = "s05"
    topomaps_folder = f"topomaps_reduced_{subject}"
    labels_folder = f"labels_reduced_{subject}"

    # Load data
    x_train, x_test, y_train, y_test = load_data(topomaps_folder, labels_folder,
                                                 0.2, False, 42)

    # Expand dimensions to (None, 40, 40, 1)
    # This is because AE is currently working with 4d tensors
    x_train = expand(x_train)
    x_test = expand(x_test)

    # Print data shapes
    # x_train deve essere circa 2700, mentre x_test circa 700
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Normalization between 0 and 1
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # Grid search
    latent_dimension = 28
    grid = custom_grid_search(x_train, latent_dimension)

    # Refit
    history, autoencoder = refit(grid, x_train, y_train, latent_dimension)
    save(history, subject)
    autoencoder.save_weights(f"checkpoints/ae_{subject}", save_format='tf')

    # Questa parte serve per serializzare i pesi e verificare che a seguito del load
    # Essi siano uguali nel file analysis.py
    # dbg
    w_before = autoencoder.get_weights()
    with open(f"w_before_{subject}_standard.pickle", "wb") as fp:
        pickle.dump(w_before, fp)

    print(f"\nTraining finished. You can transfer to client: {to_client}")
