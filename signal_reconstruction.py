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

from indian_functions import retrieveChannelInfoFromInterpolatedMap
from models import *


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


def expand(x):
    return np.expand_dims(x, -1).astype("float32")


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def get_min_max(x):
    return np.min(x), np.max(x)


def get_x_with_blinks(x, y):
    blink_indices = np.where(y == 1)[0]  # Only samples labelled with BLINK (1)
    x = x[blink_indices]
    y = y[blink_indices]
    x_only_contains_blinks = all(y) == 1
    if not x_only_contains_blinks:
        raise Exception("Something went wrong while considering only blinks images")
    return x


def denormalize(x_normalized, x):
    x_min, x_max = get_min_max(x)
    x = (x_normalized * (x_max - x_min)) + x_min
    return x

def process_topomaps(x_with_blinks, subject, x, topomaps_files, topomaps_folder, my_topomaps_dict, is_train):
    set_type = "train" if is_train else "test"
    # Per ogni topomap in x_test_blinks
    for i in range(x_with_blinks.shape[0]):
        # Mi carico la ricostruzione mascherata della SINGOLA topomap corrente e denormalizzo
        masked_rec = np.load(f"masked_rec_standard/{subject}/x_{set_type}_{i}.npy")
        masked_rec_denormalized = denormalize(masked_rec, x)
        masked_rec_denormalized = masked_rec_denormalized.squeeze()  # (40,40,1) => (40,40)

        # Per ogni file in topomaps_file (es. s01_trial36.npy, s01_trial17.npy...)
        for file in topomaps_files:
            # Mi carico il file .npy associato e ne faccio una copia per sicurezza
            topomaps_array = np.load(f"{topomaps_folder}/{file}")
            topomaps_array_modified = np.copy(topomaps_array)

            # Per ogni topomap del file corrente es s01_trial36.npy (256,...) quindi ciascuno delle 256 immagini
            for j in range(topomaps_array.shape[0]):
                # Se la topomap corrente corrisponde ad una topomap in topomaps_array
                if np.all(x_with_blinks[i] == topomaps_array[j]):
                    # Modifica specifici elementi nell'array topomaps_array_modified utilizzando
                    # gli indici specificati nel dict
                    for elem in my_topomaps_dict[file][f"topomaps_array_{set_type}"]:
                        topomaps_array_modified[elem] = masked_rec_denormalized

            # Es. topomaps_reduced_s01_mod
            folder = f"{topomaps_folder}_mod"
            os.makedirs(folder, exist_ok=True)
            trial_number = file.split("_")[1].split(".")[0]
            # Es. s01_trial03_train.npy oppure s01_trial03_test.npy
            file_name = f"{subject}_{trial_number}_{set_type}.npy"
            np.save(os.path.join(folder, file_name), topomaps_array_modified)

def find_matching_indices_in_topomaps(x_blinks, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=True):
    set_type = "train" if is_train else "test"
    for i in range(x_blinks.shape[0]):
        found = False
        for file in topomaps_files:
            topomaps_array = np.load(f"{topomaps_folder}/{file}")
            for j in range(topomaps_array.shape[0]):
                if np.all(x_blinks[i] == topomaps_array[j]):
                    if file not in my_topomaps_dict:
                        my_topomaps_dict[file] = {'topomaps_array_train': [],
                                                  'x_train_blinks': [],
                                                  'topomaps_array_test': [],
                                                  'x_test_blinks': []}
                    if is_train:
                        my_topomaps_dict[file]['topomaps_array_train'].append(j)
                        my_topomaps_dict[file]['x_train_blinks'].append(i)
                    else:
                        my_topomaps_dict[file]['topomaps_array_test'].append(j)
                        my_topomaps_dict[file]['x_test_blinks'].append(i)
                    found = True
                    print(f"x_{set_type}_blinks[{i}] == topomaps_array[{j}] e compare nel file {file}")
        if not found:
            raise Exception(f"x_{set_type}_blinks[{i}] non è stato trovato in alcun file")
    print("\n")

if __name__ == '__main__':
    print("TensorFlow GPU usage:", tf.config.list_physical_devices('GPU'))

    print("\n>>> QUESTO SCRIPT RICOSTRUISCE IL SEGNALE DALLE TOPOMAPS GENERANDO LA CARTELLA "
          "topomaps_reduced_s01_mod/<<<")

    # Dati ridotti al solo intorno del blink
    subject = "s01"
    topomaps_folder = f"topomaps_reduced_{subject}"
    labels_folder = f"labels_reduced_{subject}"

    # Load data
    x_train, x_test, y_train, y_test = load_data(topomaps_folder, labels_folder,
                                                 0.2, False, 42)

    # Only blinks
    x_train_blinks = get_x_with_blinks(x_train, y_train)
    x_test_blinks = get_x_with_blinks(x_test, y_test)

    # es. s01_trial36.npy, s01_trial17.npy...
    topomaps_files = [files for files in os.listdir(topomaps_folder) if not files.endswith('.DS_Store')]

    # Trove le corrispondenze tra x_train_blinks/x_test_blinks e le topomaps
    my_topomaps_dict = {}
    find_matching_indices_in_topomaps(x_train_blinks, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=True)
    find_matching_indices_in_topomaps(x_test_blinks, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=False)

    # Effettua le sostituzioni per generare topomaps_reduced_s01_mod/
    process_topomaps(x_train_blinks, subject, x_train, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=True)
    process_topomaps(x_test_blinks, subject, x_test, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=False)

    # TODO:
    #  1. Salvare in masked_rec_standard i dati di train mascherati (OK)
    #  2. Sostituire alle topomaps file quelli ottenuti dal mascheramento (come 172-186) (OK)
    #  3. Ripetere signal_topomaps_modified.py per i dati di train
    #  4. Dare in input al modello i dati dentro la cartella topomaps_reduced_s01_mod
    #  5. Passaggio al segnale di questi dati ricostruiti
