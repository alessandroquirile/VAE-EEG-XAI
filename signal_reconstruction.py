import os

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def process_topomaps(x_with_blinks, subject, x, topomaps_files, topomaps_folder, my_topomaps_dict, is_train, only_rec):
    set_type = "train" if is_train else "test"

    input_folder = "masked_rec_standard" if not only_rec else "rec_standard"
    suffix = "mod" if not only_rec else "rec"

    modified_arrays = {}

    # In analysis_standard, dentro mask_test:
    # x_test_blinks.shape[0] # 25
    # num_cols = len(x_test_blinks) // 2  # 12,5 => 12
    # fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))
    # for i, ax in enumerate(axs.ravel()):
    #    print(i)  # va da 0 a 23 anziché 0 a 24
    #
    # i va da 0 a 23 (e non fino a 24) perché num_cols è approssimato per difetto
    # Quindi posso "omettere" il file che viene anche omesso nel grafico generato da mask_test
    # Questo giustifica il continue dentro l'except

    # Per ogni topomap in x_test_blinks
    for i in range(x_with_blinks.shape[0]):
        # Mi carico la ricostruzione mascherata (o ricostruzione e basta) della SINGOLA topomap corrente e denormalizzo
        # ATTENZIONE: AE = _standard; VAE = rimuovere _standard
        # I dati e la cartella sono creati nel file analysis o analysis_standard
        try:
            model_output = np.load(f"{input_folder}/{subject}/x_{set_type}_{i}.npy")
        except FileNotFoundError as e:
            print(f"File not found: {e.filename} (continue)")
            # Continua con il prossimo file
            continue
        model_output_denormalized = denormalize(model_output, x)
        model_output_denormalized = model_output_denormalized.squeeze()  # (40,40,1) => (40,40)

        # Per ogni file in topomaps_file (es. s01_trial36.npy, s01_trial17.npy...)
        for file in topomaps_files:
            # Mi carico l'array associato e ne faccio una copia per sicurezza
            topomaps_array = np.load(f"{topomaps_folder}/{file}")

            # Per ogni topomap del file corrente es s01_trial36.npy (256,...) quindi ciascuno delle 256 immagini
            for j in range(topomaps_array.shape[0]):
                # Se la topomap corrente corrisponde ad una topomap in topomaps_array
                if np.all(x_with_blinks[i] == topomaps_array[j]):
                    # print(f"x_{set_type}_blinks[{i}] == topomaps_array[{j}] e compare nel file {file}")
                    # Modifica specifici elementi nell'array topomaps_array_modified utilizzando
                    # gli indici specificati nel dict
                    for elem in my_topomaps_dict[file][f"topomaps_array_{set_type}"]:
                        # print(f"Modifico topomaps_array_modified[{elem}]")
                        topomaps_array[elem] = model_output_denormalized

                    modified_arrays[file] = topomaps_array

    # Salvataggio dei risultati per ciascun file
    for file, modified_array in modified_arrays.items():
        folder = f"{topomaps_folder}_{suffix}"
        os.makedirs(folder, exist_ok=True)
        trial_number = file.split("_")[1].split(".")[0]
        file_name = f"{subject}_{trial_number}.npy"
        np.save(os.path.join(folder, file_name), modified_array)
    # print("\n")

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
                    # print(f"x_{set_type}_blinks[{i}] == topomaps_array[{j}] e compare nel file {file}")
        if not found:
            raise Exception(f"x_{set_type}_blinks[{i}] non è stato trovato in alcun file")
    # print("\n")


def dbg_files(only_rec):
    print("Debug:")

    if only_rec:
        folder_suffix = 'rec'
    else:
        folder_suffix = 'mod'

    folder_mod = f'topomaps_reduced_{subject}_{folder_suffix}'
    folder_original = f'topomaps_reduced_{subject}'

    # Ottenere la lista dei nomi dei file nelle cartelle
    files_mod = os.listdir(folder_mod)
    files_original = os.listdir(folder_original)

    all_files_same = True

    for file_name in files_mod:
        if file_name in files_original:
            path_mod = os.path.join(folder_mod, file_name)
            path_original = os.path.join(folder_original, file_name)

            data_mod = np.load(path_mod)
            data_original = np.load(path_original)

            if not np.array_equal(data_mod, data_original):
                print(f"Il file {file_name} è stato modificato")
                all_files_same = False

    if all_files_same:
        raise Exception("Nessun file è stato modificato")


if __name__ == '__main__':
    print("TensorFlow GPU usage:", tf.config.list_physical_devices('GPU'))

    # Dati ridotti al solo intorno del blink
    subject = "s01"
    topomaps_folder = f"topomaps_reduced_{subject}"
    labels_folder = f"labels_reduced_{subject}"

    print(f"\n>>> QUESTO SCRIPT GENERA LE CARTELLE topomaps_reduced_{subject}_mod/ (e _rec/) <<<")

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

    # Produce topomaps_reduced_s01_mod/ (ricostruzioni mascherate)
    process_topomaps(x_train_blinks, subject, x_train, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=True,
                     only_rec=False)
    process_topomaps(x_test_blinks, subject, x_test, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=False,
                     only_rec=False)

    dbg_files(only_rec=False)

    # Produce topomaps_reduced_s01_rec/ (solo ricostruzioni)
    process_topomaps(x_train_blinks, subject, x_train, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=True,
                     only_rec=True)
    process_topomaps(x_test_blinks, subject, x_test, topomaps_files, topomaps_folder, my_topomaps_dict, is_train=False,
                     only_rec=True)

    dbg_files(only_rec=True)
