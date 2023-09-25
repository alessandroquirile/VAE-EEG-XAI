import os

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    # Check visuale, crea i png delle topomap specificate
    subject = "s01"
    trial = "21"

    file_name = subject + "_trial" + str(trial) + ".npy"
    topomaps = np.load(f"topomaps_reduced/{file_name}")
    labels = np.load(f"labels_reduced/{file_name}")
    output_folder = os.path.join("images", subject, trial)
    os.makedirs(output_folder, exist_ok=True)
    file_name_without_extension = os.path.splitext(file_name)[0]
    for i in tqdm(range(topomaps.shape[0]), desc=f"Saving {file_name_without_extension} topomaps", unit="topomap"):
        plt.imshow(topomaps[i], cmap="gray")
        plt.title(f"{file_name}[{i}] label = {labels[i]}")
        output_file = os.path.join(output_folder, f"{file_name_without_extension}_topomap{i + 1}.png")
        plt.savefig(output_file)
        plt.clf()
