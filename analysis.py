import keras
from keras import Input

from train import *
from sklearn.metrics import roc_auc_score


def check_weights_equality(w_before_path, vae):
    w_after = vae.get_weights()
    with open(w_before_path, "rb") as fp:
        w_before = pickle.load(fp)
    for tensor_num, (w_before, w_after) in enumerate(zip(w_before, w_after), start=1):
        if not w_before.all() == w_after.all():
            raise Exception(f"Weights loading was unsuccessful for tensor {tensor_num}")


def save_clusters(file_name):
    print(f"Saving {file_name}\n")
    with open(file_name, "wb") as fp:
        pickle.dump(clusters, fp)


def show_clusters(clusters_path):
    with open(clusters_path, "rb") as fp:
        pickle.load(fp)
    plt.show()


def show_original(topomap_path):
    plt.clf()
    plt.title(f"Original image")
    original_image = np.load(topomap_path)
    plt.imshow(original_image, cmap="gray")
    plt.show()


def show_reconstructed(topomap_path):
    plt.clf()
    reconstructed_image = np.load(topomap_path)
    plt.title(f"Reconstructed image")
    plt.imshow(reconstructed_image, cmap="gray")
    plt.show()


def avg_score_dbg():
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    ssim_scores = []
    mse_scores = []
    for train_idx, val_idx in cv.split(x_train):
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        predicted = vae.predict(x_val_fold)
        ssim = scaled_ssim(x_val_fold, predicted)
        mse = mean_squared_error(x_val_fold, predicted)
        ssim_scores.append(ssim)
        mse_scores.append(mse)
    avg_ssim = np.mean(ssim_scores)
    avg_mse = np.mean(mse_scores)
    # avg_score = (avg_ssim + avg_mse) / 2
    avg_score = avg_ssim / (avg_mse + 1)
    print(f"[dbg] avg_ssim for best combination on folds: {avg_ssim:.4f}")
    print(f"[dbg] avg_mse for best combination on folds: {avg_mse:.4f}")
    print(f"[dbg] avg_score for best combination on folds: {avg_score:.4f}\n")


def calculate_score(original, reconstructed):
    original_image = np.load(original)
    reconstructed_image = np.load(reconstructed)
    ssim = scaled_ssim(original_image, reconstructed_image)
    mse = mean_squared_error(original_image, reconstructed_image)
    score = ssim / (mse + 1)
    print(f"\nssim on random test sample: {ssim:.4f}")
    print(f"mse on random test sample: {mse:.4f}")
    print(f"score on random test sample: {score:.4f}")


def calculate_score_test_set(x_test):
    ssim_scores = []
    mse_scores = []
    for i in range(len(x_test)):
        original = x_test[i]
        reconstructed = vae.predict(np.expand_dims(original, axis=0), verbose=0)  # (1, 40, 40, 1)
        ssim_score = scaled_ssim(original, reconstructed[0])  # reconstructed[0] shape is (40, 40, 1)
        mse_score = mean_squared_error(original_image, reconstructed[0])
        ssim_scores.append(ssim_score)
        mse_scores.append(mse_score)
    avg_ssim = np.mean(ssim_scores)
    avg_mse = np.mean(mse_scores)
    # avg_score = (avg_ssim + avg_mse) / 2
    avg_score = avg_ssim / (avg_mse + 1)
    print(f"\navg_ssim on test set: {avg_ssim:.4f}")
    print(f"avg_mse on test set: {avg_mse:.4f}")
    print(f"avg_score on test set: {avg_score:.4f}\n")


def histogram_25_75(vae, x_test, y_test, latent_dim, subject):
    z_mean, z_log_var, _ = vae.encoder(x_test)

    no_blink = []
    blink = []
    trans = []
    for i in range(0, len(y_test)):
        if y_test[i] == 0:
            no_blink.append(x_test[i])
        if y_test[i] == 1:
            blink.append(x_test[i])
        if y_test[i] == 2:
            trans.append(x_test[i])
    blink = np.array(blink)
    no_blink = np.array(no_blink)
    trans = np.array(trans)

    print('Il numero di blink è:', len(blink))
    print('Il numero di non blink è:', len(no_blink))
    print('Il numero di transizioni è:', len(trans))

    z_mean_blink, z_log_var_blink, _ = vae.encoder(blink)
    z_mean_no_blink, z_log_var_no_blink, _ = vae.encoder(no_blink)
    z_mean_trans, z_log_var_trans, _ = vae.encoder(trans)

    quantile_25 = np.quantile(z_mean, .25, axis=0)
    quantile_75 = np.quantile(z_log_var, .75, axis=0)
    num_rows = 7
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))
    for j in range(0, z_mean_blink.shape[1]):

        # print (f'latent component {j+1}')
        true_blink = 0
        negative_blink = 0
        negative_noblink = 0
        true_noblink = 0

        for i in range(0, z_mean_blink.shape[0]):
            # print(quantile_25[j])
            # print(z_mean_blink[i,j])
            # print(quantile_75[j])
            if z_mean_blink[i, j] <= quantile_25[j] or z_mean_blink[i, j] >= quantile_75[j]:
                # print("Out")
                true_blink = true_blink + 1
            else:
                negative_blink = negative_blink + 1

        # print('True positive (TP):',true_blink)
        # print('False negative (FN):',negative_blink)

        for z in range(0, z_mean_no_blink.shape[0]):
            # print(quantile_25[j])
            # print(z_mean_blink[i,j])
            # print(quantile_75[j])
            if z_mean_no_blink[z, j] > quantile_25[j] and z_mean_no_blink[z, j] < quantile_75[j]:
                true_noblink = true_noblink + 1
            else:
                negative_noblink = negative_noblink + 1

                # print('True negative (TN):',true_noblink)
        # print('False positive (FP):',negative_noblink)

        row_index = j // num_cols
        col_index = j % num_cols

        ax = axes[row_index, col_index]
        ax.set_title(f'histogram latent component {j + 1}')
        ax.hist(z_mean_blink[:, j], bins=100, color='green', alpha=0.6,
                label=f'Blink-TP:{true_blink}-FN:{negative_blink}')
        ax.hist(z_mean_no_blink[:, j], bins=100, color='violet', alpha=0.6,
                label=f'No Blink-TN:{true_noblink}-FP:{negative_noblink}')
        ax.hist(z_mean_trans[:, j], bins=100, color='yellow', alpha=0.6, label='Transizioni')

        ax.axvline(quantile_25[j], color='black', linestyle='-', label='Quantile 25')
        ax.axvline(quantile_75[j], color='blue', linestyle='-', label='Quantile 75')

        ax.legend()

    plt.tight_layout()

    fig.savefig(f'histogram_25_75_{subject}.png')
    print(f"histogram_25_75_{subject}.png saved")

    n_intervalli = 9

    M = np.zeros((latent_dim, n_intervalli * 2))
    start = 0.05
    end = 0.95
    step = 0.05
    Q = np.arange(start, end + step, step)

    i = 0
    for q in Q:
        # print("q",q)
        if q != 0.5:
            # print("i",i)
            M[:, i] = np.quantile(z_mean, q, axis=0)
            i = i + 1
    quantile_matrix = M
    return quantile_matrix, z_mean_blink, z_mean_no_blink, z_mean_trans


def roc_auc(quantile_matrix, z_mean_blink, z_mean_no_blink, subject):
    num_rows = 7
    num_cols = 4

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

    for j in range(z_mean_blink.shape[1]):
        FPR = []
        TPR = []
        for q in range(0, int(quantile_matrix.shape[1] / 2)):
            true_blink = 0  # TP
            negative_blink = 0  # FN
            negative_noblink = 0  # FP
            true_noblink = 0  # TN

            for i in range(0, z_mean_blink.shape[0]):
                if z_mean_blink[i, j] <= quantile_matrix[j, q] or z_mean_blink[i, j] >= quantile_matrix[j, -q - 1]:
                    true_blink = true_blink + 1
                else:
                    negative_blink = negative_blink + 1

            for z in range(0, z_mean_no_blink.shape[0]):
                if quantile_matrix[j, q] < z_mean_no_blink[z, j] < quantile_matrix[j, -q - 1]:
                    true_noblink = true_noblink + 1
                else:
                    negative_noblink = negative_noblink + 1

            FPR.append(negative_noblink / (negative_noblink + true_noblink))  # FP/(FP+TN)
            TPR.append(true_blink / (true_blink + negative_blink))  # TP/(TP+FN)

        # AUC
        auc = roc_auc_score(np.concatenate((np.zeros(len(FPR)), np.ones(len(TPR)))), np.concatenate((FPR, TPR)))

        # Calcoliamo l'indice di riga e colonna del subplot
        row_index = j // num_cols
        col_index = j % num_cols

        # Disegniamo la curva ROC sul subplot corrispondente
        ax = axes[row_index, col_index]
        ax.plot(FPR, TPR, color='b', label=f'Curva ROC - AUC: {auc:.2f}')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')

        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title(f'Colonna - latent component {j + 1}')
        ax.legend(loc='lower right')

    # Impostiamo lo spazio tra i subplot
    plt.tight_layout()

    fig.savefig(f'curve_ROC_{subject}.png')
    print(f"curve_ROC_{subject}.png saved")


if __name__ == '__main__':
    print("TensorFlow GPU usage:", tf.config.list_physical_devices('GPU'))

    # Dati ridotti al solo intorno del blink
    subject = "s01"
    topomaps_folder = f"topomaps_reduced_{subject}"
    labels_folder = f"labels_reduced_{subject}"

    # Load data
    x_train, x_test, y_train, y_test = load_data(topomaps_folder, labels_folder, 0.2, False)

    # Expand dimensions to (None, 40, 40, 1)
    # This is because VAE is currently working with 4d tensors
    x_train = expand(x_train)
    x_test = expand(x_test)

    # Normalization
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # Loading saved weights
    latent_dimension = 28
    best_l_rate = 1e-05
    encoder = Encoder(latent_dimension)
    decoder = Decoder()
    vae = VAE(encoder, decoder)
    vae.compile(Adam(best_l_rate))
    vae.train_on_batch(x_train[:1], x_train[:1])  # Fondamentale per evitare i warning
    vae.load_weights(f"checkpoints/vae_{subject}")

    # Verifico che i pesi siano inalterati prima/dopo il load
    check_weights_equality(f"w_before_{subject}.pickle", vae)

    # Verifica SSIM medio per la combinazione corrente
    avg_score_dbg()

    # Salvo i cluster - su server
    clusters = plot_label_clusters(vae, x_test, y_test)
    save_clusters(f"clusters_{subject}.pickle")

    # Leggo i cluster - solo su client
    # show_clusters(f"clusters_{subject}.pickle")

    # Salvo le immagini - su server
    original_image, reconstructed_image = reconstruction_skill(vae, x_test)
    original_image_file_name = "original.npy"
    print("Saving original.npy and reconstructed.npy")
    np.save("original.npy", original_image)
    np.save("reconstructed.npy", reconstructed_image)

    # Mostra l'immagine originale e quella ricostruita - solo su client
    """show_original("original.npy")
    show_reconstructed("reconstructed.npy") """

    # Calcolo score su un campione casuale del test set
    calculate_score("original.npy", "reconstructed.npy")

    # Calcolo score sull'intero test test
    calculate_score_test_set(x_test)

    # Grafici
    quantile_matrix, z_mean_blink, z_mean_no_blink, _ = histogram_25_75(vae, x_test, y_test, latent_dimension, subject)
    roc_auc(quantile_matrix, z_mean_blink, z_mean_no_blink, subject)

    print("\nFinished. You can transfer clusters, original, reconstructed data and png to client for showing them")
