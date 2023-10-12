from train import *


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

    # Calcolo SSIM su un campione casuale del test set
    original_image = np.load("original.npy")
    reconstructed_image = np.load("reconstructed.npy")
    ssim = scaled_ssim(original_image, reconstructed_image)
    mse = mean_squared_error(original_image, reconstructed_image)
    score = (ssim + mse) / 2
    print(f"\nssim on random test sample: {ssim:.4f}")
    print(f"mse on random test sample: {mse:.4f}")
    print(f"score on random test sample: {score:.4f}")

    # Calcolo SSIM sull'intero test test
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
    print(f"avg_score on test set: {avg_score:.4f}")

    print("\nFinished. You can transfer clusters, original and reconstructed data to client for showing them")
