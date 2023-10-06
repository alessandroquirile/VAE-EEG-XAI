from vae import *


def check_weights_equality(w_before_path, vae):
    w_after = vae.get_weights()
    with open(w_before_path, "rb") as fp:
        w_before = pickle.load(fp)
    for tensor_num, (w_before, w_after) in enumerate(zip(w_before, w_after), start=1):
        if not w_before.all() == w_after.all():
            raise Exception(f"Weights loading was unsuccessful for tensor {tensor_num}")


if __name__ == '__main__':
    print("TensorFlow GPU usage:", tf.config.list_physical_devices('GPU'))

    # Dati ridotti al solo intorno del blink
    subject = "s01"
    topomaps_folder = f"topomaps_reduced_{subject}"
    labels_folder = f"labels_reduced_{subject}"

    # Load data
    x_train, x_test, y_train, y_test = load_data(topomaps_folder, labels_folder, 0.2, False)

    # I am reducing the size of data set for speed purposes. For tests only
    # new_size = 200
    # x_train, y_train = reduce_size(x_train, y_train, new_size)
    # x_test, y_test = reduce_size(x_test, y_test, new_size)

    # Expand dimensions to (None, 40, 40, 1)
    # This is because VAE is currently working with 4d tensors
    x_train = expand(x_train)
    x_test = expand(x_test)

    # Normalization
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # From grid.best_params_
    latent_dimension = 25
    best_l_rate = 1e-05

    # Loading saved weights
    encoder = Encoder(latent_dimension)
    decoder = Decoder()
    vae = VAE(encoder, decoder)
    vae.compile(Adam(best_l_rate))
    vae.train_on_batch(x_train[:1], x_train[:1])  # Fondamentale per evitare i warning
    vae.load_weights(f"checkpoints/vae_{subject}")

    # Verifico che i pesi siano inalterati prima/dopo il load
    check_weights_equality("w_before.pickle", vae)  # dbg

    # Verifica SSIM medio per la combinazione corrente
    # dbg
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in cv.split(x_train):
        x_train_fold, x_val_fold = x_train[train_idx], x_train[val_idx]
        predicted = vae.predict(x_val_fold)
        score = my_ssim(x_val_fold, predicted)
        scores.append(score)
    avg_score = np.mean(scores)
    print(f"[dbg] avg_score (ssim) for current combination: {avg_score:.5f}")

    # plot_label_clusters(vae, x_test, y_test)
    # visually_check_reconstruction_skill(vae, x_train)
