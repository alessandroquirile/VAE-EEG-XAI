from scipy.stats import skew
from sklearn.metrics import roc_auc_score
from train import *
from scipy.stats import mode

to_client = []  # List of file names that can be transferred to client


def plot_latent_space(vae, data, points_to_sample=30, figsize=15):
    """
    Plots the latent space of a Variational Autoencoder (VAE).
    This function generates a 2D manifold plot of digits in the latent space
    of the VAE. Each point in the plot represents a digit generated by the VAE's
    decoder model based on a specific location in the latent space.

    :param vae: The trained VAE model.
    :param data: Data to have a latent representation of. Shape should be (num_samples, 40, 40).
    :param points_to_sample: The number of points to sample along each axis of the plot. Default is 30.
    :param figsize: The size of the figure (width and height) in inches. Default is 15.
    :return: None (displays the plot).
    """

    np_config.enable_numpy_behavior()

    image_size = data.shape[1]
    scale = 1.0

    # Create an empty figure to store the generated images
    figure = np.zeros((image_size * points_to_sample, image_size * points_to_sample))

    # Define linearly spaced coordinates corresponding to the 2D plot in the latent space
    grid_x = np.linspace(-scale, scale, points_to_sample)
    grid_y = np.linspace(-scale, scale, points_to_sample)[::-1]  # Reverse the order of grid_y

    # Apply t-SNE to the latent space
    z_mean, _, _ = vae.encoder(data)
    tsne = TSNE(n_components=2, verbose=1)
    z_mean_reduced = tsne.fit_transform(z_mean)

    # Generate the images by iterating over the grid coordinates
    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            # Find the nearest t-SNE point to the current coordinates
            dist = np.sqrt((z_mean_reduced[:, 0] - xi) ** 2 + (z_mean_reduced[:, 1] - yi) ** 2)
            idx = np.argmin(dist)
            z_sample = z_mean[idx]

            # Decode the latent sample to generate an image
            x_decoded = vae.decoder(np.expand_dims(z_sample, axis=0))

            # Reshape the decoded image to match the desired image size
            digit = x_decoded.reshape(image_size, image_size)

            # Add the digit to the corresponding position in the figure
            figure[i * image_size: (i + 1) * image_size, j * image_size: (j + 1) * image_size] = digit

    # Plotting the figure
    plt.figure(figsize=(figsize, figsize))

    # Define the tick positions and labels for the x and y axes
    start_range = image_size // 2
    end_range = points_to_sample * image_size + start_range
    pixel_range = np.arange(start_range, end_range, image_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)

    # Set the x and y axis labels
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")

    # Display the figure
    plt.imshow(figure)
    plt.show()


def label_clusters(vae, data, labels):
    """
    Returns a t-SNE projection of the given data, with labels represented by different colors.

    :param vae: The trained VAE (Variational Autoencoder) model.
    :param data: Input data of shape (num_samples, 40, 40).
    :param labels: Array of labels corresponding to the data of shape (num_samples,).
    :return: fig
    """

    # call vs predict: https://stackoverflow.com/a/70205891/17082611
    z_mean, _, _ = vae.encoder(data)

    data = data.reshape(data.shape[0], -1)

    tsne = TSNE(n_components=2, verbose=1)
    z_mean_reduced = tsne.fit_transform(data)

    df = pd.DataFrame()
    df["labels"] = labels.flatten()
    df["comp-1"] = z_mean_reduced[:, 0]
    df["comp-2"] = z_mean_reduced[:, 1]

    # Get the distinct labels and the number of colors needed
    distinct_labels = np.unique(labels)
    n_colors = len(distinct_labels)

    # Create the plot
    fig = plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="comp-1", y="comp-2", hue=df.labels.tolist(),
                    palette=sns.color_palette("hls", n_colors)).set(title="Data t-SNE projection")
    return fig


def plot_metric(history_path, metric):
    # Usage: history_path is "history_{subject}.pickle"
    # Usage: metric is "loss", "reconstruction_loss" or "kl_loss"
    with open(history_path, "rb") as fp:
        history_data = pickle.load(fp)
    plt.plot(history_data[metric])
    plt.plot(history_data['val_' + metric])
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


def check_weights_equality(w_before_path, vae):
    w_after = vae.get_weights()
    with open(w_before_path, "rb") as fp:
        w_before = pickle.load(fp)
    for tensor_num, (w_before, w_after) in enumerate(zip(w_before, w_after), start=1):
        if not w_before.all() == w_after.all():
            raise Exception(f"Weights loading was unsuccessful for tensor {tensor_num}")


def save_clusters(file_name):
    with open(file_name, "wb") as fp:
        pickle.dump(clusters, fp)
    to_client.append(file_name)


def show_clusters(clusters_path):
    with open(clusters_path, "rb") as fp:
        pickle.load(fp)
    plt.show()


def show_topomaps(original_file_name, reconstructed_file_name):
    # Carica l'immagine originale e quella ricostruita
    original_image = np.load(original_file_name)
    reconstructed_image = np.load(reconstructed_file_name)

    # Calcolo SSIM, MSE e score
    ssim, mse, score = calculate_score(original_image, reconstructed_image)

    # Creare una figura con due sottoplot
    plt.figure(figsize=(11, 5))

    # Sottoplot per l'immagine originale
    plt.subplot(1, 2, 1)
    plt.title(original_file_name)
    plt.imshow(original_image, cmap="gray")

    # Sottoplot per l'immagine ricostruita
    plt.subplot(1, 2, 2)
    plt.title(f"{reconstructed_file_name} (SSIM = {ssim:.4f}, MSE = {mse:.4f}, score = {score:.4f})")
    plt.imshow(reconstructed_image, cmap="gray")

    # Mostra entrambe le immagini
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
    avg_score = avg_ssim / (avg_mse + 1)
    print(f"[dbg] avg_ssim for best combination on folds: {avg_ssim:.4f}")
    print(f"[dbg] avg_mse for best combination on folds: {avg_mse:.4f}")
    print(f"[dbg] avg_score for best combination on folds: {avg_score:.4f}\n")


def calculate_score(original, reconstructed):
    ssim = scaled_ssim(original, reconstructed)
    mse = mean_squared_error(original, reconstructed)
    score = ssim / (mse + 1)
    return ssim, mse, score


def calculate_score_test_set(x_test):
    ssim_scores = []
    mse_scores = []
    for i in range(len(x_test)):
        original = x_test[i]
        reconstructed = vae.predict(np.expand_dims(original, axis=0), verbose=0)  # (1, 40, 40, 1)
        ssim = scaled_ssim(original, reconstructed[0])  # reconstructed[0] shape is (40, 40, 1)
        mse = mean_squared_error(original, reconstructed[0])
        ssim_scores.append(ssim)
        mse_scores.append(mse)
    avg_ssim = np.mean(ssim_scores)
    avg_mse = np.mean(mse_scores)
    avg_score = avg_ssim / (avg_mse + 1)
    return avg_ssim, avg_mse, avg_score


def histogram_25_75(vae, x_test, y_test, latent_dim, subject):
    z_mean, _, _ = vae.encoder(x_test)

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

    print('\nIl numero di blink è:', len(blink))
    print('Il numero di non blink è:', len(no_blink))
    print('Il numero di transizioni è:', len(trans))

    z_mean_blink, _, _ = vae.encoder(blink)
    z_mean_no_blink, _, _ = vae.encoder(no_blink)
    z_mean_trans, _, _ = vae.encoder(trans)

    # Indici di dispersione.
    # Mediana: quantile di ordine 1/2
    # Quartili: quantili di ordine 1/4, 1/2 e 3/4
    # Percentili: quantili di ordine 1/100
    quantile_25 = np.quantile(z_mean, .25, axis=0)  # Primo quartile (Q1)
    quantile_75 = np.quantile(z_mean, .75, axis=0)  # Terzo quartile (Q3)

    num_rows = 7
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

    for j in range(0, z_mean_blink.shape[1]):  # 0, 1,...,latent_dim-1
        true_blink = 0  # TP
        negative_blink = 0  # FN
        negative_noblink = 0  # FP
        true_noblink = 0  # TN
        for i in range(0, z_mean_blink.shape[0]):  # 0, 1, ..., n_blinks-1
            if z_mean_blink[i, j] <= quantile_25[j] or z_mean_blink[i, j] >= quantile_75[j]:  # Out
                true_blink = true_blink + 1
            else:
                negative_blink = negative_blink + 1
        for i in range(0, z_mean_no_blink.shape[0]):  # 0, 1, ..., n_non_blinks-1
            if quantile_25[j] < z_mean_no_blink[i, j] < quantile_75[j]:
                true_noblink = true_noblink + 1
            else:
                negative_noblink = negative_noblink + 1

        # Disegno
        row_index = j // num_cols
        col_index = j % num_cols
        ax = axes[row_index, col_index]
        ax.set_title(f'histogram latent component {j + 1}')
        ax.hist(z_mean_blink[:, j], bins=100, color='green', alpha=0.6,
                label=f'Blink. TP:{true_blink}, FN:{negative_blink}')
        ax.hist(z_mean_no_blink[:, j], bins=100, color='violet', alpha=0.6,
                label=f'No Blink. TN:{true_noblink}, FP:{negative_noblink}')
        ax.hist(z_mean_trans[:, j], bins=100, color='yellow', alpha=0.6, label='Transition')

        ax.axvline(quantile_25[j], color='black', linestyle='-', label='Quantile 25')
        ax.axvline(quantile_75[j], color='blue', linestyle='-', label='Quantile 75')
        ax.legend()

    plt.tight_layout()

    histogram_file_name = f'histogram_25_75_{subject}.png'
    fig.savefig(histogram_file_name)
    to_client.append(histogram_file_name)

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
    return quantile_matrix, z_mean, z_mean_blink, z_mean_no_blink, z_mean_trans


def auc_roc(quantile_matrix, z_mean_blink, z_mean_no_blink, subject):
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

    auc_roc_file_name = f"curve_auc_roc_{subject}.png"
    fig.savefig(auc_roc_file_name)
    to_client.append(auc_roc_file_name)


def print_latent_components_decreasing_variance(z_mean):
    # Calcola la varianza delle colonne dell'array z_mean_val lungo l'asse 0
    variance = np.var(z_mean, axis=0)
    print('\nVarianza:', variance)

    # Crea una lista di tuple contenenti l'indice della colonna e il valore di varianza
    indexed_variances = list(enumerate(variance))  # (indice, valore)

    # Ordina le tuple in indexed_variances in base al valore di varianza, in ordine decrescente
    sorted_variances = sorted(indexed_variances, key=lambda x: x[1], reverse=True)

    # Estrae gli indici delle colonne ordinate in base alla varianza
    sorted_indices = [index for index, _ in sorted_variances]

    # Estrae i valori di varianza ordinati corrispondenti agli indici
    sorted_variance_vector = [variance for _, variance in sorted_variances]

    # Stampa gli indici delle colonne in base alla varianza ordinata
    print("Indici ordinati:", sorted_indices)

    # Stampa i valori di varianza corrispondenti agli indici ordinati
    print("Varianza ordinata:", sorted_variance_vector)

    # Calcola la skewness delle colonne dell'array z_mean_val lungo l'asse 0
    skewness = skew(z_mean, axis=0)
    print('Skewness:', skewness)


def get_original_and_reconstructed(vae, x_test, image_index):
    original_image = x_test[image_index]
    x_test_reconstructed = vae.predict(x_test, verbose=0)
    reconstructed_image = x_test_reconstructed[image_index]
    return original_image, reconstructed_image


def save_test_blink_originals(x_test, subject):
    # Usa questa funzione per salvare in un unico png tutte le immagini di test con blink per un certo soggetto
    num_rows = 2
    num_cols = len(x_test) // num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))
    for i, ax in enumerate(axs.ravel()):
        ax.imshow(x_test[i], cmap="gray")
        ax.set_title(f"x_test[{i}]")
        ax.axis('off')
    # Save the entire figure
    fig.suptitle(f"{subject} test blinks", fontsize=26)
    file_name = f"test_blinks_{subject}.png"
    fig.savefig(file_name)
    to_client.append(file_name)


def save_test_blink_reconstructions(vae, x_test, subject):
    # Usa questa funzione per salvare in un unico png tutte le ricostruzioni di
    # immagini di test con blink per un certo soggetto
    reconstructions = []
    for i in range(len(x_test)):
        original = x_test[i]
        reconstructed = vae.predict(np.expand_dims(original, axis=0), verbose=0)  # (1, 40, 40, 1)
        reconstructions.append(reconstructed[0])
    num_rows = 2
    num_cols = len(x_test) // num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))
    for i, ax in enumerate(axs.ravel()):
        ax.imshow(reconstructions[i], cmap="gray")
        ax.set_title(f"x_test[{i}]")
        ax.axis('off')
    # Save the entire figure
    fig.suptitle(f"{subject} test blinks reconstructed", fontsize=26)
    file_name = f"test_blinks_reconstructed_{subject}.png"
    fig.savefig(file_name)
    to_client.append(file_name)


def mask_single_test_sample(latent_component_indices, vae, x_test, subject):
    # Usa questa funzione per salvare in un unico png un'unica immagine di test mascherata per un certo soggetto

    _, _, z = vae.encoder(x_test, training=False)
    decoder_output = vae.decoder(z, training=False)
    # print(z.shape)  # (test_samples, latent_dim)
    # print(decoder_output.shape)  # (test_samples, 40, 40, 1)

    # Mask latent components using median value for each index in the list
    z_masked = np.copy(z)
    strategy = "Mode"  # or median
    for latent_component_idx in latent_component_indices:
        # median = np.median(z[:, latent_component_idx])
        # z_masked[:, latent_component_idx] = median
        mode_val, _ = mode(z[:, latent_component_idx])
        z_masked[:, latent_component_idx] = mode_val

    decoder_output_masked = vae.decoder(z_masked, training=False)

    # Plotta le immagini
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    image_index = 1

    original = x_test[image_index]
    axs[0].imshow(original, cmap="gray")
    axs[0].set_title('Originale')

    reconstructed = decoder_output[image_index]
    axs[1].imshow(reconstructed, cmap="gray")
    axs[1].set_title('Ricostruita')

    reconstructed_masked = decoder_output_masked[image_index]
    axs[2].imshow(reconstructed_masked, cmap="gray")
    axs[2].set_title(f'{strategy} z[{", ".join(map(str, latent_component_indices))}]')

    # Workaround per calcolare gli score
    """np.save("del.npy", original)
    original = np.load("del.npy")
    np.save("del.npy", reconstructed)
    reconstructed = np.load("del.npy")
    print("original vs reconstructed", calculate_score(original, reconstructed))
    np.save("del.npy", original)
    original = np.load("del.npy")
    np.save("del.npy", reconstructed_masked)
    reconstructed_masked = np.load("del.npy")
    print("original vs reconstructed_masked", calculate_score(original, reconstructed_masked))
    np.save("del.npy", reconstructed)
    reconstructed = np.load("del.npy")
    np.save("del.npy", reconstructed_masked)
    reconstructed_masked = np.load("del.npy")
    print("reconstructed vs reconstructed_masked", calculate_score(reconstructed, reconstructed_masked))
    os.remove("del.npy")"""

    # Rimuovi i ticks sugli assi
    for ax in axs:
        ax.axis('off')

    # Salva i grafici
    plt.tight_layout()
    file_name = f"z{'_'.join(map(str, latent_component_indices))}_{strategy.lower()}_{subject}.png"
    fig.savefig(file_name)
    to_client.append(file_name)


def mask_test_set(latent_component_indices, vae, x_test, subject):
    # Usa questa funzione per salvare in un unico png tutte le immagini mascherate
    # di test con blink per un certo soggetto

    # Il mascheramento va fatto considerando la moda delle immagini di test SENZA blink
    x_test_no_blinks = get_x_test_no_blinks(x_test, y_test)
    _, _, z = vae.encoder(x_test_no_blinks, training=False)

    # Create a figure with multiple subplots
    num_rows = 2
    x_test_blinks = get_x_test_blinks(x_test, y_test)
    num_cols = len(x_test_blinks) // num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))

    strategy = "Mode"  # or Median
    for i, ax in enumerate(axs.ravel()):
        z_masked = np.copy(z)
        for latent_component_idx in latent_component_indices:
            mode_value, _ = mode(z[:, latent_component_idx], keepdims=True)
            z_masked[:, latent_component_idx] = mode_value

        decoder_output_masked = vae.decoder(z_masked, training=False)
        reconstructed_masked = decoder_output_masked[i]

        ax.imshow(reconstructed_masked, cmap="gray")
        ax.set_title(f"x_test[{i}]")
        ax.axis('off')

    # Save the entire figure
    fig.suptitle(f"{subject} test blinks. Mask strategy is: {strategy}", fontsize=26)
    file_name = f"z{'_'.join(map(str, latent_component_indices))}_{strategy.lower()}_{subject}.png"
    fig.savefig(file_name)
    to_client.append(file_name)


def mask_test_set_reversed(latent_component_indices, vae, x_test, subject):
    # Usa questa funzione per salvare in un unico png tutte le immagini mascherate
    # di test con blink per un certo soggetto
    # REVERSED nel senso che legge gli indici ma maschera TUTTI GLI ALTRI

    # Il mascheramento va fatto considerando la moda delle immagini di test SENZA blink
    x_test_no_blinks = get_x_test_no_blinks(x_test, y_test)
    _, _, z = vae.encoder(x_test_no_blinks, training=False)

    # Create a figure with multiple subplots
    num_rows = 2
    x_test_blinks = get_x_test_blinks(x_test, y_test)
    num_cols = len(x_test_blinks) // num_rows
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))

    strategy = "Mode_reversed"  # or Median
    for i, ax in enumerate(axs.ravel()):
        z_masked = np.copy(z)
        for latent_component_idx in range(z.shape[1]):
            # Maschera tutti gli altri tranne quelli specificati
            if latent_component_idx not in latent_component_indices:
                mode_value, _ = mode(z[:, latent_component_idx], keepdims=True)
                z_masked[:, latent_component_idx] = mode_value

        decoder_output_masked = vae.decoder(z_masked, training=False)
        reconstructed_masked = decoder_output_masked[i]

        ax.imshow(reconstructed_masked, cmap="gray")
        ax.set_title(f"x_test[{i}]")
        ax.axis('off')

    # Save the entire figure
    fig.suptitle(f"{subject} test blinks. Mask strategy is: {strategy}", fontsize=26)
    file_name = f"z{'_'.join(map(str, latent_component_indices))}_{strategy.lower()}_{subject}.png"
    fig.savefig(file_name)
    to_client.append(file_name)


def get_x_test_no_blinks(x_test, y_test):
    blink_indices = np.where(y_test == 0)[0]  # Only on test samples labelled with NO BLINK (0)
    x_test = x_test[blink_indices]
    y_test = y_test[blink_indices]
    x_test_only_contains_no_blinks = all(y_test) == 0
    if not x_test_only_contains_no_blinks:
        raise Exception("Something went wrong while considering only no-blinks test images")
    return x_test

def get_x_test_blinks(x_test, y_test):
    blink_indices = np.where(y_test == 1)[0]  # Only on test samples labelled with BLINK (1)
    x_test = x_test[blink_indices]
    y_test = y_test[blink_indices]
    x_test_only_contains_blinks = all(y_test) == 1
    if not x_test_only_contains_blinks:
        raise Exception("Something went wrong while considering only blinks test images")
    return x_test


if __name__ == '__main__':
    print("TensorFlow GPU usage:", tf.config.list_physical_devices('GPU'))

    # Ogni soggetto ha il proprio learning rate ottimale
    learning_rates = {
        "s01": 1e-05,
        "s02": 1e-05,
        "s03": 1e-06,
        "s04": 1e-05,
        # "s05": 1e-05,
        "s06": 1e-05,
        "s07": 1e-05,
        # "s08": 1e-07
    }

    # Indici delle componenti latenti rilevanti per ciascun soggetto
    relevant_indices = {
        "s01": [0, 1, 5, 8, 15, 18],
        "s02": [1, 5, 12, 14, 19, 27],
        "s03": [0, 1, 5, 19, 24, 26],
        "s04": [0, 4, 10],
        # "s05": [],  # Non capisco perché ma gli istogrammi non sono significativi
        "s06": [0, 1, 2, 3],
        "s07": [0, 1, 2, 3, 4, 5],
        # "s08": []  # Non capisco perché ma gli istogrammi non sono significativi
    }

    # Dati ridotti al solo intorno del blink
    subject = "s07"
    topomaps_folder = f"topomaps_reduced_{subject}"
    labels_folder = f"labels_reduced_{subject}"

    # Load data. Use the same random state used in train.py for data consistency
    x_train, x_test, y_train, y_test = load_data(topomaps_folder, labels_folder,
                                                 0.2, False, 42)

    # Expand dimensions to (None, 40, 40, 1)
    # This is because VAE is currently working with 4d tensors
    x_train = expand(x_train)
    x_test = expand(x_test)

    # Normalization
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # Loading saved weights
    latent_dimension = 28
    best_l_rate = learning_rates[subject]
    encoder = Encoder(latent_dimension)
    decoder = Decoder()
    vae = VAE(encoder, decoder)
    vae.compile(Adam(best_l_rate))
    vae.train_on_batch(x_train[:1], x_train[:1])  # Very important, avoids harmful warnings
    vae.load_weights(f"checkpoints/vae_{subject}")

    # The parameters must be the same before/after the load
    check_weights_equality(f"w_before_{subject}.pickle", vae)

    """# avg_score for current combination (dbg)
    avg_score_dbg()

    # Save clusters - on server
    clusters = label_clusters(vae, x_test, y_test)
    clusters_file_name = f"clusters_{subject}.pickle"
    save_clusters(clusters_file_name)

    # Read clusters - only on client
    # show_clusters(clusters_file_name)

    # Save images - on server
    original, reconstructed = get_original_and_reconstructed(vae, x_test, image_index=1)
    original_file_name = f"original_{subject}.npy"
    reconstructed_file_name = f"reconstructed_{subject}.npy"
    np.save(original_file_name, original)
    np.save(reconstructed_file_name, reconstructed)
    to_client.append(original_file_name)
    to_client.append(reconstructed_file_name)

    # Show original and reconstructed image - only on client
    # show_topomaps(original_file_name, reconstructed_file_name)

    # Score on random test sample
    original = np.load(original_file_name)
    reconstructed = np.load(reconstructed_file_name)
    ssim, mse, score = calculate_score(original, reconstructed)
    print(f"\nssim on random test sample: {ssim:.4f}")
    print(f"mse on random test sample: {mse:.4f}")
    print(f"score on random test sample: {score:.4f}")

    # avg_score on the whole test set
    avg_ssim, avg_mse, avg_score = calculate_score_test_set(x_test)
    print(f"\navg_ssim on test set: {avg_ssim:.4f}")
    print(f"avg_mse on test set: {avg_mse:.4f}")
    print(f"avg_score on test set: {avg_score:.4f}")

    # For each latent component a histogram is created for analyzing the test data distribution
    # The 25th and 75th percentiles are computed for each latent component in order to understand whether
    # Blinks are located outside tha range. For each histogram a confusion matrix is also computed
    quantile_matrix, z_mean, z_mean_blink, z_mean_no_blink, _ = histogram_25_75(vae, x_test, y_test,
                                                                                latent_dimension, subject)

    # For each latent component the ROC-AUC curve is created for detecting the quartile range which
    # Maximizes the TPR (True Positive Rate)
    auc_roc(quantile_matrix, z_mean_blink, z_mean_no_blink, subject)"""

    # Mask relevant latent components
    # "Relevant" means large IQR (implies more variance of data) and many TP blinks (outside the IQR)
    mask_test_set(relevant_indices[subject], vae, x_test, subject)  # maschera quelle specificate
    mask_test_set_reversed(relevant_indices[subject], vae, x_test, subject)  # maschera tutte tranne quelle specificate

    x_test = get_x_test_blinks(x_test, y_test)
    save_test_blink_originals(x_test, subject)
    save_test_blink_reconstructions(vae, x_test, subject)

    print(f"\nFinished. You can transfer to client: {to_client}")
