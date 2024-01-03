from scipy.stats import skew
from sklearn.metrics import roc_auc_score
from train_standard import *
from scipy.stats import mode

to_client = []  # List of file names that can be transferred to client


def label_clusters(vae, data, labels):
    """
    Returns a t-SNE projection of the given data, with labels represented by different colors.

    :param vae: The trained VAE (Variational Autoencoder) model.
    :param data: Input data of shape (num_samples, 40, 40).
    :param labels: Array of labels corresponding to the data of shape (num_samples,).
    :return: fig
    """

    # call vs predict: https://stackoverflow.com/a/70205891/17082611
    z = vae.encoder(data)

    data = data.reshape(data.shape[0], -1)

    tsne = TSNE(n_components=2, verbose=1)
    z = tsne.fit_transform(data)

    df = pd.DataFrame()
    df["labels"] = labels.flatten()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

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
        predicted = autoencoder.predict(x_val_fold)
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
        reconstructed = autoencoder.predict(np.expand_dims(original, axis=0), verbose=0)  # (1, 40, 40, 1)
        ssim = scaled_ssim(original, reconstructed[0])  # reconstructed[0] shape is (40, 40, 1)
        mse = mean_squared_error(original, reconstructed[0])
        ssim_scores.append(ssim)
        mse_scores.append(mse)
    avg_ssim = np.mean(ssim_scores)
    avg_mse = np.mean(mse_scores)
    avg_score = avg_ssim / (avg_mse + 1)
    return avg_ssim, avg_mse, avg_score


def histogram_25_75(autoencoder, x_train, y_train, latent_dim, subject):
    z = autoencoder.encoder(x_train)

    no_blink = []
    blink = []
    trans = []
    for i in range(0, len(y_train)):
        if y_train[i] == 0:
            no_blink.append(x_train[i])
        if y_train[i] == 1:
            blink.append(x_train[i])
        if y_train[i] == 2:
            trans.append(x_train[i])
    blink = np.array(blink)
    no_blink = np.array(no_blink)
    trans = np.array(trans)

    print('\nIl numero di blink è:', len(blink))
    print('Il numero di non blink è:', len(no_blink))
    print('Il numero di transizioni è:', len(trans))

    z_blink = autoencoder.encoder(blink)
    z_no_blink = autoencoder.encoder(no_blink)
    z_trans = autoencoder.encoder(trans)

    # Indici di dispersione.
    # Mediana: quantile di ordine 1/2
    # Quartili: quantili di ordine 1/4, 1/2 e 3/4
    # Percentili: quantili di ordine 1/100
    quantile_25 = np.quantile(z, .25, axis=0)  # Primo quartile (Q1)
    quantile_75 = np.quantile(z, .75, axis=0)  # Terzo quartile (Q3)

    num_rows = 7
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

    for j in range(0, z_blink.shape[1]):  # 0, 1,...,latent_dim-1
        true_blink = 0  # TP
        negative_blink = 0  # FN
        negative_noblink = 0  # FP
        true_noblink = 0  # TN
        for i in range(0, z_blink.shape[0]):  # 0, 1, ..., n_blinks-1
            if z_blink[i, j] <= quantile_25[j] or z_blink[i, j] >= quantile_75[j]:  # Out
                true_blink = true_blink + 1
            else:
                negative_blink = negative_blink + 1
        for i in range(0, z_no_blink.shape[0]):  # 0, 1, ..., n_non_blinks-1
            if quantile_25[j] < z_no_blink[i, j] < quantile_75[j]:
                true_noblink = true_noblink + 1
            else:
                negative_noblink = negative_noblink + 1

        # Disegno
        row_index = j // num_cols
        col_index = j % num_cols
        ax = axes[row_index, col_index]
        ax.set_title(f'histogram latent component {j}')
        ax.hist(z_blink[:, j], bins=100, color='green', alpha=0.6,
                label=f'Blink. TP:{true_blink}, FN:{negative_blink}')
        ax.hist(z_no_blink[:, j], bins=100, color='violet', alpha=0.6,
                label=f'No Blink. TN:{true_noblink}, FP:{negative_noblink}')
        ax.hist(z_trans[:, j], bins=100, color='yellow', alpha=0.6, label='Transition')

        ax.axvline(quantile_25[j], color='black', linestyle='-', label='Quantile 25')
        ax.axvline(quantile_75[j], color='blue', linestyle='-', label='Quantile 75')
        ax.legend()

    plt.tight_layout()

    histogram_file_name = f'histogram_25_75_{subject}_standard.png'
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
            M[:, i] = np.quantile(z, q, axis=0)
            i = i + 1
    quantile_matrix = M
    return quantile_matrix, z, z_blink, z_no_blink, z_trans


def histogram_cebicev(vae, x_test, y_test, latent_dim, subject):
    # La disuguaglianza triangolare di Chebyshev è un principio
    # che fornisce un limite superiore per la probabilità che una variabile casuale si discosti da un certo numero di
    # deviazioni standard rispetto alla sua media. In questo contesto specifico, la disuguaglianza triangolare di
    # Chebyshev viene utilizzata per stabilire i limiti superiore e inferiore per ciascuna componente latente in modo
    # da identificare regioni di interesse.
    #
    # La disuguaglianza triangolare di Chebyshev viene utilizzata per definire i
    # limiti (threshold_low e threshold_high) al di fuori dei quali i valori della componente latente sono
    # considerati "outliers" o "anomali". Questi limiti vengono utilizzati successivamente per valutare la presenza
    # di punti dati appartenenti a diverse classi (blink, no blink, transizione) nelle regioni specificate.
    #
    # In sostanza, questa approccio permette di individuare regioni di interesse nelle distribuzioni delle componenti
    # latenti in base a un criterio statistico che tiene conto della variabilità dei dati. Questo può essere utile
    # per identificare pattern o comportamenti anomali nei dati latenti, ad esempio, nel contesto di un modello di
    # variational autoencoder (VAE) utilizzato per la classificazione dei blink.
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

    # Use Cebicev's inequality thresholds instead of interquartile range
    cebicev_factor = 2.5  # Adjust this factor based on your requirements

    threshold_low = np.mean(z_mean, axis=0) - cebicev_factor * np.std(z_mean, axis=0)
    threshold_high = np.mean(z_mean, axis=0) + cebicev_factor * np.std(z_mean, axis=0)

    num_rows = 7
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

    for j in range(0, z_mean_blink.shape[1]):  # 0, 1,...,latent_dim-1
        true_blink = 0  # TP
        negative_blink = 0  # FN
        negative_noblink = 0  # FP
        true_noblink = 0  # TN
        for i in range(0, z_mean_blink.shape[0]):  # 0, 1, ..., n_blinks-1
            if z_mean_blink[i, j] <= threshold_low[j] or z_mean_blink[i, j] >= threshold_high[j]:  # Out
                true_blink = true_blink + 1
            else:
                negative_blink = negative_blink + 1
        for i in range(0, z_mean_no_blink.shape[0]):  # 0, 1, ..., n_non_blinks-1
            if threshold_low[j] < z_mean_no_blink[i, j] < threshold_high[j]:
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

        ax.axvline(threshold_low[j], color='black', linestyle='-', label='Cebicev Low')
        ax.axvline(threshold_high[j], color='blue', linestyle='-', label='Cebicev High')
        ax.legend()

    plt.tight_layout()

    histogram_file_name = f'histogram_cebicev_{subject}.png'
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


def auc_roc(quantile_matrix, z_blink, z_no_blink, subject):
    num_rows = 7
    num_cols = 4

    # max_auc = -1
    # relevant_indices = []
    auc_list = []  # Lista per salvare le coppie (indice, auc)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 20))

    for j in range(z_blink.shape[1]):
        FPR = []
        TPR = []
        for q in range(0, int(quantile_matrix.shape[1] / 2)):
            true_blink = 0  # TP
            negative_blink = 0  # FN
            negative_noblink = 0  # FP
            true_noblink = 0  # TN

            for i in range(0, z_blink.shape[0]):
                if z_blink[i, j] <= quantile_matrix[j, q] or z_blink[i, j] >= quantile_matrix[j, -q - 1]:
                    true_blink = true_blink + 1
                else:
                    negative_blink = negative_blink + 1

            for z in range(0, z_no_blink.shape[0]):
                if quantile_matrix[j, q] < z_no_blink[z, j] < quantile_matrix[j, -q - 1]:
                    true_noblink = true_noblink + 1
                else:
                    negative_noblink = negative_noblink + 1

            FPR.append(negative_noblink / (negative_noblink + true_noblink))  # FP/(FP+TN)
            TPR.append(true_blink / (true_blink + negative_blink))  # TP/(TP+FN)

        # AUC
        auc = roc_auc_score(np.concatenate((np.zeros(len(FPR)), np.ones(len(TPR)))), np.concatenate((FPR, TPR)))

        # Aggiungiamo l'indice e l'AUC alla lista
        auc_list.append((j, auc))

        """# Componente rilevante è quella con AUC massimo
        if auc > max_auc:
            max_auc = auc
            relevant_indices = [j]
        elif auc == max_auc:
            relevant_indices.append(j)"""

        # Calcoliamo l'indice di riga e colonna del subplot
        row_index = j // num_cols
        col_index = j % num_cols

        # Disegniamo la curva ROC sul subplot corrispondente
        ax = axes[row_index, col_index]
        ax.plot(FPR, TPR, color='b', label=f'Curva ROC - AUC: {auc:.2f}')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')

        ax.set_xlabel('False Positive Rate (FPR)')
        ax.set_ylabel('True Positive Rate (TPR)')
        ax.set_title(f'Colonna - latent component {j}')
        ax.legend(loc='lower right')

    # Impostiamo lo spazio tra i subplot
    plt.tight_layout()

    auc_roc_file_name = f"curve_auc_roc_{subject}_standard.png"
    fig.savefig(auc_roc_file_name)
    to_client.append(auc_roc_file_name)

    # Ordiniamo la lista in base ai valori di AUC in ordine decrescente
    auc_list = sorted(auc_list, key=lambda x: x[1], reverse=True)

    # return relevant_indices, max_auc
    return auc_list


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
    file_name = f"test_blinks_{subject}_standard.png"
    fig.savefig(file_name)
    to_client.append(file_name)


def save_test_blink_reconstructions(autoencoder, x_test, subject):
    # Usa questa funzione per salvare in un unico png tutte le ricostruzioni di
    # immagini di test con blink per un certo soggetto
    reconstructions = []
    for i in range(len(x_test)):
        original = x_test[i]
        reconstructed = autoencoder.predict(np.expand_dims(original, axis=0), verbose=0)  # (1, 40, 40, 1)
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
    file_name = f"test_blinks_reconstructed_{subject}_standard.png"
    fig.savefig(file_name)
    to_client.append(file_name)


def mask_single_test_sample(latent_component_indices, autoencoder, x_test, subject):
    # Usa questa funzione per salvare in un unico png un'unica immagine di test mascherata per un certo soggetto

    z = autoencoder.encoder(x_test, training=False)
    decoder_output = autoencoder.decoder(z, training=False)
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

    decoder_output_masked = autoencoder.decoder(z_masked, training=False)

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
    file_name = f"z{'_'.join(map(str, latent_component_indices))}_{strategy.lower()}_{subject}_standard.png"
    fig.savefig(file_name)
    to_client.append(file_name)


def mask_set(latent_component_indices, autoencoder, x_test, x_train, subject, is_train=False):
    """
    Questa funzione maschera i dati che vengono passati in input.
    Se i dati sono di train (is_train=True), allora salvo le ricostruzioni mascherate (e non) sotto forma di .npy
    Questo sarà utile per ricostruire il segnale EEG dalle topomap.
    Se i dati sono i test (is_train=False), allora salva anche l'immagine .png che mostra
    ciascuna immagine di test (con blink) mascherata
    """

    # Il mascheramento va fatto considerando la moda delle immagini di train SENZA blink
    x_train_no_blinks = get_x_train_no_blinks(x_train, y_train)
    z_train_no_blinks = autoencoder.encoder(x_train_no_blinks, training=False)

    # Create a figure with multiple subplots
    num_rows = 2
    if is_train:
        x_train_blinks = get_x_with_blinks(x_train, y_train)
        num_cols = len(x_train_blinks) // num_rows
    else:
        x_test_blinks = get_x_with_blinks(x_test, y_test)
        num_cols = len(x_test_blinks) // num_rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 7))

    strategy = "Median"
    for i, ax in enumerate(axs.ravel()):
        z_masked = np.copy(z_train_no_blinks)
        z_no_masked = np.copy(z_train_no_blinks)
        for latent_component_idx in latent_component_indices:
            median = np.median(z_train_no_blinks[:, latent_component_idx])
            z_masked[:, latent_component_idx] = median
            z_no_masked[:, latent_component_idx] = z_train_no_blinks[:, latent_component_idx]

        # Con mascheramento
        decoder_output_masked = autoencoder.decoder(z_masked, training=False)
        reconstructed_masked = decoder_output_masked[i]

        # Senza mascheramento (solo ricostruzione)
        decoder_output_no_masked = autoencoder.decoder(z_no_masked, training=False)
        reconstructed_no_masked = decoder_output_no_masked[i]

        # Salvo con mascheramento (masked reconstruction)
        masked_rec_standard_folder = f"masked_rec_standard/{subject}"
        os.makedirs(masked_rec_standard_folder, exist_ok=True)
        if is_train:
            # Salvo le ricostruzioni mascherate sotto forma di .npy, non mi interessa il png
            # Ad esempio, masked_red_standard/s01/x_train_15.npy
            file_path = f"{masked_rec_standard_folder}/x_train_{i}.npy"
            np.save(file_path, reconstructed_masked)
        else:
            # Salvo le ricostruzioni mascherate sotto forma di .npy
            # Ad esempio, masked_rec_standard/s01/x_train_15.npy
            file_path = f"{masked_rec_standard_folder}/x_test_{i}.npy"
            np.save(file_path, reconstructed_masked)

            # Salvo il png con tutte le ricostruzioni mascherate, nel caso di test set
            ax.imshow(reconstructed_masked, cmap="gray")
            ax.set_title(f"x_test[{i}]")
            ax.axis('off')
            fig.suptitle(f"{subject} test blinks. Mask strategy is: {strategy}", fontsize=26)
            # Ad esempio, z8_median_s01_standard.png
            file_name = f"z{'_'.join(map(str, latent_component_indices))}_{strategy.lower()}_{subject}_test_standard.png"
            fig.savefig(file_name)

        # Salvo SENZA mascheramento (solo ricostruzioni)
        rec_standard_folder = f"rec_standard/{subject}"
        os.makedirs(rec_standard_folder, exist_ok=True)
        if is_train:
            # Salvo le ricostruzioni NON mascherate sotto forma di .npy, non mi interessa il png
            # Ad esempio, rec_standard/s01/x_train_15.npy
            file_path = f"{rec_standard_folder}/x_train_{i}.npy"
            np.save(file_path, reconstructed_no_masked)
        else:
            # Salvo le ricostruzioni NON mascherate sotto forma di .npy
            # Ad esempio, rec_standard/s01/x_train_15.npy
            file_path = f"{rec_standard_folder}/x_test_{i}.npy"
            np.save(file_path, reconstructed_no_masked)

    if "masked_rec_standard/" not in to_client:
        to_client.append("masked_rec_standard/")
    if "rec_standard/" not in to_client:
        to_client.append("rec_standard/")


def get_x_train_no_blinks(x_train, y_train):
    blink_indices = np.where(y_train == 0)[0]  # Only on train samples labelled with NO BLINK (0)
    x_train = x_train[blink_indices]
    y_train = y_train[blink_indices]
    x_train_only_contains_no_blinks = all(y_train) == 0
    if not x_train_only_contains_no_blinks:
        raise Exception("Something went wrong while considering only no-blinks train images")
    return x_train

def get_x_with_blinks(x, y):
    blink_indices = np.where(y == 1)[0]  # Only samples labelled with BLINK (1)
    x = x[blink_indices]
    y = y[blink_indices]
    x_only_contains_blinks = all(y) == 1
    if not x_only_contains_blinks:
        raise Exception("Something went wrong while considering only blinks images")
    return x


if __name__ == '__main__':
    print("TensorFlow GPU usage:", tf.config.list_physical_devices('GPU'))

    # Ogni soggetto ha il proprio learning rate ottimale
    learning_rates = {
        "s01": 1e-05,
        "s02": 1e-05,
        "s03": 1e-05,
        "s04": 1e-05,
        # "s05": 1e-05, # Istogramma non significativo
        "s06": 1e-05,
        "s07": 1e-05,
        # "s08": 1e-07, # Istogramma non significativo
        "s09": 1e-05,
        # "s10": 1e-05  # Istogramma non significativo
    }

    # Dati ridotti al solo intorno del blink
    subject = "s01"
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
    encoder = EncoderStandard(latent_dimension)
    decoder = DecoderStandard()
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.compile(Adam(best_l_rate))
    autoencoder.train_on_batch(x_train[:1], x_train[:1])  # Very important, avoids harmful warnings
    autoencoder.load_weights(f"checkpoints/ae_{subject}")

    # The parameters must be the same before/after the load
    # check_weights_equality(f"w_before_{subject}_standard.pickle", autoencoder)

    """# avg_score for current combination (dbg)
    avg_score_dbg()

    # Save clusters - on server
    clusters = label_clusters(autoencoder, x_test, y_test)
    clusters_file_name = f"clusters_{subject}_standard.pickle"
    save_clusters(clusters_file_name)

    # Read clusters - only on client
    # show_clusters(clusters_file_name)

    # Save images - on server
    original, reconstructed = get_original_and_reconstructed(autoencoder, x_test, image_index=1)
    original_file_name = f"original_{subject}_standard.npy"
    reconstructed_file_name = f"reconstructed_{subject}_standard.npy"
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
    print(f"avg_score on test set: {avg_score:.4f}")"""

    # For each latent component a histogram is created for analyzing the test data distribution
    # The 25th and 75th percentiles are computed for each latent component in order to understand whether
    # Blinks are located outside tha range. For each histogram a confusion matrix is also computed
    quantile_matrix, z, z_blink, z_no_blink, _ = histogram_25_75(autoencoder, x_train, y_train, latent_dimension, subject)

    # For each latent component the ROC-AUC curve is created for detecting the quartile range which
    # Maximizes the TPR (True Positive Rate)
    auc_list = auc_roc(quantile_matrix, z_blink, z_no_blink, subject)
    # print(auc_list)
    k = 1
    top_k_indices = [pair[0] for pair in auc_list[:k]]
    # print("top_k_indices:", top_k_indices)

    # Mask relevant latent components
    # "Relevant" means large IQR (implies more variance of data) and many TP blinks (outside the IQR)
    # Se is_train=False, salva "{reconstructions_folder}/x_test_{i}.npy" e il file .png delle ricostruzioni mascherate
    # (e non). Altrimenti, salva soltanto i file .npy
    mask_set(top_k_indices, autoencoder, x_test, x_train, subject, is_train=False)  # Genera .npy e .png
    mask_set(top_k_indices, autoencoder, x_test, x_train, subject, is_train=True)  # Genera solo i .npy

    # x_test = get_x_test_blinks(x_test, y_test)
    # save_test_blink_originals(x_test, subject)
    # save_test_blink_reconstructions(autoencoder, x_test, subject)

    print(f"\nFinished. You can transfer to client: {to_client}")
