import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers
from keras.optimizers.legacy import Adam
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.manifold import TSNE
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow import keras
from tqdm import tqdm


def load_data(topomaps_folder: str, labels_folder: str, test_size, anomaly_detection):
    x, y = _create_dataset(topomaps_folder, labels_folder)

    print(f"Splitting data set into training set {1 - test_size} and test set {test_size}...")

    if anomaly_detection:
        print("For anomaly detection")
        # Training set only contains images whose label is 0 for anomaly detection
        train_indices = np.where(y == 0)[0]
        x_train = x[train_indices]
        y_train = y[train_indices]

        # Split the remaining data into testing sets
        remaining_indices = np.where(y != 0)[0]
        x_remaining = x[remaining_indices]
        y_remaining = y[remaining_indices]
        _, x_test, _, y_test = train_test_split(x_remaining, y_remaining, test_size=test_size)

        # Check dataset for anomaly detection task
        y_train_only_contains_label_0 = all(y_train) == 0
        y_test_only_contains_label_1_and_2 = all(label in [0, 1, 2] for label in y_test)
        if not y_train_only_contains_label_0 or not y_test_only_contains_label_1_and_2:
            raise Exception("Data was not loaded successfully")
    else:
        print("For latent space analysis")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    return x_train, x_test, y_train, y_test


def _create_dataset(topomaps_folder, labels_folder):
    topomaps_files = os.listdir(topomaps_folder)
    labels_files = os.listdir(labels_folder)

    topomaps_files.sort()
    labels_files.sort()

    x = []
    y = []

    n_files = len(topomaps_files)

    for topomaps_file, labels_file in tqdm(zip(topomaps_files, labels_files), total=n_files, desc="Loading data set"):
        topomaps_array = np.load(f"{topomaps_folder}/{topomaps_file}")
        labels_array = np.load(f"{labels_folder}/{labels_file}")
        if topomaps_array.shape[0] != labels_array.shape[0]:
            raise Exception("Shapes must be equal")
        for i in range(topomaps_array.shape[0]):
            x.append(topomaps_array[i])
            y.append(labels_array[i])

    x = np.array(x)
    y = np.array(y)

    return x, y


def flat_mae(x, y):
    return mean_absolute_error(y.flatten(), x.flatten())


def sample(z_mean, z_log_var):
    """
    Generates random samples from a Gaussian distribution using the reparameterization trick.

    :param z_mean: (tf.Tensor) A tensor representing the mean of the distribution. Shape: [batch_size, dim]
    :param z_log_var: (tf.Tensor) A tensor representing the logarithm of the variance of the distribution. Shape: [batch_size, dim]
    :return: (tf.Tensor) A tensor containing the generated samples. Shape: [batch_size, dim]
    """
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    stddev = tf.exp(0.5 * z_log_var)
    return z_mean + stddev * epsilon


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
    image_size = data.shape[1]
    scale = 1.0

    # Create an empty figure to store the generated images
    figure = np.zeros((image_size * points_to_sample, image_size * points_to_sample))

    # Define linearly spaced coordinates corresponding to the 2D plot in the latent space
    grid_x = np.linspace(-scale, scale, points_to_sample)
    grid_y = np.linspace(-scale, scale, points_to_sample)[::-1]  # Reverse the order of grid_y

    # Apply t-SNE to the latent space
    z_mean, _, _ = vae.encoder.predict(data)
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
            x_decoded = vae.decoder.predict(np.expand_dims(z_sample, axis=0))

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


def plot_label_clusters(vae, data, labels):
    """
    Plots a t-SNE projection of the given data, with labels represented by different colors.

    :param vae: The trained VAE (Variational Autoencoder) model.
    :param data: Input data of shape (num_samples, 40, 40).
    :param labels: Array of labels corresponding to the data of shape (num_samples,).
    :return: None (displays the plot)
    """
    z_mean, _, _ = vae.encoder.predict(data)

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
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="comp-1", y="comp-2", hue=df.labels.tolist(),
                    palette=sns.color_palette("hls", n_colors)).set(title="Data t-SNE projection")
    plt.show()


def plot_metric(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric])
    plt.title(metric)
    plt.ylabel(metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()


def reduce_size(x, y, new_size):
    return x[:new_size], y[:new_size]


def expand(x):
    return np.expand_dims(x, -1).astype("float32")


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def grid_search_vae(x_train, latent_dimension):
    param_grid = {
        'epochs': [100, 200, 300, 400, 500, 600, 700],
        'l_rate': [0.001, 0.005, 0.009, 0.01, 0.05, 0.09, 0.1, 0.5, 0.9],
        'batch_size': [32, 64, 128, 256]
    }
    print("\nI am tuning the hyper parameters:", param_grid.keys())
    mae_scorer = make_scorer(flat_mae, greater_is_better=False)
    # refit=True gives problems when using GridSearchCV on non-sklearn models
    grid = GridSearchCV(
        VAEWrapper(encoder=Encoder(latent_dimension), decoder=Decoder()),
        param_grid, scoring=mae_scorer, cv=5, refit=False
    )
    grid.fit(x_train, x_train)
    return grid


def refit(fitted_grid, x_train, y_train, latent_dimension):
    # Since refit=True gives problems when using GridSearchCV on non-sklearn models
    # I refitted the best model manually
    print("\nRefitting based on:", fitted_grid.best_params_)

    best_epochs = fitted_grid.best_params_["epochs"]
    best_l_rate = fitted_grid.best_params_["l_rate"]
    best_batch_size = fitted_grid.best_params_["batch_size"]

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)

    encoder = Encoder(latent_dimension)
    decoder = Decoder()
    vae = VAE(encoder, decoder, best_epochs, best_l_rate, best_batch_size)
    vae.compile(Adam(best_l_rate))

    history = vae.fit(x_train, x_train, best_batch_size, best_epochs, validation_data=(x_val, x_val))
    return history, vae


def visually_check_reconstruction_skill(vae, x_test):
    image_index = 5
    plt.title(f"Original image {image_index}")
    original_image = x_test[image_index]
    plt.imshow(original_image, cmap="gray")
    plt.show()

    plt.title(f"Reconstructed image {image_index}, latent_dim = {latent_dimension}, batch_size = {vae.batch_size},"
              f"epochs = {vae.epochs}, l_rate = {vae.l_rate}")
    x_test_reconstructed = vae.predict(x_test)
    reconstructed_image = x_test_reconstructed[image_index]
    plt.imshow(reconstructed_image, cmap="gray")
    plt.show()


class VAEWrapper:
    def __init__(self, **kwargs):
        self.vae = VAE(**kwargs)
        self.vae.compile(Adam())

    def fit(self, x, y, **kwargs):
        self.vae.fit(x, y, **kwargs)

    def get_config(self):
        return self.vae.get_config()

    def get_params(self, deep):
        return self.vae.get_params(deep)

    def set_params(self, **params):
        return self.vae.set_params(**params)


class VAE(keras.Model, BaseEstimator):
    def __init__(self, encoder, decoder, epochs=None, l_rate=None, batch_size=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.epochs = epochs  # For grid search
        self.l_rate = l_rate  # For grid search
        self.batch_size = batch_size  # For grid search
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs)
        outputs = self.decoder(z)
        return outputs

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        data, labels = data
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Compute losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        # Compute gradient
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update my own metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        data, labels = data
        # Forward pass
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        # Compute losses
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss

        # Update my own metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


@keras.saving.register_keras_serializable()
class Encoder(keras.layers.Layer):
    def __init__(self, latent_dimension):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dimension

        self.conv1 = layers.Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(filters=128, kernel_size=3, activation="relu", strides=2, padding="same")
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(filters=256, kernel_size=3, activation="relu", strides=2, padding="same")
        self.bn3 = layers.BatchNormalization()

        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=100, activation="relu")

        self.z_mean = layers.Dense(latent_dimension, name="z_mean")
        self.z_log_var = layers.Dense(latent_dimension, name="z_log_var")

        self.sampling = sample

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.flatten(x)
        x = self.dense(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


@keras.saving.register_keras_serializable()
class Decoder(keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense1 = layers.Dense(units=4096, activation="relu")
        self.bn1 = layers.BatchNormalization()

        self.dense2 = layers.Dense(units=1024, activation="relu")
        self.bn2 = layers.BatchNormalization()

        self.dense3 = layers.Dense(units=4096, activation="relu")
        self.bn3 = layers.BatchNormalization()

        self.reshape = layers.Reshape((4, 4, 256))
        self.deconv1 = layers.Conv2DTranspose(filters=256, kernel_size=3, activation="relu", strides=2, padding="same")
        self.bn4 = layers.BatchNormalization()

        self.deconv2 = layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=1, padding="same")
        self.bn5 = layers.BatchNormalization()

        self.deconv3 = layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=2, padding="valid")
        self.bn6 = layers.BatchNormalization()

        self.deconv4 = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=1, padding="valid")
        self.bn7 = layers.BatchNormalization()

        self.deconv5 = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="valid")
        self.bn8 = layers.BatchNormalization()

        self.deconv6 = layers.Conv2DTranspose(filters=1, kernel_size=2, activation="sigmoid", padding="valid")

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.bn1(x)
        x = self.dense2(x)
        x = self.bn2(x)
        x = self.dense3(x)
        x = self.bn3(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.bn4(x)
        x = self.deconv2(x)
        x = self.bn5(x)
        x = self.deconv3(x)
        x = self.bn6(x)
        x = self.deconv4(x)
        x = self.bn7(x)
        x = self.deconv5(x)
        x = self.bn8(x)
        decoder_outputs = self.deconv6(x)
        return decoder_outputs


if __name__ == '__main__':
    # Load data
    x_train, x_test, y_train, y_test = load_data("topomaps", "labels", 0.2, False)

    # I am reducing the size of data set for speed purposes
    # Remove this in production
    new_size = 500
    x_train, y_train = reduce_size(x_train, y_train, new_size)
    x_test, y_test = reduce_size(x_test, y_test, new_size)

    # Expand dimensions to (None, 40, 40, 1)
    x_train = expand(x_train)
    x_test = expand(x_test)

    # Print data shapes
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Normalization
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # Grid search
    latent_dimension = 25  # Longo's paper
    fitted_grid = grid_search_vae(x_train, latent_dimension)

    # Refit
    history, vae = refit(fitted_grid, x_train, y_train, latent_dimension)

    # Plot learning curves
    plot_metric(history, "loss")
    plot_metric(history, "reconstruction_loss")
    plot_metric(history, "kl_loss")

    # plot_latent_space(vae, x_train)
    # plot_label_clusters(vae, x_train, y_train)

    # Check reconstruction skills against a random test sample
    visually_check_reconstruction_skill(vae, x_test)
