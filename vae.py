import os

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers
from keras.optimizers.legacy import Adam
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm


def load_data(topomaps_folder: str, labels_folder: str, test_size, anomaly_detection):
    """
    Load data returning training set and test set

    :param topomaps_folder: (str) Path to the folder containing topomaps
    :param labels_folder: (str) Path to the folder containing labels
    :param test_size: Test set size
    :return: x_train, x_test, y_train, y_test
    """

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
    """
    Creates a data set
    :param topomaps_folder: (str) Path to the folder containing topomaps
    :param labels_folder: (str) Path to the folder containing labels
    :return: data and labels
    """
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
    :param data: Data to have a latent representation of. Shape should be (num_samples, 32, 32).
    :param points_to_sample: The number of points to sample along each axis of the plot. Default is 30.
    :param figsize: The size of the figure (width and height) in inches. Default is 15.
    :return: None (displays the plot).
    """
    image_size = 32
    scale = 1.0

    # Create an empty figure to store the generated images
    # Width: image_size * points_to_sample (default 32x15 = 480)
    # Height: image_size * points_to_sample (default 32x15 = 480)
    figure = np.zeros((image_size * points_to_sample, image_size * points_to_sample))

    # Define linearly spaced coordinates corresponding to the 2D plot in the latent space
    grid_x = np.linspace(-scale, scale, points_to_sample)
    grid_y = np.linspace(-scale, scale, points_to_sample)[::-1]  # Reverse the order of grid_y

    # Reshape data to (num_samples, 32*32) instead of (num_samples, 32, 32)
    # data = data.reshape(data.shape[0], -1)

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
    :param data: Input data of shape (num_samples, 32, 32).
    :param labels: Array of labels corresponding to the data of shape (num_samples,).
    :return: None (displays the plot)
    """
    z_mean, _, _ = vae.encoder.predict(data)

    # Reshape data to (num_samples, 32x32) instead of (num_samples, 32, 32)
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


class VAE(keras.Model):
    """
    Variational Autoencoder (VAE) model implementation.

    This class represents a VAE model, which consists of an encoder and a decoder.
    It inherits from the `keras.Model` class.

    Attributes:

    - encoder (keras.Model): The encoder model responsible for encoding input data into latent space.
    - decoder (keras.Model): The decoder model responsible for decoding latent space representations into output data.
    - total_loss_tracker (keras.metrics.Mean): A metric tracker for the total loss of the VAE during training (reconstruction + kl).
    - reconstruction_loss_tracker (keras.metrics.Mean): A metric tracker for the reconstruction loss component of the VAE during training.
    - kl_loss_tracker (keras.metrics.Mean): A metric tracker for the KL divergence loss component of the VAE during training.

    Methods:

    - train_step(data): Performs a single training step on the VAE model.
    """

    def __init__(self, encoder, decoder, **kwargs):
        """
        Initializes a VAE model instance.

        :param encoder: The encoder model for the VAE.
        :param decoder: The decoder model for the VAE.
        :param kwargs: Additional keyword arguments to be passed to the base `keras.Model` class constructor.
        """
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def call(self, inputs, training=None, mask=None):
        _, _, z = self.encoder(inputs)
        outputs = self.decoder(z)
        return outputs

    @property
    def metrics(self):
        """
        Returns a list of metrics tracked by the VAE model during training.
        :return: List of metrics tracked by the VAE model.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """
        Performs a single training step on the VAE model.

        :param data: Input data batch for training the VAE.
        :return: Dictionary containing the updated metric values.
        """
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
        """
        Performs a forward pass and computes losses for a single testing step in a Variational Autoencoder.

        Parameters:
            - data (tensor): The input data for testing.

        Returns:
            dict: A dictionary containing the computed losses for the testing step.

        Example usage:
            test_data = ...
            losses = model.test_step(test_data)

        Note:
            This method is typically used as part of the testing/evaluation loop in a Variational Autoencoder model.

        """
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


class Encoder(keras.Model):
    """
    Encoder model implementation.

    This class represents an encoder model, which encodes input data into a latent space.
    It inherits from the `keras.Model` class.

    Attributes:

    - latent_dim (int): The dimensionality of the latent space.
    - conv_block1 (keras.Sequential): Sequential model for the first convolutional block.
    - conv_block2 (keras.Sequential): Sequential model for the second convolutional block.
    - conv_block3 (keras.Sequential): Sequential model for the third convolutional block.
    - flatten (keras.layers.Flatten): Flatten layer.
    - dense (keras.layers.Dense): Dense layer for the output.
    - z_mean (keras.layers.Dense): Dense layer for the mean of the latent space.
    - z_log_var (keras.layers.Dense): Dense layer for the log variance of the latent space.
    - sampling (function): Function for sampling from the latent space distribution.

    Methods:

    - call(inputs, training=None, mask=None): Executes a forward pass on the encoder.
    """

    def __init__(self, latent_dimension, input_shape):
        """
        Initializes an Encoder model instance.

        :param latent_dimension: The dimensionality of the latent space.
        """
        super(Encoder, self).__init__()
        self.latent_dim = latent_dimension
        self.conv_block1 = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same"),
            layers.BatchNormalization()
        ])
        self.conv_block2 = keras.Sequential([
            layers.Conv2D(filters=128, kernel_size=3, activation="relu", strides=2, padding="same"),
            layers.BatchNormalization()
        ])
        self.conv_block3 = keras.Sequential([
            layers.Conv2D(filters=256, kernel_size=3, activation="relu", strides=2, padding="same"),
            layers.BatchNormalization()
        ])
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(units=100, activation="relu")
        self.z_mean = layers.Dense(latent_dimension, name="z_mean")
        self.z_log_var = layers.Dense(latent_dimension, name="z_log_var")
        self.sampling = sample

    def call(self, inputs, training=None, mask=None):
        """
        Executes a forward pass on the encoder.

        :param inputs: Input data to the encoder.
        :param training: Boolean flag indicating whether the model is in training mode.
        :param mask: Mask tensor.
        :return: Tuple containing the mean, log variance, and sampled representation from the latent space.
        """
        x = self.conv_block1(inputs)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.dense(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(keras.Model):
    """
    Decoder model implementation.

    This class represents a decoder model, which decodes latent space representations into output data.
    It inherits from the `keras.Model` class.

    Attributes:

    - latent_dim (int): The dimensionality of the latent space.
    - dense1 (keras.layers.Dense): Dense layer 1.
    - dense2 (keras.layers.Dense): Dense layer 2.
    - dense3 (keras.layers.Dense): Dense layer 3.
    - reshape (keras.layers.Reshape): Reshape layer.
    - deconv1 (keras.layers.Conv2DTranspose): Convolutional transpose layer 1.
    - deconv2 (keras.layers.Conv2DTranspose): Convolutional transpose layer 2.
    - deconv3 (keras.layers.Conv2DTranspose): Convolutional transpose layer 3.
    - deconv4 (keras.layers.Conv2DTranspose): Convolutional transpose layer 4.
    - deconv5 (keras.layers.Conv2DTranspose): Convolutional transpose layer 5.
    - deconv6 (keras.layers.Conv2DTranspose): Convolutional transpose layer 6.

    Methods:

    - call(inputs, training=None, mask=None): Executes a forward pass on the decoder.
    """

    def __init__(self):
        """
        Initializes a Decoder model instance.

        """
        super(Decoder, self).__init__()
        self.dense1 = keras.Sequential([
            layers.Dense(units=4096, activation="relu"),
            layers.BatchNormalization()
        ])
        self.dense2 = keras.Sequential([
            layers.Dense(units=1024, activation="relu"),
            layers.BatchNormalization()
        ])
        self.dense3 = keras.Sequential([
            layers.Dense(units=4096, activation="relu"),
            layers.BatchNormalization()
        ])
        self.reshape = layers.Reshape((4, 4, 256))
        self.deconv1 = keras.Sequential([
            layers.Conv2DTranspose(filters=256, kernel_size=3, activation="relu", strides=2, padding="same"),
            layers.BatchNormalization()
        ])
        self.deconv2 = keras.Sequential([
            layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=1, padding="same"),
            layers.BatchNormalization()
        ])
        self.deconv3 = keras.Sequential([
            layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=2, padding="valid"),
            layers.BatchNormalization()
        ])
        self.deconv4 = keras.Sequential([
            layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=1, padding="valid"),
            layers.BatchNormalization()
        ])
        self.deconv5 = keras.Sequential([
            layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="valid"),
            layers.BatchNormalization()
        ])
        self.deconv6 = layers.Conv2DTranspose(filters=1, kernel_size=2, activation="sigmoid", padding="valid")

    def call(self, inputs, training=None, mask=None):
        """
        Executes a forward pass on the decoder.

        :param inputs: Input data to the decoder.
        :param training: Boolean flag indicating whether the model is in training mode.
        :param mask: Mask tensor.
        :return: Output data generated by the decoder.
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        decoder_outputs = self.deconv6(x)
        return decoder_outputs


if __name__ == '__main__':
    # tf.config.run_functions_eagerly(True)

    # Load data
    x_train, x_test, y_train, y_test = load_data("topomaps", "labels", 0.2, False)

    # I am reducing the size of data set for speed purposes
    x_train = x_train[:500]
    y_train = y_train[:500]
    x_test = x_test[:500]
    y_test = y_test[:500]

    # Expand dimensions to (None, 40, 40, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Print data shapes
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Normalize the data
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Compiling the VAE
    latent_dimension = 25  # Longo's paper
    encoder = Encoder(latent_dimension, (40, 40, 1))
    decoder = Decoder()
    vae = VAE(encoder, decoder)
    vae.compile(Adam(learning_rate=0.001))

    # Training
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    epochs = 100
    batch_size = 128
    history = vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val,))

    # Plot learning curves
    plot_metric(history, "loss")
    plot_metric(history, "reconstruction_loss")
    plot_metric(history, "kl_loss")

    # plot_latent_space(vae, x_train)
    # plot_label_clusters(vae, x_train, y_train)

    # Check reconstruction skills against a random test sample
    image_index = 5
    plt.title(f"Original image {image_index}")
    original_image = x_test[image_index]
    plt.imshow(original_image, cmap="gray")
    plt.show()

    plt.title(f"Reconstructed image {image_index}, latent_dim = {latent_dimension}, epochs = {epochs}, "
              f"batch_size = {batch_size}")
    x_test_reconstructed = vae.predict(x_test)
    reconstructed_image = x_test_reconstructed[image_index]
    plt.imshow(reconstructed_image, cmap="gray")
    plt.show()
