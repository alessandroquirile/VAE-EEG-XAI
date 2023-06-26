import numpy as np
import tensorflow as tf
from keras import layers
from matplotlib import pyplot as plt
from tensorflow import keras


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


def plot_latent_space(vae, points_to_sample=30, figsize=15):
    """
    Plots the latent space of a Variational Autoencoder (VAE).
    This function generates a 2D manifold plot of digits in the latent space
    of the VAE. Each point in the plot represents a digit generated by the VAE's
    decoder model based on a specific location in the latent space.

    :param vae: The trained VAE model.
    :param points_to_sample: The number of points to sample along each axis of the plot. Default is 30.
    :param figsize: The size of the figure (width and height) in inches. Default is 15.
    :return: None (displays the plot).
    """
    image_size = 32
    scale = 1.0
    figure = np.zeros((image_size * points_to_sample, image_size * points_to_sample, 3))
    # linearly spaced coordinates corresponding to the 2D plot in the latent space
    grid_x = np.linspace(-scale, scale, points_to_sample)
    grid_y = np.linspace(-scale, scale, points_to_sample)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(image_size, image_size, 3)
            figure[
            i * image_size: (i + 1) * image_size,
            j * image_size: (j + 1) * image_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = image_size // 2
    end_range = points_to_sample * image_size + start_range
    pixel_range = np.arange(start_range, end_range, image_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


def plot_label_clusters(vae, data, labels):
    """
    Plots the digit classes in the latent space of a Variational Autoencoder (VAE).

    This function generates a 2D plot of the digit classes in the latent space
    of the VAE. Each point in the plot represents a digit from the given data,
    projected onto the latent space based on the encoder model of the VAE.

    :param vae: The trained VAE model.
    :param data: The input data containing the digits.
    :param labels: The corresponding labels for the digits.
    :return: None (displays the plot).
    """
    z_mean, _, _ = vae.encoder.predict(data)

    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


class VAE(keras.Model):
    """
    Variational Autoencoder (VAE) model implementation.

    This class represents a VAE model, which consists of an encoder and a decoder.
    It inherits from the `keras.Model` class.

    Attributes:

    - encoder (keras.Model): The encoder model responsible for encoding input data into latent space.
    - decoder (keras.Model): The decoder model responsible for decoding latent space representations into output data.
    - total_loss_tracker (keras.metrics.Mean): A metric tracker for the total loss of the VAE during training.
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
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

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
    - conv1 (keras.layers.Conv2D): Convolutional layer 1.
    - conv2 (keras.layers.Conv2D): Convolutional layer 2.
    - flatten (keras.layers.Flatten): Flatten layer.
    - dense1 (keras.layers.Dense): Dense layer 1.
    - z_mean (keras.layers.Dense): Dense layer for the mean of the latent space.
    - z_log_var (keras.layers.Dense): Dense layer for the log variance of the latent space.
    - sampling (function): Function for sampling from the latent space distribution.

    Methods:

    - call(inputs, training=None, mask=None): Executes a forward pass on the encoder.
    """

    def __init__(self, latent_dimension):
        """
        Initializes an Encoder model instance.

        :param latent_dimension: The dimensionality of the latent space.
        """
        super(Encoder, self).__init__()
        self.latent_dim = latent_dimension
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(units=16, activation="relu")
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
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.dense1(x)
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
    - dense (keras.layers.Dense): Dense layer.
    - reshape (keras.layers.Reshape): Reshape layer.
    - deconv1 (keras.layers.Conv2DTranspose): Convolutional transpose layer 1.
    - deconv2 (keras.layers.Conv2DTranspose): Convolutional transpose layer 2.
    - deconv3 (keras.layers.Conv2DTranspose): Convolutional transpose layer 3.

    Methods:

    - call(inputs, training=None, mask=None): Executes a forward pass on the decoder.
    """

    def __init__(self, latent_dimension):
        """
        Initializes a Decoder model instance.

        :param latent_dimension: The dimensionality of the latent space.
        """
        super(Decoder, self).__init__()
        self.latent_dim = latent_dimension
        self.dense = layers.Dense(units=8 * 8 * 64, activation="relu")
        self.reshape = layers.Reshape((8, 8, 64))
        self.deconv1 = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")
        self.deconv2 = layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu", strides=2, padding="same")
        self.deconv3 = layers.Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same")

    def call(self, inputs, training=None, mask=None):
        """
        Executes a forward pass on the decoder.

        :param inputs: Input data to the decoder.
        :param training: Boolean flag indicating whether the model is in training mode.
        :param mask: Mask tensor.
        :return: Output data generated by the decoder.
        """
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        decoder_outputs = self.deconv3(x)
        return decoder_outputs


latent_dimension = 2

encoder = Encoder(latent_dimension)
encoder_inputs = keras.Input(shape=(32, 32, 3))  # shape is [None, 32, 32, 3] = [batch_size, height, width, depth]
z_mean, z_log_var, z = encoder(encoder_inputs)

decoder = Decoder(latent_dimension)
latent_inputs = keras.Input(shape=(latent_dimension,))
decoder_outputs = decoder(latent_inputs)

(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.legacy.Adam())
vae.fit(x_train, epochs=2, batch_size=128)

plot_latent_space(vae)

plot_label_clusters(vae, x_train, y_train)
