import tensorflow as tf
from keras import Sequential, Model
from keras.initializers import he_uniform
from keras.layers import BatchNormalization, Reshape, Dense, Conv2D, Flatten
from keras.layers import Layer
from keras.src.layers import MaxPooling2D, UpSampling2D
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


# MODELLO COME NEL PAPER
@keras.saving.register_keras_serializable()
class Encoder(Layer):
    def __init__(self, latent_dimension):
        super(Encoder, self).__init__()
        self.latent_dimension = latent_dimension

        seed = 42
        n_filters = 40

        self.conv1 = Sequential([
            Conv2D(filters=n_filters, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization(),
            MaxPooling2D((2, 2), padding='same')
        ])

        self.conv2 = Sequential([
            Conv2D(filters=n_filters * 2, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization(),
            MaxPooling2D((2, 2), padding='same')
        ])

        self.conv3 = Sequential([
            Conv2D(filters=n_filters * 3, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization(),
            MaxPooling2D((2, 2), padding='same')
        ])

        self.conv4 = Conv2D(filters=2, kernel_size=(3, 3), activation='relu', padding='same',
                            kernel_initializer=he_uniform(seed=seed))

        self.flatten = Flatten()

        self.z_mean = Dense(latent_dimension, name="z_mean")
        self.z_log_var = Dense(latent_dimension, name="z_log_var")
        self.sampling = sample

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


# MODELLO COME NEL PAPER
@keras.saving.register_keras_serializable()
class Decoder(Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense1 = Dense(units=5 * 5 * 2, activation="relu")
        self.reshape = Reshape((5, 5, 2))  # encoder.conv4 shape is (None, 5, 5, 2)

        seed = 42
        n_filters = 40

        self.deconv1 = Sequential([
            Conv2D(filters=n_filters * 3, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization(),
            UpSampling2D(size=(2, 2), data_format="channels_last")
        ])

        self.deconv2 = Sequential([
            Conv2D(filters=n_filters * 2, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization(),
            UpSampling2D(size=(2, 2), data_format="channels_last")
        ])

        self.deconv3 = Sequential([
            Conv2D(filters=n_filters, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization(),
            UpSampling2D(size=(2, 2), data_format="channels_last")
        ])

        self.deconv4 = Conv2D(filters=1, kernel_size=(3, 3), activation="sigmoid", padding="same",
                              kernel_initializer=he_uniform(seed))

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        outputs = self.deconv4(x)
        return outputs


class VAE(Model):
    def __init__(self, encoder, decoder, epochs=None, l_rate=None, batch_size=None, patience=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.epochs = epochs  # For grid search
        self.l_rate = l_rate  # For grid search
        self.batch_size = batch_size  # For grid search
        self.patience = patience  # For grid search
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

        # Update metrics
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

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# QUESTI SONO I MIEI MODELLI
"""@keras.saving.register_keras_serializable()
class Encoder(keras.layers.Layer):
    def __init__(self, latent_dimension):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dimension

        seed = 42

        # TODO: rimuovere l1
        self.conv1 = Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same",
                            kernel_initializer=he_uniform(seed), kernel_regularizer="l1")
        self.bn1 = BatchNormalization()

        self.conv2 = Conv2D(filters=128, kernel_size=3, activation="relu", strides=2, padding="same",
                            kernel_initializer=he_uniform(seed), kernel_regularizer="l1")
        self.bn2 = BatchNormalization()

        self.conv3 = Conv2D(filters=256, kernel_size=3, activation="relu", strides=2, padding="same",
                            kernel_initializer=he_uniform(seed), kernel_regularizer="l1")
        self.bn3 = BatchNormalization()

        self.flatten = Flatten()
        self.dense = Dense(units=100, activation="relu", kernel_regularizer="l1")

        self.z_mean = Dense(latent_dimension, name="z_mean")
        self.z_log_var = Dense(latent_dimension, name="z_log_var")

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
        # TODO: se durante training fai reparam; altrimenti restituisci solo le medie
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


@keras.saving.register_keras_serializable()
class Decoder(keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        self.dense1 = Dense(units=4096, activation="relu", kernel_regularizer="l1")
        self.bn1 = BatchNormalization()

        self.dense2 = Dense(units=1024, activation="relu", kernel_regularizer="l1")
        self.bn2 = BatchNormalization()

        self.dense3 = Dense(units=4096, activation="relu", kernel_regularizer="l1")
        self.bn3 = BatchNormalization()

        seed = 42

        self.reshape = Reshape((4, 4, 256))
        self.deconv1 = Conv2DTranspose(filters=256, kernel_size=3, activation="relu", strides=2, padding="same",
                                       kernel_initializer=he_uniform(seed), kernel_regularizer="l1")
        self.bn4 = BatchNormalization()

        self.deconv2 = Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=1, padding="same",
                                       kernel_initializer=he_uniform(seed), kernel_regularizer="l1")
        self.bn5 = BatchNormalization()

        self.deconv3 = Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=2, padding="valid",
                                       kernel_initializer=he_uniform(seed), kernel_regularizer="l1")
        self.bn6 = BatchNormalization()

        self.deconv4 = Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=1, padding="valid",
                                       kernel_initializer=he_uniform(seed), kernel_regularizer="l1")
        self.bn7 = BatchNormalization()

        self.deconv5 = Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="valid",
                                       kernel_initializer=he_uniform(seed), kernel_regularizer="l1")
        self.bn8 = BatchNormalization()

        self.deconv6 = Conv2DTranspose(filters=1, kernel_size=2, activation="sigmoid", padding="valid",
                                       kernel_initializer=he_uniform(seed), kernel_regularizer="l1")

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
        return decoder_outputs"""
