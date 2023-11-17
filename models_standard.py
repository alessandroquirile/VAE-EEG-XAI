import tensorflow as tf
from keras import Sequential, Model
from keras.initializers import he_uniform
from keras.layers import BatchNormalization, Reshape, Dense, Conv2D, Flatten
from keras.layers import Layer
from keras.src import losses
from keras.src.layers import Conv2DTranspose
from tensorflow import keras

class Autoencoder(Model):
    def __init__(self, encoder, decoder, epochs=None, l_rate=None, batch_size=None, patience=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.epochs = epochs  # For grid search
        self.l_rate = l_rate  # For grid search
        self.batch_size = batch_size  # For grid search
        self.patience = patience  # For grid search
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs, training=None, mask=None):
        bottleneck = self.encoder(inputs)
        reconstruction = self.decoder(bottleneck)
        return reconstruction

    @property
    def metrics(self):
        return [
            self.loss_tracker
        ]

    def train_step(self, data):
        data, labels = data
        with tf.GradientTape() as tape:
            # Forward pass
            bottleneck = self.encoder(data)
            reconstruction = self.decoder(bottleneck)

            # Compute losses
            loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )

        # Compute gradient
        grads = tape.gradient(loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update metrics
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result()
        }

    def test_step(self, data):
        data, labels = data

        # Forward pass
        bottleneck = self.encoder(data)
        reconstruction = self.decoder(bottleneck)

        # Compute losses
        loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
            )
        )

        # Update metrics
        self.loss_tracker.update_state(loss)

        return {
            "loss": self.loss_tracker.result()
        }


@keras.saving.register_keras_serializable()
class EncoderStandard(Layer):
    def __init__(self, latent_dimension):
        super(EncoderStandard, self).__init__()
        self.latent_dim = latent_dimension

        seed = 42

        self.conv1 = Sequential([
            Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization()
        ])

        self.conv2 = Sequential([
            Conv2D(filters=128, kernel_size=3, activation="relu", strides=2, padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization()
        ])

        self.conv3 = Sequential([
            Conv2D(filters=256, kernel_size=3, activation="relu", strides=2, padding="same",
                   kernel_initializer=he_uniform(seed)),
            BatchNormalization()
        ])

        self.flatten = Flatten()
        self.dense = Dense(units=latent_dimension, activation="relu")

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


@keras.saving.register_keras_serializable()
class DecoderStandard(Layer):
    def __init__(self):
        super(DecoderStandard, self).__init__()

        seed = 42

        self.dense = Sequential([
            Dense(units=4096, activation="relu"),
            BatchNormalization()
        ])

        self.reshape = Reshape((4, 4, 256))

        self.deconv1 = Sequential([
            Conv2DTranspose(filters=256, kernel_size=3, activation="relu", strides=2, padding="same",
                            kernel_initializer=he_uniform(seed)),
            BatchNormalization()
        ])

        self.deconv2 = Sequential([
            Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=1, padding="same",
                            kernel_initializer=he_uniform(seed)),
            BatchNormalization()
        ])

        self.deconv3 = Sequential([
            Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=2, padding="valid",
                            kernel_initializer=he_uniform(seed)),
            BatchNormalization()
        ])

        self.deconv4 = Sequential([
            Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=1, padding="valid",
                            kernel_initializer=he_uniform(seed)),
            BatchNormalization()
        ])

        self.deconv5 = Sequential([
            Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="valid",
                            kernel_initializer=he_uniform(seed)),
            BatchNormalization()
        ])

        self.deconv6 = Conv2DTranspose(filters=1, kernel_size=2, activation="sigmoid", padding="valid",
                                       kernel_initializer=he_uniform(seed))

    def call(self, inputs, training=None, mask=None):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        return x
