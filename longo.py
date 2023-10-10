from keras import Sequential
from keras.src.layers import MaxPooling2D, UpSampling2D

from vae import *


@keras.saving.register_keras_serializable()
class Encoder(keras.layers.Layer):
    def __init__(self, latent_dimension):
        super(Encoder, self).__init__()
        self.latent_dimension = latent_dimension

        seed = 42
        n_filters = 40

        self.conv1 = Sequential([
            Conv2D(filters=n_filters, kernel_size=3, activation="relu", strides=2, padding="same",
                   kernel_initializer=he_uniform(seed), kernel_regularizer="l1"),
            BatchNormalization(),
            MaxPooling2D((2, 2), padding='same')
        ])

        self.conv2 = Sequential([
            Conv2D(filters=n_filters * 2, kernel_size=3, activation="relu", strides=2, padding="same",
                   kernel_initializer=he_uniform(seed), kernel_regularizer="l1"),
            BatchNormalization(),
            MaxPooling2D((2, 2), padding='same')
        ])

        self.conv3 = Sequential([
            Conv2D(filters=n_filters * 3, kernel_size=3, activation="relu", strides=2, padding="same",
                   kernel_initializer=he_uniform(seed), kernel_regularizer="l1"),
            BatchNormalization(),
            MaxPooling2D((2, 2), padding='same')
        ])

        self.flatten = Flatten()
        self.dense = Dense(units=100, activation="relu", kernel_regularizer="l1")

        self.z_mean = Dense(latent_dimension, name="z_mean")
        self.z_log_var = Dense(latent_dimension, name="z_log_var")

        self.sampling = sample

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
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
        self.dense1 = Dense(units=40 * 40 * 1, activation="relu", kernel_regularizer="l1")
        self.reshape = Reshape((40, 40, 1))

        seed = 42
        n_filters = 40

        self.deconv1 = Sequential([
            Conv2D(filters=n_filters * 3, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed), kernel_regularizer="l1"),
            BatchNormalization(),
            UpSampling2D(size=(2, 2), data_format="channels_last")
        ])

        self.deconv2 = Sequential([
            Conv2D(filters=n_filters * 2, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed), kernel_regularizer="l1"),
            BatchNormalization(),
            UpSampling2D(size=(2, 2), data_format="channels_last")
        ])

        self.deconv3 = Sequential([
            Conv2D(filters=n_filters, kernel_size=(3, 3), activation="relu", padding="same",
                   kernel_initializer=he_uniform(seed), kernel_regularizer="l1"),
            BatchNormalization(),
            UpSampling2D(size=(2, 2), data_format="channels_last")
        ])

        self.deconv4 = Conv2D(filters=1, kernel_size=(3, 3), activation="relu", padding="same",
                              kernel_initializer=he_uniform(seed), kernel_regularizer="l1")

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.reshape(x)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        outputs = self.deconv4(x)
        return outputs


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

    # Print data shapes
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Normalization
    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # Grid search
    latent_dimension = 28
    grid = custom_grid_search(x_train, latent_dimension)  # By me
    # grid = grid_search_vae(x_train, latent_dimension)  # By sklearn

    # Manually refit, since refit=True raises problems with TensorFlow models
    history, vae = refit(grid, x_train, y_train, latent_dimension)
    save(history, subject)
    vae.save_weights(f"checkpoints/vae_{subject}", save_format='tf')

    # Questa parte serve per serializzare i pesi e verificare che a seguito del load
    # Essi siano uguali nel file analysis.py
    # dbg
    w_before = vae.get_weights()
    with open(f"w_before_{subject}.pickle", "wb") as fp:
        pickle.dump(w_before, fp)

    print(f"Finished training. You can transfer w_before_{subject}.pickle to client for consistency check")
