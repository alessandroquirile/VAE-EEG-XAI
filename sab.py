import os
import pickle

import numpy as np

''' device setting '''
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # "0" first GPU
# "1" second GPU (se c'è altrimenti usa la CPU)
# "-1" CPU
import tensorflow as tf  # importare dopo aver settato il device

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()
tf.compat.v1.global_variables()
tf.executing_eagerly()
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Conv2D, Flatten, Dense, Lambda, Reshape, BatchNormalization, \
    Conv2DTranspose
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm
from keras import backend as K, Input, Model
from keras.initializers import he_uniform
from keras.losses import binary_crossentropy


def load_data(topomaps_folder: str, labels_folder: str, test_size, anomaly_detection):
    x, y = _create_dataset(topomaps_folder, labels_folder)

    print(f"Splitting data set into training set {1 - test_size} and test set {test_size}...")

    seed = 42

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
        _, x_test, _, y_test = train_test_split(x_remaining, y_remaining, test_size=test_size, random_state=seed)

        # Check dataset for anomaly detection task
        y_train_only_contains_label_0 = all(y_train) == 0
        y_test_only_contains_label_1_and_2 = all(label in [0, 1, 2] for label in y_test)
        if not y_train_only_contains_label_0 or not y_test_only_contains_label_1_and_2:
            raise Exception("Data was not loaded successfully")
    else:
        print("For latent space analysis")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)

    return x_train, x_test, y_train, y_test


def _create_dataset(topomaps_folder, labels_folder):
    topomaps_files = os.listdir(topomaps_folder)
    labels_files = os.listdir(labels_folder)

    topomaps_files.sort()
    labels_files.sort()

    x = []
    y = []

    n_files = len(topomaps_files)

    for topomaps_file, labels_file in tqdm(zip(topomaps_files, labels_files), total=n_files, desc="Loading data"):
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


def reduce_size(x, y, new_size):
    return x[:new_size], y[:new_size]


def expand(x):
    return np.expand_dims(x, -1).astype("float32")


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def compute_latent(x):
    mu, sigma = x
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    eps = K.random_normal(shape=(batch, latent_dim), mean=0., stddev=1.)
    return mu + K.exp(sigma) * eps


def kl_reconstruction_loss(true, pred):
    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    # KL divergence loss
    kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # Total loss = 50% rec + 50% KL divergence loss
    return K.mean(reconstruction_loss + kl_loss)


def recon_loss(true, pred):
    # Reconstruction loss
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height
    return K.mean(reconstruction_loss)


def latent_loss(inputs, outputs):
    # KL divergence loss
    kl_loss = 1 + sigma - K.square(mu) - K.exp(sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.mean(kl_loss)


def save(history):
    with open('history.pickle', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)


if __name__ == '__main__':
    print("TensorFlow GPU usage:", tf.config.list_physical_devices('GPU'))

    # Load data
    x_train, x_test, y_train, y_test = load_data("topomaps", "labels", 0.2, False)

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

    # Hyperparameters
    latent_dim = 25  # by Longo's paper
    epochs = 2500
    batch_size = 64

    # (40, 40, 1)
    img_width = x_train.shape[1]
    img_height = x_train.shape[2]
    input_shape = (img_width, img_height, 1)

    # For reproducibility
    seed = 42

    # Encoder
    encoder_input = Input(input_shape)
    encoder_conv1 = Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same",
                           kernel_initializer=he_uniform(seed))(encoder_input)
    bn1 = BatchNormalization()(encoder_conv1)

    encoder_conv2 = Conv2D(filters=128, kernel_size=3, activation="relu", strides=2, padding="same",
                           kernel_initializer=he_uniform(seed))(bn1)
    bn2 = BatchNormalization()(encoder_conv2)

    encoder_conv3 = Conv2D(filters=256, kernel_size=3, activation="relu", strides=2, padding="same",
                           kernel_initializer=he_uniform(seed))(bn2)
    bn3 = BatchNormalization()(encoder_conv3)
    flatten = Flatten()(bn3)
    dense = Dense(units=100, activation="relu")(flatten)
    mu = Dense(latent_dim)(dense)
    sigma = Dense(latent_dim)(dense)
    latent_space = Lambda(compute_latent)([mu, sigma])

    encoder = keras.Model(encoder_input, latent_space)
    # encoder.summary()

    # Decoder
    decoder_input = Input((latent_dim,))
    dense1 = Dense(units=4096, activation="relu")(decoder_input)
    bn1 = BatchNormalization()(dense1)
    dense2 = Dense(units=1024, activation="relu")(bn1)
    bn2 = BatchNormalization()(dense2)
    dense3 = Dense(units=4096, activation="relu")(bn2)
    bn3 = BatchNormalization()(dense3)
    reshape = Reshape((4, 4, 256))(bn3)
    deconv1 = Conv2DTranspose(filters=256, kernel_size=3, activation="relu", strides=2, padding="same",
                              kernel_initializer=he_uniform(seed))(reshape)
    bn4 = BatchNormalization()(deconv1)
    deconv2 = Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=1, padding="same",
                              kernel_initializer=he_uniform(seed))(bn4)
    bn5 = BatchNormalization()(deconv2)
    deconv3 = Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=2, padding="valid",
                              kernel_initializer=he_uniform(seed))(bn5)
    bn6 = BatchNormalization()(deconv3)
    deconv4 = Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=1, padding="valid",
                              kernel_initializer=he_uniform(seed))(bn6)
    bn7 = BatchNormalization()(deconv4)
    deconv5 = Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="valid",
                              kernel_initializer=he_uniform(seed))(bn7)
    bn8 = BatchNormalization()(deconv5)
    deconv6 = Conv2DTranspose(filters=1, kernel_size=2, activation="sigmoid", padding="valid",
                              kernel_initializer=he_uniform(seed))(bn8)
    decoder = Model(decoder_input, deconv6)
    # decoder.summary()

    # VAE
    vae = Model(encoder_input, decoder(encoder(encoder_input)))
    vae.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-6), loss=kl_reconstruction_loss,
                metrics=[recon_loss, latent_loss])

    # Training
    val_size = 0.2
    patience = 30
    print(f"latent_dim {latent_dim}, epochs {epochs}, batch_size {batch_size}, val_size is {val_size} of training "
          f"data, patience {patience}")
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='min')  # Valutare se verbose=1
    x_train, x_val, _, _ = train_test_split(x_train, y_train, test_size=val_size, random_state=seed)
    history = vae.fit(
        x_train,
        x_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, x_val),
        callbacks=[early_stopping]
    )
    save(history)
    vae.save_weights("checkpoints/vae")
