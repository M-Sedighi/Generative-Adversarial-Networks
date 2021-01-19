import numpy as np
import tensorflow as tf
import sys
import tensorflow.keras as keras

# loads data from mnist dataset to training and test sets
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_train = tf.cast(X_train, tf.float32)

# The generator is similar
# to an autoencoder’s decoder, and the discriminator is a regular binary classifier (it
# takes an image as input and ends with a Dense layer containing a single unit and
# using the sigmoid activation function).

codings_size = 30
generator = keras.models.Sequential([
    keras.layers.Dense(100, activation="selu", input_shape=[codings_size]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(28 * 28, activation="sigmoid"),
    keras.layers.Reshape([28, 28])
])
discriminator = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(150, activation="selu"),
    keras.layers.Dense(100, activation="selu"),
    keras.layers.Dense(1, activation="sigmoid")
])
gan = keras.models.Sequential([generator, discriminator])

# As the discriminator is a binary classifier, we
# can naturally use the binary cross-entropy loss. The generator will only be trained
# through the gan model, so we do not need to compile it at all. The gan model is also a
# binary classifier, so it can use the binary cross-entropy loss. Importantly, the discrimi‐
# nator should not be trained during the second phase, so we make it non-trainable
# before compiling the gan model

discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")
discriminator.trainable = False
gan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# Since the training loop is unusual, we cannot use the regular fit() method. Instead,
# we will write a custom training loop. For this, we first need to create a Dataset to
# iterate through the images

batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)


# training loop

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        for X_batch in dataset:
            # phase 1 - training the discriminator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            generated_images = generator(noise)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis=0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            # phase 2 - training the generator
            noise = tf.random.normal(shape=[batch_size, codings_size])
            y2 = tf.constant([[1.]] * batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)


train_gan(gan, dataset, batch_size, codings_size)
