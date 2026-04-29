import tensorflow as tf
from tensorflow import keras
import numpy as np


# Charger les données MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

""" encodeur """

encoder_input = layers.Input(
    shape=(28,28,1), name = "encoder_input"
)

x = layers.Conv2D(32 , (3,3), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D(64 , (3,3), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D(128 , (3,3), strides = 2, activation = 'relu', padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x= layers.Flatten()(x)
encoder_output = layers.Dense(10, name="encoder_output")(x)

encoder = models.Model(encoder_input, encoder_output)

""" décodeur """

decoder_input = layers.Input(shape=(10,), name = "decoder_input")

x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)

x = layers.Conv2DTranspose(128, (3, 3), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose(64 , (3,3), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose(32 , (3,3), strides = 2, activation = 'relu', padding="same")(x)

decoder_output = layers.Conv2D(1,(3,3), strides = 1, activation = "sigmoid", padding= "same", name="decoder_output")(x)

decoder = (decoder_input, decoder_output)

autoencoder = Model(encoder_input, decoder(encoder_output))

autoencoder.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

batch_size = 100
model.fit(x_train, y_train, epochs=10, batch_size=batch_size, verbose=1)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Exactitude du test : {test_accuracy:.4f}")

model.save('training/cnn.keras')