import tensorflow as tf
from tensorflow.keras import layers, models, K, Model
import numpy as np
import yaml

with open("configs/ae_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Charger les données MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normaliser les données
x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

# One-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

latent_dim = config['model_params']['latent_dim']
conv_filters = config['model_params']['conv_filters']
k = config['model_params']['kernel_size']

""" encodeur """

encoder_input = layers.Input(
    shape=(28,28,1), name = "encoder_input"
)

x = layers.Conv2D( conv_filters[0] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D( conv_filters[1] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D( conv_filters[2] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x= layers.Flatten()(x)
encoder_output = layers.Dense(latent_dim, name="encoder_output")(x)

encoder = models.Model(encoder_input, encoder_output)

""" décodeur """

decoder_input = layers.Input(shape=(latent_dim,), name = "decoder_input")

x = layers.Dense(np.prod(shape_before_flattening))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)

x = layers.Conv2DTranspose( conv_filters[2] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose( conv_filters[1] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose( conv_filters[0] , (k,k), strides = 2, activation = 'relu', padding="same")(x)

decoder_output = layers.Conv2D(1,(3,3), strides = 1, activation = "sigmoid", padding= "same", name="decoder_output")(x)

decoder = (decoder_input, decoder_output)

autoencoder = Model(encoder_input, decoder(encoder_output))

""" Paramètres d'entraînements du modèle """

learning_rate = config['training_params']['learning_rate']
batch_size = config['training_params']['epochs']
epochs = config['training_params']['batch_size']

autoencoder.compile(
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Exactitude du test : {test_accuracy:.4f}")

paths = config['paths']['save_path']

model.save(paths)