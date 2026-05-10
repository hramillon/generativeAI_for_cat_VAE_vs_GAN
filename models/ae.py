import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K
import numpy as np
import yaml

with open("configs/ae_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ---------------------------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------------------------
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# On ajoute 2 pixels pour faire du 32*32
x_train = np.pad(x_train, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
x_test = np.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')

latent_dim = config['model_params']['latent_dim']
conv_filters = config['model_params']['conv_filters']
k = config['model_params']['kernel_size']

# ---------------------------------------------------------------------------
# AUTOENCODER MODEL
# ---------------------------------------------------------------------------

""" encodeur """

encoder_input = layers.Input(
    shape=(32,32,1), name = "encoder_input"
)
# 32 -> 16 -> 8
x = layers.Conv2D( conv_filters[0] , (k,k), strides = 2, activation = 'relu', padding="same")(encoder_input)
x = layers.Conv2D( conv_filters[1] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D( conv_filters[2] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x= layers.Flatten()(x)
# fin de l'encodeur on encode sur la dimension de l'espace latent pour pouvoir l'echantillonner
encoder_output = layers.Dense(latent_dim, name="encoder_output")(x)

encoder = models.Model(encoder_input, encoder_output)

""" décodeur """
#inverse exacte de l'encodeur
decoder_input = layers.Input(shape=(latent_dim,), name = "decoder_input")

x = layers.Dense(int(np.prod(shape_before_flattening)))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)

x = layers.Conv2DTranspose( conv_filters[2] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose( conv_filters[1] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose( conv_filters[0] , (k,k), strides = 2, activation = 'relu', padding="same")(x)

decoder_output = layers.Conv2D(1,(k,k), strides = 1, activation = "sigmoid", padding= "same", name="decoder_output")(x)

decoder = models.Model(decoder_input, decoder_output)

latent_code = encoder(encoder_input)

reconstruction = decoder(latent_code)

#on définit le model qui prend du 32*32*1 et en sortie la reconstruction donc du 32*32*1 aussi 
autoencoder = Model(inputs=encoder_input, outputs=reconstruction)

""" Paramètres d'entraînements du modèle """

# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------

learning_rate = config['training_params']['learning_rate']
batch_size = config['training_params']['batch_size']
epochs = config['training_params']['epochs']

autoencoder.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, verbose=1)

test_loss, test_accuracy = autoencoder.evaluate(x_test, x_test, verbose=0)
print(f"Exactitude du test : {test_accuracy:.4f}")

paths = config['paths']['save_path']

autoencoder.save(paths)