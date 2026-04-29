import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K
import numpy as np
import yaml
from tensorflow.keras import losses, metrics

with open("configs/vae_config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Charger les données MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# On ajoute 2 pixels pour faire du 32*32
x_train = np.pad(x_train, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
x_test = np.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')

print(f"Nouvelle forme : {x_train.shape}")

latent_dim = config['model_params']['latent_dim']
conv_filters = config['model_params']['conv_filters']
k = config['model_params']['kernel_size']

""" sampling to decode """

class Sampling(layers.Layer):
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch,dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

""" encodeur """

encoder_input = layers.Input(
    shape=(32,32,1), name = "encoder_input"
)

x = layers.Conv2D( conv_filters[0] , (k,k), strides = 2, activation = 'relu', padding="same")(encoder_input)
x = layers.Conv2D( conv_filters[1] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2D( conv_filters[2] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
shape_before_flattening = K.int_shape(x)[1:]

x= layers.Flatten()(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder_output = layers.Dense(latent_dim, name="encoder_output")(x)

encoder = models.Model(encoder_input, [z_mean, z_log_var,z], name="encoder")

""" décodeur """

decoder_input = layers.Input(shape=(latent_dim,), name = "decoder_input")

x = layers.Dense(int(np.prod(shape_before_flattening)))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)

x = layers.Conv2DTranspose( conv_filters[2] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose( conv_filters[1] , (k,k), strides = 2, activation = 'relu', padding="same")(x)
x = layers.Conv2DTranspose( conv_filters[0] , (k,k), strides = 2, activation = 'relu', padding="same")(x)

decoder_output = layers.Conv2D(1,(k,k), strides = 1, activation = "sigmoid", padding= "same", name="decoder_output")(x)

decoder = models.Model(decoder_input, decoder_output)

z_mean, z_log_var, z = encoder(encoder_input)

reconstruction = decoder(z)

autoencoder = Model(inputs=encoder_input, outputs=reconstruction)

""" entainement VAE """

class VAE(models.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return[
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker]
        
    def call(self, inputs):
        z_mean, z_log_var, z = encoder(inputs)
        reconstruction = decoder(z)
        return z_mean, z_log_var,  reconstruction
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                500 *
                losses.binary_crossentropy(
                    data, reconstruction, axis=(1,2,3)
                )
            )
            kl_loss = tf.reduce_mean(
                tf.reduce_sum(
                    -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)),
                    axis = 1,
                )
            )
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {m.name: m.result() for m in self.metrics}

vae = VAE (encoder, decoder)

""" Paramètres d'entraînements du modèle """

learning_rate = config['training_params']['learning_rate']
batch_size = config['training_params']['batch_size']
epochs = config['training_params']['epochs']

vae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)

vae.fit(x_train, epochs=epochs, batch_size=batch_size, verbose=1)

save_encoder = config['paths']['save_encoder']
save_decoder = config['paths']['save_decoder']

encoder.save(save_encoder)
decoder.save(save_decoder)