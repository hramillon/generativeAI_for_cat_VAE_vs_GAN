import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K
import numpy as np
import yaml
from tensorflow.keras import losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

with open("configs/cat_vae_config.yaml", "r") as f:
    config = yaml.safe_load(f)

ressources = config['paths']['ressources']
batch_size = config['training_params']['batch_size']

train_data = tf.keras.utils.image_dataset_from_directory(
    ressources,
    labels=None,
    color_mode="rgb",
    image_size=(64, 64),
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    interpolation="bilinear"
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
])

def preprocess(img):
    img = tf.cast(img, "float32") / 255.0  
    return img

train = train_data.map(lambda x: preprocess(x))
train = train.map(lambda x: data_augmentation(x, training=True))

train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

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
IMAGE_SIZE = config['data_params']['image_size']
CHANNELS = config['data_params']['channels']

encoder_input = layers.Input(
    shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")

# Bloc 1 : 64x64 -> 32x32
x = layers.Conv2D(conv_filters[0], kernel_size=k, strides=2, padding="same")(encoder_input)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# Bloc 2 : 32x32 -> 16x16
x = layers.Conv2D(conv_filters[1], kernel_size=k, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# Bloc 3 : 16x16 -> 8x8
x = layers.Conv2D(conv_filters[2], kernel_size=k, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# Bloc 4 : 8x8 -> 4x4
x = layers.Conv2D(conv_filters[3], kernel_size=k, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# [4, 4, 256]
shape_before_flattening = K.int_shape(x)[1:]

x = layers.Flatten()(x)

x = layers.Dense(1024)(x)
x = layers.LeakyReLU()(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])

encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

""" décodeur """

decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")

x = layers.Dense(int(np.prod(shape_before_flattening)))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)

# Bloc 1 : 4x4 -> 8x8
x = layers.Conv2DTranspose(conv_filters[3], kernel_size=k, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# Bloc 2 : 8x8 -> 16x16
x = layers.Conv2DTranspose(conv_filters[2], kernel_size=k, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# Bloc 3 : 16x16 -> 32x32
x = layers.Conv2DTranspose(conv_filters[1], kernel_size=k, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

# Bloc 4 : 32x32 -> 64x64
x = layers.Conv2DTranspose(conv_filters[0], kernel_size=k, strides=2, padding="same")(x)
x = layers.BatchNormalization()(x)
x = layers.LeakyReLU()(x)

decoder_output = layers.Conv2DTranspose(
    CHANNELS, kernel_size=k, strides=1, padding="same", activation="sigmoid", name="decoder_output"
)(x)

decoder = models.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

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
                2000 *
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
epochs = config['training_params']['epochs']

vae.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
)

early_stopping = EarlyStopping(
    monitor="total_loss",
    patience=5,
    mode="min",
    restore_best_weights=True
)

vae.fit(train, epochs=epochs, batch_size=batch_size, verbose=1)

history = vae.fit(
    train, 
    epochs=epochs, 
    callbacks=[early_stopping],
    verbose=1
)

save_encoder = config['paths']['save_encoder']
save_decoder = config['paths']['save_decoder']

encoder.save(save_encoder)
decoder.save(save_decoder)