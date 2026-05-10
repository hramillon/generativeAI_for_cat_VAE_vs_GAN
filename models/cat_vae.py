import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K
import numpy as np
import yaml
from tensorflow.keras import losses, metrics

# ---------------------------------------------------------------------------
# PREPARE DATA AND CONFIGURATION
# ---------------------------------------------------------------------------
with open("configs/cat_vae_config.yaml", "r") as f:
    config = yaml.safe_load(f)

ressources = config['paths']['ressources']
batch_size = config['training_params']['batch_size']
IMAGE_SIZE = config['data_params']['image_size']
CHANNELS = config['data_params']['channels']
latent_dim = config['model_params']['latent_dim']
conv_filters = config['model_params']['conv_filters']
k = config['model_params']['kernel_size']

# 30k images of cat 64*64*3
train_data = tf.keras.utils.image_dataset_from_directory(
    ressources,
    labels=None,
    color_mode="rgb",
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

#data augmentation : small roation + horizontal
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
])

#normalization
def preprocess(img):
    img = tf.cast(img, "float32") / 255.0  
    return img

train = train_data.map(preprocess).map(lambda x: data_augmentation(x, training=True))
train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

""" sampling to decode """
# Very important for a VAE it shapes the latent space
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
# ---------------------------------------------------------------------------
# VAE MODEL
# ---------------------------------------------------------------------------
""" encodeur """
encoder_input = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS), name="encoder_input")
x = encoder_input
for filters in conv_filters:
    x = layers.Conv2D(filters, kernel_size=k, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

shape_before_flattening = K.int_shape(x)[1:]
x = layers.Flatten()(x)
x = layers.Dense(1024)(x)
x = layers.LeakyReLU()(x)

z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = models.Model(encoder_input, [z_mean, z_log_var, z], name="encoder")

""" décodeur """

decoder_input = layers.Input(shape=(latent_dim,), name="decoder_input")
x = layers.Dense(int(np.prod(shape_before_flattening)))(decoder_input)
x = layers.Reshape(shape_before_flattening)(x)

for filters in reversed(conv_filters):
    x = layers.Conv2DTranspose(filters, kernel_size=k, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

decoder_output = layers.Conv2DTranspose(CHANNELS, kernel_size=k, strides=1, padding="same", activation="sigmoid")(x)
decoder = models.Model(decoder_input, decoder_output, name="decoder")

# ---------------------------------------------------------------------------
# VAE CLASS
# ---------------------------------------------------------------------------

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
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]
        
    def train_step(self, data):
        # Gradient Tape permet de calculer les gradients d'un model pendant la passe avant 
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            reconstruction_loss = tf.reduce_mean(
                6000 * losses.binary_crossentropy(data, reconstruction), 
                axis=(1, 2)
            )
            #Beta = 6000
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            # perte = parte de reconstruction + divergence kl
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {m.name: m.result() for m in self.metrics}

""" Paramètres d'entraînements du modèle """

# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------

vae = VAE(encoder, decoder)
learning_rate = config['training_params']['learning_rate']
epochs = config['training_params']['epochs']

vae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

history = vae.fit(
    train, 
    epochs=epochs, 
    verbose=1
)

encoder.save(config['paths']['save_encoder'])
decoder.save(config['paths']['save_decoder'])