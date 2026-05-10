import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K
import numpy as np
import yaml
from tensorflow.keras import losses, metrics
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PREPARE DATA AND CONFIGURATION
# ---------------------------------------------------------------------------

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = (x_train.reshape(-1, 28, 28, 1).astype('float32') - 127.5) / 127.5
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# On ajoute 2 pixels pour faire du 32*32
x_train = np.pad(x_train, ((0,0), (2,2), (2,2), (0,0)), mode='constant', constant_values=-1)
x_test = np.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')


with open("configs/gan_config.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_dim = config['model_params']['latent_dim']
conv_filters = config['model_params']['conv_filters']
k = config['model_params']['kernel_size']
epochs = config['training_params']['epochs']
learning_rate = config['training_params']['learning_rate']
batch_size = config['training_params']['batch_size']

# ---------------------------------------------------------------------------
# GAN MODEL
# ---------------------------------------------------------------------------

""" discriminteur """

def build_discriminator():
    model = models.Sequential(name="discriminator")
    
    model.add(layers.Conv2D(32, kernel_size=3, strides=2, input_shape=(32, 32, 1), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(layers.ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.25))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

""" générateur """

def build_generator(latent_dim):
    model = models.Sequential(name="generator")
    
    model.add(layers.Dense(8 * 8 * 128, input_dim=latent_dim))
    model.add(layers.Reshape((8, 8, 128)))
    
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.LeakyReLU(alpha=0.2))
    
    model.add(layers.Conv2D(1, kernel_size=4, padding="same", activation="tanh"))
    return model

discriminator = build_discriminator()
generator = build_generator(latent_dim)

# ---------------------------------------------------------------------------
# GAN CLASS
# ---------------------------------------------------------------------------

class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        # dans le DCGAN on utilise la binary cross entropy (à rediscuter)
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")
    
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
        
    def train_step(self, data):
        real_images = data 
        batch_size = tf.shape(real_images)[0]
        # d'abord pour entrainer on echantillone un lot de vecteur de l'espace multivariée
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # le générateur utilise les vecteurs pour faire des images
            generated_images = self.generator(random_latent_vectors, training=True)
            
            # on demande l'avis du critique sur les images réelles
            real_predictions = self.discriminator(real_images, training=True)
            # puis sur les imagesinventées
            fake_predictions = self.discriminator(generated_images, training=True)

            real_labels = tf.ones_like(real_predictions) 
            fake_labels = tf.zeros_like(fake_predictions)

            real_noisy_labels = real_labels - 0.1 * tf.random.uniform(tf.shape(real_predictions))
            fake_noisy_labels = fake_labels + 0.1 * tf.random.uniform(tf.shape(fake_predictions))

            # Loss = perte d'entropie croisée sur les réelles + les fausses (critique)
            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0
            #loss = entropie croisée binaire entre prédictions du discriminateur et une étiquette ayant la valeur 1
            g_loss = self.loss_fn(real_labels, fake_predictions)
        
        grad_d = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        grad_g = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        #on met à jour les poids séparément 
        self.d_optimizer.apply_gradients(zip(grad_d, self.discriminator.trainable_variables))
        self.g_optimizer.apply_gradients(zip(grad_g, self.generator.trainable_variables))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}

# ---------------------------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------------------------

dcgan = DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
dcgan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5),
    g_optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5),
)

history = dcgan.fit(x_train, epochs=epochs, batch_size=batch_size)


save_generator = config['paths']['save_generator']
save_discriminator = config['paths']['save_discriminator']

generator.save(save_generator)
discriminator.save(save_discriminator)

# ---------------------------------------------------------------------------
# GRAPHIQUE(S)
# ---------------------------------------------------------------------------

plt.figure(figsize=(10, 5))
plt.plot(history.history["d_loss"], label="Discriminateur (D)")
plt.plot(history.history["g_loss"], label="Générateur (G)")
plt.title("Évolution des Pertes (Loss) pendant l'entraînement")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Sauvegarde du graphique
plt.savefig("gan_loss_history.png")
plt.show()