import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K
import numpy as np
import yaml
from tensorflow.keras import losses, metrics

# Charger les données MNIST
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# On ajoute 2 pixels pour faire du 32*32
x_train = np.pad(x_train, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
x_test = np.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')


with open("configs/gan_config.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_dim = config['model_params']['latent_dim']
conv_filters = config['model_params']['conv_filters']
k = config['model_params']['kernel_size']
epochs = config['training_params']['epochs']
learning_rate = config['training_params']['learning_rate']

""" discriminteur """

discrimintor_input = layers.Input(shape=(32,32,1))
x = layers.Conv2D(conv_filters[0], kernel_size=k, strides = 2, padding="same", use_bias = False)(discrimintor_input)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(conv_filters[1], kernel_size=k, strides = 2, padding="same", use_bias = False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(conv_filters[2], kernel_size=k, strides = 2, padding="same", use_bias = False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(conv_filters[3], kernel_size=k, strides = 2, padding="same", use_bias = False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(conv_filters[4], kernel_size=k, strides = 2, padding="same", use_bias = False)(x)
discriminator_output = layers.Flatten()(x)

discriminator = models.Model(discrimintor_input, discriminator_output)

""" générateur """

generator_input = layers.Input(shape=(100,))
x = layers.Reshape((1,1,100))(generator_input)
x = layers.Conv2DTranspose(conv_filters[3], kernel_size=k, strides = 2, padding="same", use_bias = False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(conv_filters[2], kernel_size=k, strides = 2, padding="same", use_bias = False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(conv_filters[1], kernel_size=k, strides = 2, padding="same", use_bias = False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
x = layers.Conv2DTranspose(conv_filters[0], kernel_size=k, strides = 2, padding="same", use_bias = False)(x)
x = layers.BatchNormalization(momentum=0.9)(x)
x = layers.LeakyReLU(0.2)(x)
generator_output = layers.Conv2DTranspose(conv_filters[4], kernel_size = k, strides = 2, padding = "same", user_bias = False, activation= tanh)(x)

generator = models.Model(generator_input, generator_output)

class DCGAN(models.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer):
        super(DCGAN, self).compile()
        self.loss_fn = losses.BinaryCrossentropy()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_metric = metrics.Mean(name="d_loss")
        self.g_loss_metric = metrics.Mean(name="g_loss")
    
    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
        
    def train_step(self, data):
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size,self.latent_dim))

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(
                random_latent_vectors, training = True
            )
            real_predictions = self.discriminator(real_images, training = True)
            fake_predictions = self.discriminator(generated_images, training= True)

            real_labels = tf.zeros_like(fake_predictions)
            real_noisy_labels = real_labels + 0.1 * tf.random.uniform(tf.shape(real_predictions))
            fake_labels = tf.zeros_like(fake_predictions)
            fake_noisy_labels = fake_labels - 0.1 * tf.random.uniform(tf.shape(fake_predictions))

            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)
            d_loss = (d_real_loss + d_fake_loss) / 2.0

            g_loss = self.loss_fn(real_labels, fake_predictions)
        
        gradient_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        gradient_of_generator     = gen_tape.gradient(g_loss, self.generator.trainable_variables)

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)

        return {m.name: m.result() for m in self.metrics}
dcgan =DCGAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
dcgan.compile(
    d_optimizer=optimizer.Adam(learning_rate=learning_rate, beta_1 = 0.5, beta_2 = 0.999),
    g_optimizer=optimizer.Adam(learning_rate=learning_rate, beta_1 = 0.5, beta_2 = 0.999),
)

dcgan.fit(train,epochs=epochs)


save_generator = config['paths']['save_generator']
save_discriminator = config['paths']['save_discriminator']

generator.save(save_generator)
discriminator.save(save_discriminator)