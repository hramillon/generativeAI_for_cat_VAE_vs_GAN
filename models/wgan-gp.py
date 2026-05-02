import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Model
import tensorflow.keras.backend as K
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tensorflow.keras import losses, metrics

with open("configs/cat_wgan.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_dim = config['model_params']['latent_dim']
conv_filters = config['model_params']['conv_filters']
k = config['model_params']['kernel_size']
epochs = config['training_params']['epochs']
learning_rate = config['training_params']['learning_rate']
ressources = config['paths']['ressources']
batch_size = config['training_params']['batch_size']

train_data = tf.keras.utils.image_dataset_from_directory(
    ressources,
    labels=None,
    color_mode="rgb",
    image_size=(64, 64),
    batch_size=batch_size,
    shuffle=True,
    seed=42
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
])

def preprocess(img):
    img = (tf.cast(img, "float32") - 127.5) / 127.5
    return img

train = train_data.map(preprocess).map(lambda x: data_augmentation(x, training=True))
train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

def build_generator(latent_dim):
    model = models.Sequential(name="generator")
    
    model.add(layers.Dense(4 * 4 * 512, input_dim=latent_dim))
    model.add(layers.Reshape((4, 4, 512)))
    
    model.add(layers.UpSampling2D(size=(2, 2), interpolation="bilinear"))
    model.add(layers.Conv2D(256, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.UpSampling2D(size=(2, 2), interpolation="bilinear"))
    model.add(layers.Conv2D(128, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.UpSampling2D(size=(2, 2), interpolation="bilinear"))
    model.add(layers.Conv2D(64, 3, padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.UpSampling2D(size=(2, 2), interpolation="bilinear"))
    model.add(layers.Conv2D(3, 3, padding="same", activation="tanh"))
    return model

def build_critic():
    model = models.Sequential(name="critic")
    
    model.add(layers.Conv2D(64, 4, strides=2, padding="same", input_shape=(64, 64, 3)))
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2D(128, 4, strides=2, padding="same"))
    model.add(layers.LayerNormalization()) 
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2D(256, 4, strides=2, padding="same"))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2D(512, 4, strides=2, padding="same"))
    model.add(layers.LayerNormalization())
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(1)) 
    
    return model

class WGAN(keras.Model):
    def __init__(self, generator, critic, latent_dim, gp_weight=10.0, n_critic=5):
        super().__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.gp_weight = gp_weight
        self.n_critic = n_critic

    def compile(self, g_optimizer, c_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.c_optimizer = c_optimizer
        self.c_loss_metric = keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    def gradient_penalty(self, batch_size, real_images, fake_images):
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        
        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]

        for i in range(self.n_critic):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.critic(fake_images, training=True)
                real_logits = self.critic(real_images, training=True)

                cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                c_loss = cost + gp * self.gp_weight

            c_grad = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_grad, self.critic.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(random_latent_vectors, training=True)
            gen_img_logits = self.critic(generated_images, training=True)
            g_loss = -tf.reduce_mean(gen_img_logits)

        g_grad = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))
        
        self.c_loss_metric.update_state(c_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"c_loss": self.c_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

generator = build_generator(latent_dim)
critic = build_critic()

wgan = WGAN(generator=generator, critic=critic, latent_dim=latent_dim)

wgan.compile(
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9),
    c_optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.0, beta_2=0.9)
)

history = wgan.fit(train, epochs=epochs)

save_gen_path = config['paths']['save_generator']
save_crit_path = config['paths']['save_discriminator']

generator.save(save_gen_path)
critic.save(save_crit_path)

plt.figure(figsize=(10, 5))
plt.plot(history.history["c_loss"], label="Critique (C)")
plt.plot(history.history["g_loss"], label="Générateur (G)")
plt.title("Évolution des Pertes (WGAN-GP)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("wgan_loss_history.png")
plt.show()