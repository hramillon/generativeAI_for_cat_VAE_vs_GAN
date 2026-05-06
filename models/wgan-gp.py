import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import yaml
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
# --- CONFIGURATION ---

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            # Permet de ne consommer que la VRAM nécessaire
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU détecté et activé : {gpus[0]}")
    except RuntimeError as e:
        print(e)

with open("configs/cat_wgan.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_dim = config['model_params']['latent_dim'] 
epochs = config['training_params']['epochs']
ressources = config['paths']['ressources']
batch_size = config['training_params']['batch_size']

# --- DATASET ---
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
    layers.RandomZoom(0.1),
])

def preprocess(img):
    # 1. Normalisation entre -1 et 1
    img = (tf.cast(img, "float32") - 127.5) / 127.5
    # 2. Augmentation (uniquement pendant l'entraînement)
    img = data_augmentation(img, training=True)
    return img

train = train_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train = train.prefetch(buffer_size=tf.data.AUTOTUNE)

# --- ARCHITECTURE ---
def build_generator(latent_dim):
    model = models.Sequential(name="generator")
    model.add(layers.Input(shape=(latent_dim,)))
    
    model.add(layers.Dense(4 * 4 * 1024))
    model.add(layers.Reshape((4, 4, 1024)))
    model.add(layers.ReLU())

    for filters in [512, 256, 128]:
        model.add(layers.UpSampling2D(size=(2, 2)))
        model.add(layers.Conv2D(filters, kernel_size=3, padding="same"))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

    model.add(layers.UpSampling2D(size=(2, 2)))
    model.add(layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh"))
    
    return model

def build_critic():
    model = models.Sequential(name="critic")
    model.add(layers.Input(shape=(64, 64, 3)))
    
    model.add(layers.Conv2D(64, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2D(128, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2D(256, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Conv2D(512, 4, strides=2, padding="same"))
    model.add(layers.LeakyReLU(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))
    return model

# --- WGAN-GP LOGIC ---
class WGAN(keras.Model):
    def __init__(self, generator, critic, latent_dim, gp_weight=10.0, n_critic=8):
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
        interpolated = real_images + alpha * (fake_images - real_images)
        
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)
        
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # Epsilon à 1e-12 pour la précision mathématique
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # 1. Entraînement du Critic (toujours n_critic fois)
        for i in range(self.n_critic):
            z = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(z, training=True)
                fake_logits = self.critic(fake_images, training=True)
                real_logits = self.critic(real_images, training=True)
                
                cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                c_loss = cost + gp * self.gp_weight
                
            grads = tape.gradient(c_loss, self.critic.trainable_variables)
            # Suppression du clip_by_global_norm : la GP s'occupe de la contrainte !
            self.c_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

        # 2. Entraînement du Générateur
        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            generated_images = self.generator(z, training=True)
            gen_logits = self.critic(generated_images, training=True)
            g_loss = -tf.reduce_mean(gen_logits)
            
        grads = tape.gradient(g_loss, self.generator.trainable_variables)
        # Suppression du clipping manuel ici aussi pour laisser le flot de gradient naturel
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        
        self.c_loss_metric.update_state(c_loss)
        self.g_loss_metric.update_state(g_loss)
        return {"c_loss": self.c_loss_metric.result(), "g_loss": self.g_loss_metric.result()}

# --- MONITORING (MODIFIÉ POUR 5 EPOCHS) ---
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=16, latent_dim=100):
        super().__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.seed = tf.random.normal([num_img, latent_dim])

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10 == 0 or epoch == 0:
            generated_images = self.model.generator(self.seed, training=False)
            generated_images = (generated_images * 127.5) + 127.5
            plt.figure(figsize=(8, 8))
            for i in range(self.num_img):
                plt.subplot(4, 4, i+1)
                img = keras.utils.array_to_img(generated_images[i])
                plt.imshow(img)
                plt.axis('off')
            plt.savefig(f"training/gen_img_epoch_{epoch+1}.png")
            plt.close()
            # Sauvegarde du checkpoint
            self.model.generator.save(f"training/wgan_gen_checkpoints/gen_epoch_{epoch+1}.keras")
            self.model.critic.save(f"training/wgan_cri_checkpoints/cri_epoch_{epoch+1}.keras")

# --- EXECUTION ---
os.makedirs("training/wgan_gen_checkpoints", exist_ok=True)
os.makedirs("training/wgan_cri_checkpoints", exist_ok=True)

generator = build_generator(latent_dim)
critic = build_critic()

generator.load_weights("training/wgan_gen_checkpoints/gen_epoch_100.keras")
critic.load_weights("training/wgan_cri_checkpoints/cri_epoch_100.keras")

wgan = WGAN(generator=generator, critic=critic, latent_dim=latent_dim)

wgan.compile(
    g_optimizer=Adam(learning_rate=0.00005, beta_1=0.0, beta_2=0.9),
    c_optimizer=Adam(learning_rate=0.00005, beta_1=0.0, beta_2=0.9)
)

img_monitor = GANMonitor(num_img=16, latent_dim=latent_dim)
history = wgan.fit(train, epochs=epochs, callbacks=[img_monitor])

save_gen_path = config['paths']['save_generator']
save_crit_path = config['paths']['save_discriminator']

generator.save(save_gen_path)
critic.save(save_crit_path)

plt.figure(figsize=(10, 5))
plt.plot(history.history["c_loss"], label="Critique")
plt.plot(history.history["g_loss"], label="Générateur")
plt.title("Évolution des Pertes (WGAN-GP)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("wgan_loss_history_fine_tunning.png")
plt.show()