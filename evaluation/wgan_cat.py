import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
import seaborn as sns
from tensorflow import keras
import glob
from tensorflow.keras.preprocessing.image import load_img, img_to_array

with open("configs/cat_wgan.yaml", "r") as f:
    config = yaml.safe_load(f)

LATENT_DIM = config['model_params']['latent_dim']
save_generator = config['paths']['save_generator']
generator = tf.keras.models.load_model(save_generator, compile=False)
IMG_SIZE = 64

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon



code_A = np.random.normal(size=(1, LATENT_DIM))
code_B = np.random.normal(size=(1, LATENT_DIM))

steps = 10
plt.figure(figsize=(20, 4))
plt.suptitle("Interpolation dans l'espace latent (Chat A vers Chat B)", fontsize=14)

for i in range(steps):
    alpha = i / (steps - 1)
    inter_code = (1 - alpha) * code_A + alpha * code_B
    
    inter_img = generator.predict(inter_code, verbose=0)[0]
    
    inter_img = (inter_img * 127.5 + 127.5) / 255.0
    
    plt.subplot(1, steps, i + 1)
    plt.imshow(np.clip(inter_img, 0, 1))
    plt.axis('off')
    plt.title(f"{int(alpha*100)}%")
plt.show()