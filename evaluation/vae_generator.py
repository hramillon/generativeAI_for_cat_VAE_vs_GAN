import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tensorflow import keras
import seaborn as sns
from sklearn.manifold import TSNE

mnist = keras.datasets.mnist
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = np.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), mode='cosnstant')

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


with open("configs/vae_config.yaml", "r") as f:
    config = yaml.safe_load(f)
latent_dim = config['model_params']['latent_dim']
save_encoder = config['paths']['save_encoder']
save_decoder = config['paths']['save_decoder']

# Charger l'encodeur et le décodeur séparément (plus simple pour le VAE)
encoder = tf.keras.models.load_model(save_encoder, custom_objects={'Sampling': Sampling})
decoder = tf.keras.models.load_model(save_decoder)


""" carte des territoires  """

# 1. Obtenir les codes latents (on prend z_mean pour voir la structure apprise)
n_samples = 5000
z_mean, _, _ = encoder.predict(x_test[:n_samples])

# 2. Choisir quelques dimensions à afficher (par exemple les 6 premières)
num_dims_to_plot = min(6, latent_dim)

plt.figure(figsize=(15, 10))
plt.suptitle("Distribution des dimensions latentes (doit ressembler à N(0,1))", fontsize=16)

for i in range(num_dims_to_plot):
    plt.subplot(2, 3, i + 1)
    # Histogramme + Courbe de densité
    sns.histplot(z_mean[:, i], kde=True, color="skyblue", stat="density")
    
    # Superposer une vraie Normale Standard pour comparer
    x = np.linspace(-4, 4, 100)
    p = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)
    plt.plot(x, p, 'r--', lw=2, label='N(0,1) théorique')
    
    plt.title(f"Dimension latente n°{i+1}")
    plt.xlim(-4, 4)
    plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""  trajet entre deux chiffres réels"""
res_A = encoder.predict(x_test[0:1])
res_B = encoder.predict(x_test[1:2])
code_A = res_A[0]
code_B = res_B[0]

steps = 10
grid = np.zeros((32, 32 * steps))

for i in range(steps):
    alpha = i / (steps - 1)
    inter_code = (1 - alpha) * code_A + alpha * code_B
    inter_img = decoder.predict(inter_code)
    grid[:, i*32:(i+1)*32] = inter_img.reshape(32, 32)

plt.figure(figsize=(15, 5))
plt.title("Chiffre A vers Chiffre B")
plt.imshow(grid, cmap='gray')
plt.axis('off')
plt.show()

""" trajet vers random) """
code_A = encoder.predict(x_test[0:1])[0] 
code_B = np.random.normal(size=(1, latent_dim))

grid_random = np.zeros((32, 32 * steps))

for i in range(steps):
    alpha = i / (steps - 1)
    inter_code = (1 - alpha) * code_A + alpha * code_B
    inter_img = decoder.predict(inter_code)
    grid_random[:, i*32:(i+1)*32] = inter_img.reshape(32, 32)

plt.figure(figsize=(15, 5))
plt.title("Morphing : Réel vers Aléatoire")
plt.imshow(grid_random, cmap='gray')
plt.axis('off')
plt.show()