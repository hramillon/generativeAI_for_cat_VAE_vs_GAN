import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tensorflow import keras
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

""" on pépare le dataset et le modèle """
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

x_train = np.pad(x_train, ((0,0), (2,2), (2,2), (0,0)), mode='constant')
x_test = np.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')

with open("configs/ae_config.yaml", "r") as f:
    config = yaml.safe_load(f)

ae = tf.keras.models.load_model('training/ae_mnist.keras')
decoder = ae.get_layer('functional_1')
encoder = ae.get_layer('functional')

""" carte des territoires """

# PCA #

n_samples = 2000
codes_latents = encoder.predict(x_test[:n_samples])

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#pca = PCA(n_components=2)
codes_2d = tsne.fit_transform(codes_latents)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(codes_2d[:, 0], codes_2d[:, 1], c=y_test[:n_samples], cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10))
plt.title("Projection 2D de l'espace latent")
plt.xlabel("Composante Principale 1")
plt.ylabel("Composante Principale 2")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

""" Trajet dans le vide """
 
code_A = encoder.predict(x_test[0:1])
code_B = encoder.predict(x_test[1:2])

# 2. Créer 10 étapes entre A et B
steps = 10
grid = np.zeros((32, 32 * steps))

for i in range(steps):
    alpha = i / (steps - 1)
    # Mélange linéaire des deux codes
    inter_code = (1 - alpha) * code_A + alpha * code_B
    inter_img = decoder.predict(inter_code)
    grid[:, i*32:(i+1)*32] = inter_img.reshape(32, 32)

plt.figure(figsize=(15, 5))
plt.imshow(grid, cmap='gray')
plt.axis('off')
plt.show()

latent_dim = config['model_params']['latent_dim']
code_A = encoder.predict(x_test[0:1])
code_B = np.random.normal(size=(1, latent_dim))

# 2. Créer 10 étapes entre A et B
steps = 10
grid = np.zeros((32, 32 * steps))

for i in range(steps):
    alpha = i / (steps - 1)
    # Mélange linéaire des deux codes
    inter_code = (1 - alpha) * code_A + alpha * code_B
    inter_img = decoder.predict(inter_code)
    grid[:, i*32:(i+1)*32] = inter_img.reshape(32, 32)

plt.figure(figsize=(15, 5))
plt.imshow(grid, cmap='gray')
plt.axis('off')
plt.show()