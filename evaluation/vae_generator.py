import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow as tf
import yaml

with open("configs/vae_config.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_dim = config['model_params']['latent_dim']
save_encoder = config['paths']['save_encoder']
save_decoder = config['paths']['save_decoder']

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

encoder = tf.keras.models.load_model(
    save_encoder, 
    custom_objects={"Sampling": Sampling}
)
decoder = tf.keras.models.load_model(save_decoder)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = np.pad(x_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')

def plot_vae_results(n_samples=8):
    originals = x_test[:n_samples]
    _, _, latent_points = encoder.predict(originals)
    reconstructions = decoder.predict(latent_points)

    random_latent = np.random.normal(size=(n_samples, latent_dim))
    generated = decoder.predict(random_latent)

    steps = n_samples
    start_point = latent_points[0]
    end_point = latent_points[1]  
    
    lin_spaced = np.linspace(0, 1, steps)
    interp_latent = np.array([(1-t)*start_point + t*end_point for t in lin_spaced])
    transitions = decoder.predict(interp_latent)

    fig, axs = plt.subplots(3, n_samples, figsize=(n_samples*2, 6))
    
    for i in range(n_samples):
        axs[0, i].imshow(reconstructions[i].reshape(32, 32), cmap="gray")
        axs[0, i].set_title("Reconst.")
        axs[0, i].axis("off")
        
        axs[1, i].imshow(generated[i].reshape(32, 32), cmap="gray")
        axs[1, i].set_title("Random")
        axs[1, i].axis("off")

        axs[2, i].imshow(transitions[i].reshape(32, 32), cmap="gray")
        axs[2, i].set_title(f"T={lin_spaced[i]:.1f}")
        axs[2, i].axis("off")

    plt.tight_layout()
    plt.show()

plot_vae_results()