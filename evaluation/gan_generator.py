import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

with open("configs/gan_config.yaml", "r") as f:
    config = yaml.safe_load(f)

latent_dim = config['model_params']['latent_dim']
generator = tf.keras.models.load_model(config['paths']['save_generator'])

def plot_gan_interpolation(steps=10):
    z_A = np.random.normal(size=(1, latent_dim))
    z_B = np.random.normal(size=(1, latent_dim))
    
    grid = np.zeros((32, 32 * steps))
    
    for i in range(steps):
        alpha = i / (steps - 1)
        inter_code = (1 - alpha) * z_A + alpha * z_B
        
        inter_img = generator.predict(inter_code, verbose=0)
        
        inter_img = (inter_img + 1.0) / 2.0
        grid[:, i*32:(i+1)*32] = inter_img.reshape(32, 32)
    
    plt.figure(figsize=(15, 3))
    plt.imshow(grid, cmap='gray')
    plt.title("Interpolation entre deux bruits aléatoires")
    plt.axis('off')
    plt.show()

def plot_latent_exploration(n_variations=10, epsilon=0.5):
    z_base = np.random.normal(size=(1, latent_dim))
    
    grid = np.zeros((32, 32 * n_variations))
    
    for i in range(n_variations):
        noise = np.random.normal(size=(1, latent_dim)) * epsilon
        z_variant = z_base + noise
        
        inter_img = generator.predict(z_variant, verbose=0)
        inter_img = (inter_img + 1.0) / 2.0
        grid[:, i*32:(i+1)*32] = inter_img.reshape(32, 32)

    plt.figure(figsize=(15, 3))
    plt.imshow(grid, cmap='gray')
    plt.title("Variations autour d'un même code")
    plt.axis('off')
    plt.show()

plot_gan_interpolation()
plot_latent_exploration()