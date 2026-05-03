import numpy as np
import matplotlib
# Force le mode non-interactif avant d'importer pyplot
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml

# 1. Chargement de la config
with open("configs/cat_wgan.yaml", "r") as f:
    config = yaml.safe_load(f)

LATENT_DIM = config['model_params']['latent_dim']
save_generator = config['paths']['save_generator']

# 2. Chargement du modèle (compile=False suffit pour de l'inférence)
generator = tf.keras.models.load_model(save_generator, compile=False)

# 3. Création des points de départ et d'arrivée
code_A = np.random.normal(size=(1, LATENT_DIM))
code_B = np.random.normal(size=(1, LATENT_DIM))

steps = 10
# On crée un batch de 10 vecteurs interpolés d'un seul coup
alphas = np.linspace(0, 1, steps)
interpolated_codes = np.array([(1 - a) * code_A[0] + a * code_B[0] for a in alphas])

# 4. Génération en une seule passe (beaucoup plus rapide sur CPU)
inter_imgs = generator.predict(interpolated_codes, verbose=0)

# 5. Post-traitement (WGAN sort du [-1, 1], on veut du [0, 1])
inter_imgs = (inter_imgs * 0.5) + 0.5 
inter_imgs = np.clip(inter_imgs, 0, 1)

# 6. Visualisation et Sauvegarde
plt.figure(figsize=(20, 4))
plt.suptitle("Interpolation WGAN : Chat A vers Chat B", fontsize=14)

for i in range(steps):
    plt.subplot(1, steps, i + 1)
    plt.imshow(inter_imgs[i])
    plt.axis('off')
    plt.title(f"{int(alphas[i]*100)}%")

# REMPLACE plt.show() par savefig
plt.savefig("interpolation_wgan_result.png")
print("Image sauvegardée sous : interpolation_wgan_result.png")
plt.close()