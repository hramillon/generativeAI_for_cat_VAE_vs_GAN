import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os

# --- UTILITAIRES ---
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# --- ARCHITECTURE ---
class MaskedConvLayer(layers.Layer):
    """
    Couche de convolution masquée pour respecter la causalité des pixels.
    Type A : Ne voit pas le pixel central (utilisé en entrée).
    Type B : Voit le pixel central (utilisé dans les couches cachées).
    """
    def __init__(self, mask_type, filters, kernel_size, **kwargs):
        super(MaskedConvLayer, self).__init__(**kwargs)
        self.mask_type = mask_type
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=self.filters, 
            kernel_size=self.kernel_size, 
            padding="same"
        )
        self.conv.build(input_shape)
        
        kh, kw, cin, cout = self.conv.kernel.shape
        mask = np.ones((kh, kw, cin, cout), dtype='float32')
        center_h, center_w = kh // 2, kw // 2
        #supprime ligne
        mask[center_h + 1:, :, :, :] = 0.0
        #pareil colonne
        mask[center_h, center_w + 1:, :, :] = 0.0
        if self.mask_type == 'A':
            #  supprime le pixel actuel si on est type A
            mask[center_h, center_w, :, :] = 0.0
            
        self.mask = tf.constant(mask, dtype=tf.float32)
        super(MaskedConvLayer, self).build(input_shape)

    def call(self, inputs):
        masked_kernel = self.conv.kernel * self.mask
        return tf.nn.conv2d(inputs, masked_kernel, strides=1, padding="SAME") + self.conv.bias

def build_pixelcnn(config):
    cfg = config['model']
    inputs = layers.Input(shape=cfg['input_shape'])
    
    x = MaskedConvLayer(mask_type='A', filters=cfg['filters'], kernel_size=cfg['kernel_size'])(inputs)
    x = layers.Activation('relu')(x)
    
    for _ in range(cfg['num_layers']):
        x = MaskedConvLayer(mask_type='B', filters=cfg['filters'], kernel_size=cfg['kernel_size'])(x)
        x = layers.Activation('relu')(x)
        
    x = layers.Conv2D(128, 1, activation='relu')(x)
    out = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    return models.Model(inputs, out)

def main():
    config_path = "configs/pixelcnn_config.yaml"
    if not os.path.exists(config_path):
        print(f"Erreur: {config_path} introuvable.")
        return
    config = load_config(config_path)

    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = np.pad(x_train, ((0,0), (2,2), (2,2)), mode='constant')
    x_train = (x_train > 127).astype('float32')[..., np.newaxis]

    model = build_pixelcnn(config)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='binary_crossentropy'
    )

    model.fit(
        x_train, x_train, 
        batch_size=config['training']['batch_size'], 
        epochs=config['training']['epochs']
    )

    generate_images(model, config)

def generate_images(model, config):
    num = config['generation']['num_samples']
    h, w, c = config['model']['input_shape']
    samples = np.zeros((num, h, w, c), dtype='float32')
    
    for i in range(h):
        for j in range(w):
            probs = model.predict(samples, verbose=0)[:, i, j, 0]
            samples[:, i, j, 0] = (np.random.rand(num) < probs).astype('float32')
    
    # Visualisation
    plt.figure(figsize=(6, 6))
    for i in range(num):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()