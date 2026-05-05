import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import matplotlib.pyplot as plt

class MaskedConvLayer(layers.Layer):
    def __init__(self, mask_type, filters, kernel_size, **kwargs):
        super(MaskedConvLayer, self).__init__(**kwargs)
        self.mask_type = mask_type
        self.filters = filters
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")

    def build(self, input_shape):
        self.conv.build(input_shape)
        super(MaskedConvLayer, self).build(input_shape)

    def call(self, inputs):
        kh, kw, cin, cout = self.conv.kernel.shape
        center_h, center_w = kh // 2, kw // 2
        
        mask = np.ones((kh, kw, cin, cout), dtype='float32')
        mask[center_h + 1:, :, :, :] = 0.0
        mask[center_h, center_w + 1:, :, :] = 0.0
        if self.mask_type == 'A':
            mask[center_h, center_w, :, :] = 0.0
            
        masked_kernel = self.conv.kernel * tf.cast(mask, self.conv.kernel.dtype)
        return tf.nn.conv2d(inputs, masked_kernel, strides=1, padding="SAME") + self.conv.bias

def build_pixelcnn_simple():
    inputs = layers.Input(shape=(32, 32, 1))
    
    x = MaskedConvLayer(mask_type='A', filters=128, kernel_size=7)(inputs)
    x = layers.Activation('relu')(x)
    
    for _ in range(12):
        x = MaskedConvLayer(mask_type='B', filters=128, kernel_size=7)(x)
        x = layers.Activation('relu')(x)
        
    x = layers.Conv2D(128, 1, activation='relu')(x)
    x = layers.Conv2D(128, 1, activation='relu')(x)
    out = layers.Conv2D(1, 1, activation='sigmoid')(x)
    
    return models.Model(inputs, out)

model = build_pixelcnn_simple()
model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy')

(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = np.pad(x_train, ((0,0), (2,2), (2,2)), mode='constant')
x_train = (x_train > 127).astype('float32')[..., np.newaxis]

print("Entraînement en cours...")
model.fit(x_train, x_train, batch_size=128, epochs=10)

def generate(model, num=16):
    samples = np.zeros((num, 32, 32, 1), dtype='float32')
    print("Génération des images...")
    for i in range(32):
        for j in range(32):
            probs = model.predict(samples, verbose=0)[:, i, j, 0]
            samples[:, i, j, 0] = (np.random.rand(num) < probs).astype('float32')
        print(f"Ligne {i+1}/32", end='\r')
    
    plt.figure(figsize=(8, 8))
    for i in range(num):
        plt.subplot(4, 4, i+1)
        plt.imshow(samples[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

generate(model)