import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import Layer, Dense, Conv2D, LeakyReLU, Activation, Flatten, Reshape, UpSampling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
tf.keras.backend.set_floatx('float32')

LATENT_DIM = 512
RESSOURCES = "ressources/"
OUTPUT_DIR = "training"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FILTERS = [512, 512, 512, 512, 256, 128, 64]

# ---------------------------------------------------------------------------
# COUCHES PERSONNALISÉES
# ---------------------------------------------------------------------------

class PixelNormalization(Layer):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)
    
    def call(self, inputs):
        mean_square = tf.reduce_mean(tf.square(inputs), axis=-1, keepdims=True)
        l2 = tf.math.rsqrt(mean_square + 1.0e-8)
        return inputs * l2

class WeightScaling(Layer):
    def __init__(self, shape, gain=np.sqrt(2), **kwargs):
        super(WeightScaling, self).__init__(**kwargs)
        fan_in = tf.math.reduce_prod(shape)
        self.wscale = gain * tf.math.rsqrt(tf.cast(fan_in, tf.float32))
      
    def call(self, inputs):
        return inputs * self.wscale

class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(shape=(input_shape[-1],), initializer='zeros', trainable=True, name="bias")

    def call(self, inputs):
        return inputs + self.bias

class WeightScalingDense(Layer):
    def __init__(self, n_units, gain, use_pixelnorm=False, activate=None, **kwargs):
        super(WeightScalingDense, self).__init__(**kwargs)
        self.n_units = n_units
        self.gain = gain
        self.use_pixelnorm = use_pixelnorm
        self.activate = activate

    def build(self, input_shape):
        self.dense = Dense(self.n_units, use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(0., 1.))
        self.bias = Bias()
        self.wscale = WeightScaling(shape=(input_shape[-1],), gain=self.gain)

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.wscale(x)
        x = self.bias(x)
        if self.activate == 'LeakyReLU': x = LeakyReLU(0.2)(x)
        if self.activate == 'tanh': x = Activation('tanh')(x)
        if self.use_pixelnorm: x = PixelNormalization()(x)
        return x

class WeightScalingConv(Layer):
    def __init__(self, n_filters, kernel_size, gain, use_pixelnorm=False, activate=None, strides=(1,1), **kwargs):
        super(WeightScalingConv, self).__init__(**kwargs)
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.gain = gain
        self.use_pixelnorm = use_pixelnorm
        self.activate = activate
        self.strides = strides

    def build(self, input_shape):
        self.conv = Conv2D(self.n_filters, self.kernel_size, strides=self.strides, use_bias=False, padding="same", 
                           kernel_initializer=tf.keras.initializers.RandomNormal(0., 1.))
        self.bias = Bias()
        self.wscale = WeightScaling(shape=(self.kernel_size[0], self.kernel_size[1], input_shape[-1]), gain=self.gain)
      
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.wscale(x)
        x = self.bias(x)
        if self.activate == 'LeakyReLU': x = LeakyReLU(0.2)(x)
        if self.activate == 'tanh': x = Activation('tanh')(x)
        if self.use_pixelnorm: x = PixelNormalization()(x)
        return x 

class MinibatchStdev(Layer):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)
    
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True) + 1e-8)
        average_stddev = tf.reduce_mean(stddev, keepdims=True)
        shape = tf.shape(inputs)
        minibatch_stddev = tf.tile(average_stddev, (shape[0], shape[1], shape[2], 1))
        return tf.concat([inputs, minibatch_stddev], axis=-1)

class WeightedSum(Layer):
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = tf.Variable(0., dtype=tf.float32, trainable=False, name="alpha_var")
    def call(self, inputs):
        return ((1.0 - self.alpha) * inputs[0] + (self.alpha * inputs[1]))

# ---------------------------------------------------------------------------
# PROGAN MODEL
# ---------------------------------------------------------------------------

class ProGAN(Model):
    def __init__(self, latent_dim, d_steps=1, gp_weight=10.0, drift_weight=0.001):
        super(ProGAN, self).__init__()
        self.latent_dim = latent_dim
        self.d_steps = d_steps
        self.gp_weight = gp_weight
        self.drift_weight = drift_weight
        self.n_depth = 0
        self.discriminator = self.init_discriminator()
        self.generator = self.init_generator()

    def init_discriminator(self):
        img_input = Input(shape=(4, 4, 3))
        x = WeightScalingConv(FILTERS[0], (1, 1), np.sqrt(2), activate='LeakyReLU')(img_input)
        x = MinibatchStdev()(x)
        x = WeightScalingConv(FILTERS[0], (3, 3), np.sqrt(2), activate='LeakyReLU')(x)
        x = WeightScalingConv(FILTERS[0], (4, 4), np.sqrt(2), activate='LeakyReLU', strides=(4, 4))(x)
        x = Flatten()(x)
        x = WeightScalingDense(1, gain=1.)(x)
        return Model(img_input, x, name='discriminator')

    def init_generator(self):
        noise = Input(shape=(self.latent_dim,))
        x = PixelNormalization()(noise)      
        x = WeightScalingDense(4*4*FILTERS[0], np.sqrt(2)/4, activate='LeakyReLU', use_pixelnorm=True)(x)
        x = Reshape((4, 4, FILTERS[0]))(x)
        x = WeightScalingConv(FILTERS[0], (4, 4), np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)(x)
        x = WeightScalingConv(FILTERS[0], (3, 3), np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)(x)
        x = WeightScalingConv(3, (1, 1), 1., activate='tanh', use_pixelnorm=False)(x)
        return Model(noise, x, name='generator')

    def fade_in_generator(self):
        self.n_depth += 1
        block_end = self.generator.layers[-2].output 
        block_end = UpSampling2D((2, 2))(block_end)
        x1 = self.generator.layers[-1](block_end)

        x2 = WeightScalingConv(FILTERS[self.n_depth], (3, 3), np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)(block_end)
        x2 = WeightScalingConv(FILTERS[self.n_depth], (3, 3), np.sqrt(2), activate='LeakyReLU', use_pixelnorm=True)(x2)      
        x2 = WeightScalingConv(3, (1, 1), 1., activate='tanh', use_pixelnorm=False)(x2)

        self.generator_stabilize = Model(self.generator.input, x2, name='generator_stab')
        x = WeightedSum(name="weighted_sum")([x1, x2])
        self.generator = Model(self.generator.input, x, name='generator_fade')

    def fade_in_discriminator(self):
        old_shape = self.discriminator.input.shape[1]
        img_input = Input(shape=(old_shape * 2, old_shape * 2, 3))
        
        x2 = WeightScalingConv(FILTERS[self.n_depth], (1, 1), np.sqrt(2), activate='LeakyReLU')(img_input)
        x2 = WeightScalingConv(FILTERS[self.n_depth], (3, 3), np.sqrt(2), activate='LeakyReLU')(x2)
        x2 = WeightScalingConv(FILTERS[self.n_depth-1], (3, 3), np.sqrt(2), activate='LeakyReLU')(x2)
        x2 = AveragePooling2D(pool_size=(2, 2))(x2)

        x1 = AveragePooling2D(pool_size=(2, 2))(img_input)
        x1 = WeightScalingConv(FILTERS[self.n_depth-1], (1, 1), np.sqrt(2), activate='LeakyReLU')(x1)

        x = WeightedSum(name="weighted_sum")([x1, x2])

        for layer in self.discriminator.layers[2:]:
            x = layer(x)
            x2 = layer(x2)

        self.discriminator_stabilize = Model(img_input, x2, name='discriminator_stab')
        self.discriminator = Model(img_input, x, name='discriminator_fade')

    def stabilize_generator(self):
        self.generator = self.generator_stabilize

    def stabilize_discriminator(self):
        self.discriminator = self.discriminator_stabilize

    def compile(self, d_optimizer, g_optimizer):
        super(ProGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

    def gradient_penalty(self, batch_size, real_images, fake_images):
        epsilon = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolation = epsilon * real_images + (1 - epsilon) * fake_images
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolation)
            prediction = self.discriminator(interpolation, training=True)
        grads = gp_tape.gradient(prediction, interpolation)
        l2_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-8)
        return tf.reduce_mean((l2_norm - 1)**2)

    def train_step(self, real_images):
        if isinstance(real_images, tuple): real_images = real_images[0]
        batch_size = tf.shape(real_images)[0]
        
        for _ in range(self.d_steps):
            z = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(z, training=True)
                fake_logits = self.discriminator(fake_images, training=True)
                real_logits = self.discriminator(real_images, training=True)
                d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                drift = tf.reduce_mean(tf.square(real_logits))
                d_loss = d_cost + self.gp_weight * gp + self.drift_weight * drift
            d_grad = tape.gradient(d_loss, self.discriminator.trainable_variables)
            self.d_optimizer.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        z = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            gen_imgs = self.generator(z, training=True)
            gen_logits = self.discriminator(gen_imgs, training=True)
            g_loss = -tf.reduce_mean(gen_logits)
        g_grad = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grad, self.generator.trainable_variables))
        
        return {'d_loss': d_loss, 'g_loss': g_loss}

# ---------------------------------------------------------------------------
# HELPERS & CALLBACKS
# ---------------------------------------------------------------------------

class ShowImage(tf.keras.callbacks.Callback):
    def __init__(self, res, latent_dim=512):
        self.res = res
        self.seed = tf.random.normal([16, latent_dim])
    def on_epoch_end(self, epoch, logs=None):
        imgs = self.model.generator(self.seed, training=False)
        imgs = (imgs.numpy() * 127.5 + 127.5).clip(0, 255).astype("uint8")
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for ax, img in zip(axes.flat, imgs):
            ax.imshow(img); ax.axis("off")
        plt.savefig(os.path.join(OUTPUT_DIR, f"res_{self.res}_ep_{epoch}.png"))
        plt.close()

def get_dataset(res, batch_size, path):
    ds = tf.keras.utils.image_dataset_from_directory(
        path, labels=None, image_size=(res, res), batch_size=batch_size, shuffle=True
    )
    return ds.map(lambda x: (tf.cast(x, tf.float32) - 127.5) / 127.5).repeat().prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------------------
# MAIN TRAIN LOOP
# ---------------------------------------------------------------------------

def train():
    RESOLUTIONS = [4, 8, 16, 32, 64]
    BATCH_SIZES = [64, 32, 16, 8, 4]
    STEPS = 1000
    EPOCHS = 50

    pgan = ProGAN(latent_dim=LATENT_DIM)
    
    # Init optimizers
    d_opt = Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    g_opt = Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
    pgan.compile(d_optimizer=d_opt, g_optimizer=g_opt)

    for i, res in enumerate(RESOLUTIONS):
        bs = BATCH_SIZES[i]
        ds = get_dataset(res, bs, RESSOURCES)
        
        if res > 4:
            pgan.fade_in_generator()
            pgan.fade_in_discriminator()
            
            # RECREATE OPTIMIZERS for new variables
            d_opt = Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
            g_opt = Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
            pgan.compile(d_opt, g_opt)
            
            for epoch in range(10):
                alpha = epoch / 10.0
                pgan.generator.get_layer("weighted_sum").alpha.assign(alpha)
                pgan.discriminator.get_layer("weighted_sum").alpha.assign(alpha)
                pgan.fit(ds.take(STEPS), epochs=1)

            pgan.stabilize_generator()
            pgan.stabilize_discriminator()
            
            # RECREATE OPTIMIZERS after stabilization
            d_opt = Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
            g_opt = Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99, epsilon=1e-8)
            pgan.compile(d_opt, g_opt)

        pgan.fit(ds.take(STEPS), epochs=EPOCHS, callbacks=[ShowImage(res, LATENT_DIM)])

if __name__ == "__main__":
    train()