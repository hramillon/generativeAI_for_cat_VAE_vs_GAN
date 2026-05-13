import os, math, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics, optimizers, activations, losses, callbacks

# ---------------------------------------------------------------------------
# HYPERPARAMS
# ---------------------------------------------------------------------------

IMAGE_SIZE = 64
BATCH_SIZE = 64
NOISE_EMBEDDING_SIZE = 32
PLOT_DIFFUSION_STEPS = 20

EMA = 0.999
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 100

# ---------------------------------------------------------------------------
# PREPARE DATA
# ---------------------------------------------------------------------------

train_data = tf.keras.utils.image_dataset_from_directory(
    "ressources/", labels=None, color_mode="rgb",
    image_size=(IMAGE_SIZE, IMAGE_SIZE), batch_size=BATCH_SIZE, shuffle=True, seed=42
)

def preprocess(img):
    img = tf.image.random_flip_left_right(img)
    img = (tf.cast(img, tf.float32) + tf.random.uniform(tf.shape(img))) / 256.0
    img = tf.clip_by_value(img, 1e-5, 1 - 1e-5)
    return tf.math.log(img) - tf.math.log(1.0 - img)

normalized_data = train_data.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
    .cache().shuffle(2000).prefetch(tf.data.AUTOTUNE)

# ---------------------------------------------------------------------------
# DIFFUSION SCHEDULES
# ---------------------------------------------------------------------------

def linear_diffusion_schedule(diffusion_times):
    min_rate, max_rate = 0.0001, 0.02
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1 - betas
    alpha_bars = tf.math.cumprod(alphas)
    return tf.sqrt(1 - alpha_bars), tf.sqrt(alpha_bars)

def cosine_diffusion_schedule(diffusion_times):
    return tf.sin(diffusion_times * math.pi / 2), tf.cos(diffusion_times * math.pi / 2)

def offset_cosine_diffusion_schedule(diffusion_times):
    min_signal_rate, max_signal_rate = 0.02, 0.95
    start_angle = tf.acos(max_signal_rate)
    end_angle   = tf.acos(min_signal_rate)
    diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)
    return tf.sin(diffusion_angles), tf.cos(diffusion_angles)

# ---------------------------------------------------------------------------
# U-NET BUILDING BLOCKS
# ---------------------------------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class SinusoidalEmbedding(layers.Layer):
    def call(self, x):
        frequencies = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), NOISE_EMBEDDING_SIZE // 2))
        angular_speeds = 2.0 * math.pi * frequencies
        return tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)

@tf.keras.utils.register_keras_serializable()
class AttentionBlock(layers.Layer):
    def call(self, x):
        B = tf.shape(x)[0]
        H, W, C = x.shape[1], x.shape[2], x.shape[3]
        seq = tf.reshape(x, (B, H * W, C))
        scores = tf.matmul(seq, seq, transpose_b=True) / tf.sqrt(tf.cast(C, tf.float32))
        weights = tf.nn.softmax(scores, axis=-1)
        out = tf.matmul(weights, seq)
        out = tf.reshape(out, tf.shape(x))
        return x + out

def ResidualBlock(width):
    def apply(x):
        residual = x if x.shape[3] == width else layers.Conv2D(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same", activation=activations.swish)(x)
        x = layers.Conv2D(width, kernel_size=3, padding="same")(x)
        return layers.Add()([x, residual])
    return apply

def DownBlock(width, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width)(x)
            skips.append(x)
        return layers.AveragePooling2D(pool_size=2)(x)
    return apply

def UpBlock(width, block_depth):
    def apply(x):
        x, skips = x
        x = layers.UpSampling2D(size=2, interpolation="bilinear")(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width)(x)
        return x
    return apply

# ---------------------------------------------------------------------------
# U-NET ARCHITECTURE
# ---------------------------------------------------------------------------

noisy_images    = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
noise_variances = layers.Input(shape=(1, 1, 1))

x = layers.Conv2D(32, kernel_size=1)(noisy_images)
noise_embedding = SinusoidalEmbedding()(noise_variances)
noise_embedding = layers.UpSampling2D(size=IMAGE_SIZE, interpolation="nearest")(noise_embedding)
x = layers.Concatenate()([x, noise_embedding])

skips = []
x = DownBlock(32,  block_depth=2)([x, skips])
x = DownBlock(64,  block_depth=2)([x, skips])
x = DownBlock(96,  block_depth=2)([x, skips])

x = ResidualBlock(128)(x)
x = AttentionBlock()(x)
x = ResidualBlock(128)(x)

x = UpBlock(96,  block_depth=2)([x, skips])
x = UpBlock(64,  block_depth=2)([x, skips])
x = UpBlock(32,  block_depth=2)([x, skips])

x = layers.Conv2D(3, kernel_size=1, kernel_initializer="zeros")(x)

unet = models.Model([noisy_images, noise_variances], x, name="unet")

# ---------------------------------------------------------------------------
# DIFFUSION MODEL
# ---------------------------------------------------------------------------

class DiffusionModel(models.Model):
    def __init__(self):
        super().__init__()
        self.normalizer         = layers.Normalization()
        self.network            = unet
        self.ema_network        = models.clone_model(self.network)
        self.diffusion_schedule = offset_cosine_diffusion_schedule
        self.noise_loss_tracker = metrics.Mean(name="n_loss")

    def compile(self, **kwargs):
        super().compile(**kwargs)

    @property
    def metrics(self):
        return [self.noise_loss_tracker]

    def call(self, images, training=False):
        images = self.normalizer(images, training=False)
        batch_size = tf.shape(images)[0]
        noises = tf.random.normal(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        diffusion_times = tf.ones((batch_size, 1, 1, 1)) * 0.5
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates, training=training)
        return pred_images

    def denormalize(self, images):
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        return tf.clip_by_value(images, 0.0, 1.0)

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        network = self.network if training else self.ema_network
        pred_noises = network([noisy_images, noise_rates ** 2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size  = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = tf.ones((num_images, 1, 1, 1)) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images  = self.denoise(current_images, noise_rates, signal_rates, training=False)
            next_noise_rates, next_signal_rates = self.diffusion_schedule(diffusion_times - step_size)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = tf.random.normal(shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, 3))
        return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))

    def train_step(self, images):
        images = self.normalizer(images, training=True)
        batch_size = tf.shape(images)[0]
        noises = tf.random.normal(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
            noise_loss = self.loss(noises, pred_noises)

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))
        self.noise_loss_tracker.update_state(noise_loss)

        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(EMA * ema_weight + (1 - EMA) * weight)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images):
        images = self.normalizer(images, training=False)
        batch_size = tf.shape(images)[0]
        noises = tf.random.normal(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
        diffusion_times = tf.random.uniform(shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
        self.noise_loss_tracker.update_state(self.loss(noises, pred_noises))
        return {m.name: m.result() for m in self.metrics}

# ---------------------------------------------------------------------------
# CALLBACKS
# ---------------------------------------------------------------------------

def display(images, save_to):
    fig, axes = plt.subplots(1, len(images), figsize=(2 * len(images), 2))
    for img, ax in zip(images, axes):
        ax.imshow(img); ax.axis("off")
    plt.tight_layout(); plt.savefig(save_to); plt.close()

class ImageGenerator(callbacks.Callback):
    def __init__(self, num_img):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        generated = self.model.generate(num_images=self.num_img, diffusion_steps=PLOT_DIFFUSION_STEPS).numpy()
        os.makedirs("./training", exist_ok=True)
        display(generated, save_to=f"./training/generated_img_{epoch:03d}.png")

# ---------------------------------------------------------------------------
# TRAIN
# ---------------------------------------------------------------------------

os.makedirs("./checkpoint", exist_ok=True)
os.makedirs("./logs", exist_ok=True)

STEPS_PER_EPOCH = 467
total_steps  = EPOCHS * STEPS_PER_EPOCH
warmup_steps = total_steps // 10 

lr_schedule = optimizers.schedules.CosineDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=total_steps - warmup_steps,
    warmup_steps=warmup_steps,
    warmup_target=LEARNING_RATE,
    alpha=1e-5,
)

ddm = DiffusionModel()
ddm.normalizer.adapt(normalized_data)

for sample_batch in normalized_data.take(1):
    ddm(sample_batch, training=False)

ddm.compile(
    optimizer=optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=WEIGHT_DECAY,
        clipnorm=1.0,
    ),
    loss=losses.MeanAbsoluteError(),
)

ddm.fit(
    normalized_data,
    epochs=EPOCHS,
    callbacks=[
        callbacks.ModelCheckpoint("./checkpoint/checkpoint.weights.h5", save_weights_only=True, save_freq="epoch", verbose=0),
        ImageGenerator(num_img=10),
    ],
)