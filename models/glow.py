import os, yaml, numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, metrics, optimizers
from scipy.linalg import lu

OUTPUT_DIR = "training/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open("configs/cat_wgan.yaml") as f:
    config = yaml.safe_load(f)

train_data = tf.keras.utils.image_dataset_from_directory(
    "ressources/", labels=None, color_mode="rgb",
    image_size=(64, 64), batch_size=64, shuffle=True, seed=42
)

def preprocess(img):
    img = (tf.cast(img, tf.float32) + tf.random.uniform(tf.shape(img))) / 256.0
    img = tf.clip_by_value(img, 1e-5, 1 - 1e-5)
    return tf.math.log(img) - tf.math.log(1.0 - img)

normalized_data = (
    train_data
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(2000).cache().prefetch(tf.data.AUTOTUNE)
)

N_DIMS = 64 * 64 * 3 


# ─────────────────────────────────────────────
# Couches
# ─────────────────────────────────────────────

class ActNorm(layers.Layer):
    def build(self, s):
        c = s[-1]
        self.scale = self.add_weight(
            shape=(1, 1, 1, c), name="scale", initializer="ones", trainable=True)
        self.bias = self.add_weight(
            shape=(1, 1, 1, c), name="bias", initializer="zeros", trainable=True)

    def data_init(self, x):
        m = tf.reduce_mean(x, [0, 1, 2], keepdims=True)
        s = tf.math.reduce_std(x, [0, 1, 2], keepdims=True) + 1e-6
        self.scale.assign(1.0 / s)
        self.bias.assign(-m / s)

    def call(self, x, forward=True):
        hw = tf.cast(tf.shape(x)[1] * tf.shape(x)[2], tf.float32)
        ld = hw * tf.reduce_sum(tf.math.log(tf.abs(self.scale) + 1e-8))
        if forward:
            return self.scale * x + self.bias, ld
        return (x - self.bias) / (self.scale + 1e-8), -ld


class Inv1x1Conv(layers.Layer):
    def __init__(self, c, **kw):
        super().__init__(**kw)
        self.c = c

    def build(self, _):
        q = np.linalg.qr(np.random.randn(self.c, self.c))[0].astype("float32")
        P, L, U = lu(q)
        self.P_var = self.add_weight(
            shape=(self.c, self.c), name="P",
            initializer=tf.constant_initializer(P), trainable=False)
        self.L_var = self.add_weight(
            shape=(self.c, self.c), name="L",
            initializer=tf.constant_initializer(L), trainable=True)
        U_off = U - np.diag(np.diag(U))
        self.U_off = self.add_weight(
            shape=(self.c, self.c), name="U_off",
            initializer=tf.constant_initializer(U_off), trainable=True)
        self.log_s = self.add_weight(
            shape=(self.c,), name="log_s",
            initializer=tf.constant_initializer(
                np.log(np.abs(np.diag(U)) + 1e-8)), trainable=True)

    def _W(self):
        L = (tf.linalg.band_part(self.L_var, -1, 0)
             - tf.linalg.diag(tf.linalg.diag_part(self.L_var))
             + tf.eye(self.c))
        U = (tf.linalg.band_part(self.U_off, 0, -1)
             - tf.linalg.diag(tf.linalg.diag_part(self.U_off))
             + tf.linalg.diag(tf.exp(self.log_s)))
        return self.P_var @ L @ U

    def call(self, x, forward=True):
        hw = tf.cast(tf.shape(x)[1] * tf.shape(x)[2], tf.float32)
        ld = hw * tf.reduce_sum(self.log_s)
        W  = self._W()
        if not forward:
            W  = tf.linalg.inv(W)
            ld = -ld
        return tf.nn.conv2d(x, tf.reshape(W, [1, 1, self.c, self.c]), [1, 1, 1, 1], "SAME"), ld


def CouplingNet(c_in, dim=64):
    inp = layers.Input(shape=(None, None, c_in))
    x = layers.Conv2D(dim, 3, padding="same", activation="relu")(inp)
    for _ in range(2):
        skip = x
        x = layers.Conv2D(dim, 3, padding="same")(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(dim, 3, padding="same")(x)
        x = layers.LayerNormalization()(x)
        x = layers.Add()([x, skip])
        x = layers.ReLU()(x)
    s = layers.Conv2D(c_in, 3, padding="same",
                      kernel_initializer="zeros", bias_initializer="zeros")(x)
    t = layers.Conv2D(c_in, 3, padding="same",
                      kernel_initializer="zeros", bias_initializer="zeros")(x)
    return models.Model(inp, [s, t])


# ─────────────────────────────────────────────
# Modèle GLOW
# ─────────────────────────────────────────────

class GLOW(models.Model):
    def __init__(self, L=3, K=8, dim=64, T=0.7):
        super().__init__()
        self.L = L
        self.K = K
        self.T = T

        c = 3
        level_channels = []
        for l in range(L):
            c = c * 4
            level_channels.append(c)
            if l < L - 1:
                c = c // 2
        self.level_channels = level_channels

        self.actnorms  = [[ActNorm()                                 for _ in range(K)] for l in range(L)]
        self.inv1x1s   = [[Inv1x1Conv(level_channels[l])            for _ in range(K)] for l in range(L)]
        self.couplings = [[CouplingNet(level_channels[l] // 2, dim) for _ in range(K)] for l in range(L)]

        self._log_prob = lambda z: -0.5 * (z ** 2 + tf.math.log(2 * np.pi))
        self.loss_tracker = metrics.Mean(name="loss")
        self.bpd_tracker  = metrics.Mean(name="bpd")

    @property
    def metrics(self):
        return [self.loss_tracker, self.bpd_tracker]

    def _step(self, x, l, i, fwd=True):
        if fwd:
            x, ld1 = self.actnorms[l][i](x, forward=True)
            x, ld2 = self.inv1x1s[l][i](x, forward=True)
            x1, x2 = tf.split(x, 2, axis=-1)
            s, t = self.couplings[l][i](x1)
            s = tf.tanh(s)
            ld3 = tf.reduce_sum(s, [1, 2, 3])
            return tf.concat([x1, x2 * tf.exp(s) + t], -1), ld1 + ld2 + ld3
        else:
            x1, x2 = tf.split(x, 2, axis=-1)
            s, t = self.couplings[l][i](x1)
            s = tf.tanh(s)
            x, _ = self.inv1x1s[l][i](
                tf.concat([x1, (x2 - t) * tf.exp(-s)], -1), forward=False)
            x, _ = self.actnorms[l][i](x, forward=False)
            return x

    def encode(self, x):
        log_det = 0.0
        zs = []
        for l in range(self.L):
            x = tf.nn.space_to_depth(x, 2)
            for i in range(self.K):
                x, ld = self._step(x, l, i, True)
                log_det += ld
            if l < self.L - 1:
                x, z = tf.split(x, 2, axis=-1)
                zs.append(z)
            else:
                zs.append(x)
        return zs, log_det

    def decode(self, zs):
        x = zs[-1]
        for l in reversed(range(self.L)):
            if l < self.L - 1:
                x = tf.concat([x, zs[l]], -1)
            for i in reversed(range(self.K)):
                x = self._step(x, l, i, False)
            x = tf.nn.depth_to_space(x, 2)
        return x

    def call(self, x, training=True):
        return self.encode(x) if training else self.decode(x)

    def actnorm_data_init(self, x):
        """
        Initialise chaque ActNorm sur les activations réelles en rejouant
        exactement le chemin de encode() step par step.
        """
        for l in range(self.L):
            x = tf.nn.space_to_depth(x, 2)
            for i in range(self.K):
                self.actnorms[l][i](x, forward=True)
                self.actnorms[l][i].data_init(x)
                x, _ = self.actnorms[l][i](x, forward=True)
                x, _ = self.inv1x1s[l][i](x, forward=True)
                x1, x2 = tf.split(x, 2, axis=-1)
                s, t = self.couplings[l][i](x1)
                s = tf.tanh(s)
                x = tf.concat([x1, x2 * tf.exp(s) + t], -1)
            if l < self.L - 1:
                x, _ = tf.split(x, 2, axis=-1)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            zs, log_det = self(data, training=True)
            log_lik = sum(tf.reduce_sum(self._log_prob(z), [1, 2, 3]) for z in zs)
            loss = -tf.reduce_mean((log_lik + log_det) / N_DIMS)

        grads = tape.gradient(loss, self.trainable_variables)
        grads = [tf.zeros_like(v) if g is None else g
                 for g, v in zip(grads, self.trainable_variables)]
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        bpd = (loss + tf.math.log(256.0)) / tf.math.log(2.0)
        self.loss_tracker.update_state(loss)
        self.bpd_tracker.update_state(bpd)
        return {"loss": self.loss_tracker.result(), "bpd": self.bpd_tracker.result()}

    def generate(self, n=8):
        zs = []
        c  = 3
        for l in range(self.L):
            c = c * 4
            h = 64 // (2 ** (l + 1))
            if l < self.L - 1:
                zs.append(tf.random.normal([n, h, h, c // 2]) * self.T)
                c = c // 2
            else:
                zs.append(tf.random.normal([n, h, h, c]) * self.T)
        return tf.clip_by_value(tf.sigmoid(self.decode(zs)), 0.0, 1.0)


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────

class DiagnosticCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 10:
            return
        imgs = self.model.generate(8)
        fig, axes = plt.subplots(1, 8, figsize=(16, 2.5))
        for ax, img in zip(axes, imgs):
            ax.imshow(img.numpy())
            ax.axis("off")
        fig.suptitle(f"Epoch {epoch+1} | loss={logs['loss']:.4f} | bpd={logs['bpd']:.3f}")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}glow_{epoch+1:04d}.png", dpi=100)
        plt.close()


class WarmupLR(tf.keras.callbacks.Callback):
    def __init__(self, lr=1e-4, n=100):
        self.lr = lr
        self.n  = n

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.n:
            self.model.optimizer.learning_rate = float(self.lr * (epoch + 1) / self.n)


# ─────────────────────────────────────────────
# Entraînement
# ─────────────────────────────────────────────

model = GLOW(L=3, K=8, dim=64, T=0.7)
model.compile(optimizer=optimizers.Adam(1e-4), jit_compile=True)

init_batch = next(iter(normalized_data))
model.actnorm_data_init(init_batch)
print("ActNorm initialisé.")

model.fit(
    normalized_data,
    epochs=1000,
    callbacks=[
        DiagnosticCallback(),
        WarmupLR(lr=1e-4, n=100),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_DIR + "best.keras", save_best_only=True, monitor="loss"),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=50, min_lr=1e-6),
        tf.keras.callbacks.CSVLogger(OUTPUT_DIR + "history.csv"),
    ]
)