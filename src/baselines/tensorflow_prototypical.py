"""Optional fresh Keras baseline. The absent historical model is not reconstructed."""
import numpy as np
from .pixel import preprocess
from src.active.scene_generation import SceneGenerator
from src.scene.renderer import draw_scene


def build_encoder(seed=42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    return tf.keras.Sequential([
        tf.keras.layers.Input((48, 48, 1)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16),
    ], name="prototypical_shape_encoder")


def train_encoder(rules, episodes=160, seed=42):
    import tensorflow as tf
    encoder = build_encoder(seed)
    optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
    generator = SceneGenerator(seed)

    @tf.function
    def train_step(support, labels, queries, targets):
        with tf.GradientTape() as tape:
            embedded = encoder(tf.concat([support, queries], axis=0), training=True)
            support_z, query_z = embedded[:12], embedded[12:]
            prototypes = tf.stack([tf.reduce_mean(tf.boolean_mask(support_z, labels == c), axis=0) for c in (0, 1)])
            logits = -tf.reduce_mean(tf.square(query_z[:, None, :]-prototypes[None, :, :]), axis=-1)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True))
        gradient = tape.gradient(loss, encoder.trainable_variables)
        optimizer.apply_gradients(zip(gradient, encoder.trainable_variables))
        return loss

    losses = []
    for i in range(episodes):
        target = rules[i % len(rules)]
        support = generator.balanced_examples(target, 6)
        query = generator.balanced_examples(target, 4)
        sx = np.stack([preprocess(draw_scene(s, 96)) for s, _ in support])
        qx = np.stack([preprocess(draw_scene(s, 96)) for s, _ in query])
        loss = train_step(sx, np.array([y for _, y in support], dtype=np.int32),
                          qx, np.array([y for _, y in query], dtype=np.int32))
        losses.append(float(loss))
        if (i+1) % 30 == 0 or i == episodes-1:
            print(f"Neural training: {i+1}/{episodes} episodes; loss {losses[-1]:.4f}", flush=True)
    return encoder, losses
