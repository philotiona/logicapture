import argparse

def main():
    parser = argparse.ArgumentParser(description='Train and demonstrate the Bongard practice baseline.')
    parser.add_argument('--no-show', action='store_true', help='Save plots without opening windows.')
    args = parser.parse_args()

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # CPU is enough for this small prototype.
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    import math
    import time
    import json
    from pathlib import Path
    import numpy as np
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import tensorflow as tf
    from tensorflow import keras

    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    keras.utils.set_random_seed(7)
    NOTEBOOK_MODE = False
    SHOW_WINDOWS = not args.no_show
    OUTPUT_DIR = Path('bongard_results')
    SIZE = 48
    SUPPORT_PER_GROUP = 6
    QUERIES_PER_GROUP = 4
    RULES = ('circle', 'filled', 'two_objects')
    RULE_NAMES = {
        'circle': 'All objects are circles',
        'filled': 'Objects are filled',
        'two_objects': 'There are two objects',
    }

    def finish_plot(filename):
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        plt.gcf().savefig(OUTPUT_DIR / filename, dpi=120, bbox_inches='tight')
        if NOTEBOOK_MODE:
            plt.show()
        elif not SHOW_WINDOWS:
            plt.close(plt.gcf())

    print('TensorFlow:', tf.__version__, '| device: CPU')

    def draw_scene(rng, rule, label, shifted=False):
        shape = int(rng.integers(0, 3))  # 0=circle, 1=triangle, 2=square
        filled = bool(rng.integers(0, 2))
        count = int(rng.integers(1, 3))
        if rule == 'circle':
            shape = 0 if label else int(rng.integers(1, 3))
        elif rule == 'filled':
            filled = bool(label)
        elif rule == 'two_objects':
            count = 2 if label else 1
        else:
            raise ValueError('Unknown practice rule: ' + rule)

        # Render at double resolution, then downsample for smoother edges.
        scale = 2
        image = Image.new('L', (SIZE * scale, SIZE * scale), 0)
        draw = ImageDraw.Draw(image)
        centers = [(24, 24)] if count == 1 else [(13, 24), (35, 24)]
        dx, dy = rng.uniform(-3, 3, size=2)
        for cx, cy in centers:
            radius = rng.uniform(6, 9)
            x, y = (cx + dx) * scale, (cy + dy) * scale
            r = radius * scale
            fill = 255 if filled else None
            if shape == 0:
                draw.ellipse((x-r, y-r, x+r, y+r), fill=fill, outline=255, width=4)
            else:
                sides = 3 if shape == 1 else 4
                angle = -math.pi / 2 if sides == 3 else math.pi / 4
                angle += rng.uniform(-0.2, 0.2)
                points = [(x+r*math.cos(angle+2*math.pi*k/sides),
                           y+r*math.sin(angle+2*math.pi*k/sides)) for k in range(sides)]
                draw.polygon(points, fill=fill)
                draw.line(points + [points[0]], fill=255, width=4, joint='curve')
        if shifted:
            image = image.rotate(float(rng.uniform(-50, 50)),
                                 resample=Image.Resampling.BICUBIC, fillcolor=0)
        image = image.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        if shifted:
            pixels = np.clip(pixels + rng.normal(0, 0.06, pixels.shape), 0, 1)
        return pixels.astype(np.float32)


    def make_episode(rule, seed, shifted=False):
        rng = np.random.default_rng(seed)
        # Support order is fixed and known: positive group, then negative group.
        support_labels = [1] * SUPPORT_PER_GROUP + [0] * SUPPORT_PER_GROUP
        query_labels = rng.permutation([0] * QUERIES_PER_GROUP + [1] * QUERIES_PER_GROUP)
        labels = support_labels + query_labels.tolist()
        images = np.stack([draw_scene(rng, rule, label, shifted) for label in labels])
        x = tf.convert_to_tensor(images[..., None], dtype=tf.float32)
        y = tf.convert_to_tensor(query_labels, dtype=tf.int32)
        return x, y


    def show_episode(images, answers, probabilities=None, title='Practice puzzle', filename='puzzle.png'):
        images = images.numpy()[..., 0]
        answers = answers.numpy()
        fig, axes = plt.subplots(3, 8, figsize=(13, 5.4))
        for ax in axes.flat:
            ax.axis('off')
        for row, start, name in [(0, 0, 'Positive'), (1, 6, 'Negative')]:
            for col in range(6):
                axes[row, col].imshow(images[start+col], cmap='gray_r', vmin=0, vmax=1)
                axes[row, col].set_title(name + ' ' + str(col+1), fontsize=9)
        for col, answer in enumerate(answers):
            ax = axes[2, col]
            ax.imshow(images[12+col], cmap='gray_r', vmin=0, vmax=1)
            if probabilities is None:
                text = 'Query ' + str(col+1) + '\nTrue: ' + ('+' if answer else '-')
                color = '#222222'
            else:
                score = float(probabilities[col])
                prediction = int(score >= 0.5)
                text = f"Pred: {'+' if prediction else '-'}  True: {'+' if answer else '-'}\nP(+): {score:.2f}"
                color = '#166534' if prediction == answer else '#b91c1c'
            ax.set_title(text, fontsize=9, color=color)
        fig.suptitle(title, fontsize=13)
        plt.tight_layout()
        finish_plot(filename)


    example_x, example_y = make_episode('circle', seed=900_001)
    show_episode(example_x, example_y, title='Input format: 6 positive examples, 6 negative examples, 8 queries', filename='input_examples.png')
    print('Image tensor:', tuple(example_x.shape))
    print('Query labels:', example_y.numpy().tolist())

    def make_encoder():
        return keras.Sequential([
            keras.layers.Input(shape=(SIZE, SIZE, 1)),
            keras.layers.Conv2D(8, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
            keras.layers.MaxPooling2D(2),
            keras.layers.Flatten(),
            keras.layers.Dense(64),
            keras.layers.UnitNormalization(axis=-1),
        ], name='visual_encoder')


    def classify(embeddings):
        positive_center = tf.reduce_mean(embeddings[:6], axis=0)
        negative_center = tf.reduce_mean(embeddings[6:12], axis=0)
        centers = tf.stack([negative_center, positive_center])  # labels 0, 1
        queries = embeddings[12:]
        distances = tf.reduce_sum(
            tf.square(queries[:, None, :] - centers[None, :, :]), axis=2)
        return -10.0 * distances


    def pixel_baseline(images):
        # No trainable weights: compare small brightness grids directly.
        pooled = tf.nn.avg_pool2d(images, ksize=4, strides=4, padding='VALID')
        vectors = tf.reshape(pooled, (tf.shape(images)[0], -1))
        return tf.math.l2_normalize(vectors, axis=1)


    model = make_encoder()
    print('Trainable parameters:', model.count_params())
    print('Encoder output:', tuple(model(example_x).shape))

    def evaluate(encoder, shifted=False, episodes_per_rule=20):
        results = {}
        for rule_index, rule in enumerate(RULES):
            correct, total = 0, 0
            for episode in range(episodes_per_rule):
                seed = 1_000_000 + rule_index * 1000 + episode
                images, answers = make_episode(rule, seed, shifted=shifted)
                predictions = tf.argmax(classify(encoder(images)), axis=1, output_type=tf.int32)
                correct += int(tf.reduce_sum(tf.cast(predictions == answers, tf.int32)))
                total += len(answers)
            results[rule] = correct / total
        return results


    pixel_results = evaluate(pixel_baseline)
    untrained_results = evaluate(model)
    print('Pixel baseline:', pixel_results)
    print('Untrained CNN:', untrained_results)

    EPISODES = 1000
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_function = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train_step(images, answers):
        with tf.GradientTape() as tape:
            embeddings = model(images, training=True)
            scores = classify(embeddings)
            loss = loss_function(answers, scores)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    loss_history = []
    training_start = time.perf_counter()
    for step in range(EPISODES):
        rule = RULES[step % len(RULES)]
        images, answers = make_episode(rule, seed=step)
        loss = train_step(images, answers)
        loss_history.append(float(loss))
        if (step + 1) % 90 == 0:
            print(f'Episode {step+1:3d}/{EPISODES}: last-90 mean loss = {np.mean(loss_history[-90:]):.4f}')
    training_seconds = time.perf_counter() - training_start
    print(f'Training time on this machine: {training_seconds:.1f} seconds')

    fig, ax = plt.subplots(figsize=(8, 3))
    window = 30
    smoothed_loss = np.convolve(loss_history, np.ones(window) / window, mode='valid')
    ax.plot(np.arange(window, EPISODES+1), smoothed_loss, color='#2563eb')
    ax.set(xlabel='Training episode', ylabel='Cross-entropy (30-episode mean)', title='Actual training loss')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    finish_plot('training_loss.png')

    trained_results = evaluate(model)
    shifted_pixel_results = evaluate(pixel_baseline, shifted=True)
    shifted_results = evaluate(model, shifted=True)

    print(f"{'Condition':<25} {'Circle':>9} {'Filled':>9} {'Two':>9} {'Mean':>9}")
    for name, result in [
        ('Pixels / familiar style', pixel_results),
        ('Untrained CNN / familiar', untrained_results),
        ('Trained CNN / familiar', trained_results),
        ('Pixels / rotated + noisy', shifted_pixel_results),
        ('Trained CNN / shifted', shifted_results),
    ]:
        values = [result[r] for r in RULES]
        print(f'{name:<25}' + ''.join(f'{v:>9.1%}' for v in values) + f'{np.mean(values):>9.1%}')

    fig, ax = plt.subplots(figsize=(8, 3.8))
    positions = np.arange(len(RULES))
    width = 0.25
    for offset, name, result, color in [
        (-width, 'Pixel baseline', pixel_results, '#94a3b8'),
        (0, 'Trained CNN', trained_results, '#2563eb'),
        (width, 'Trained CNN: rotated + noisy', shifted_results, '#d97706'),
    ]:
        ax.bar(positions+offset, [100*result[r] for r in RULES], width, label=name, color=color)
    ax.axhline(50, color='#555555', linestyle='--', linewidth=1, label='50% reference')
    ax.set(xticks=positions, xticklabels=['Circles', 'Filled objects', 'Two objects'],
           ylabel='Query accuracy (%)', ylim=(0, 108), title='Fresh images; the three rules were seen during training')
    ax.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    finish_plot('accuracy.png')

    DEMO_RULE = 'circle'  # 'circle', 'filled', or 'two_objects'
    DEMO_SEED = 2_000_001
    SHIFTED = False
    demo_x, demo_y = make_episode(DEMO_RULE, DEMO_SEED, shifted=SHIFTED)
    demo_scores = classify(model(demo_x, training=False))
    demo_probabilities = tf.nn.softmax(demo_scores, axis=1)[:, 1].numpy()
    show_episode(demo_x, demo_y, demo_probabilities,
                 title=RULE_NAMES[DEMO_RULE] + (' | rotated + noisy' if SHIFTED else ' | familiar style'),
                 filename='demo_predictions.png')

    failure_found = False
    for attempt in range(30):
        rule = RULES[attempt % len(RULES)]
        x, y = make_episode(rule, seed=3_000_000+attempt, shifted=True)
        probabilities = tf.nn.softmax(classify(model(x, training=False)), axis=1)[:, 1].numpy()
        errors = (probabilities >= 0.5).astype(int) != y.numpy()
        if errors.any():
            show_episode(x, y, probabilities, title='Failure illustration: ' + RULE_NAMES[rule], filename='failure_example.png')
            print(f'{int(errors.sum())} wrong predictions out of {len(y)} in this selected episode.')
            failure_found = True
            break
    if not failure_found:
        print('No error found in these 30 episodes. This does not prove general reasoning.')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR / 'encoder.keras')
    report = {
        'tensorflow_version': tf.__version__,
        'training_episodes': EPISODES,
        'queries_per_condition': 480,
        'training_seconds': training_seconds,
        'pixel_baseline': pixel_results,
        'untrained_cnn': untrained_results,
        'trained_cnn': trained_results,
        'shifted_pixel_baseline': shifted_pixel_results,
        'shifted_trained_cnn': shifted_results,
        'scope': 'Generated practice images; three familiar rules. Not official Bongard results.',
    }
    with (OUTPUT_DIR / 'results.json').open('w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    print('Saved graphs, measurements, and encoder to:', OUTPUT_DIR.resolve())

    if SHOW_WINDOWS:
        print('Close the plot windows when you finish viewing them.')
        plt.show()

if __name__ == '__main__':
    main()
