import tensorflow as tf
import tensorflow_datasets as tfds
from .config import IMAGE_SIZE, BATCH_SIZE


def resize_image(image, label):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    return image, label


def augment_image(image, label):
    image, label = resize_image(image, label)

    image = tf.image.rot90(
        image,
        k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    )
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    return image, label


def load_and_split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    dataset, info = tfds.load(
        "malaria",
        as_supervised=True,
        shuffle_files=True,
        with_info=True
    )

    dataset = dataset["train"]

    total_size = info.splits["train"].num_examples
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    train_data = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_data = remaining.take(val_size)
    test_data = remaining.skip(val_size)

    train_data = (
        train_data
        .shuffle(1000, reshuffle_each_iteration=True)
        .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_data = (
        val_data
        .map(resize_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test_data = (
        test_data
        .map(resize_image)
        .batch(1)
    )

    return train_data, val_data, test_data
