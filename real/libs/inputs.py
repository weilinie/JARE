import numpy as np
import tensorflow as tf
import os


def get_filename_queue(split_file, data_dir):
    with open(split_file, 'r') as f:
        filenames = f.readlines()
    filenames = [os.path.join(data_dir, fname.strip()) for fname in filenames]

    for f in filenames:
        if not os.path.exists(f):
            raise ValueError('Failed to find file: ' + f)
    filename_queue = tf.train.string_input_producer(filenames)

    return filename_queue


def create_batch(inputs, batch_size=64, num_preprocess_threads=12, min_queue_examples=1000):
    # Generate a batch of images and labels by building up a queue of examples.
    batch = tf.train.shuffle_batch(
        inputs,
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples
    )

    return batch


def get_input_image(filename_queue, output_size, image_size, c_dim):
    # Read a record, getting filenames from the filename_queue.
    reader = tf.WholeFileReader()
    key, value = reader.read(filename_queue)
    image = tf.image.decode_image(value, channels=c_dim)
    image = tf.cast(image, tf.float32)/255.

    image_shape = tf.shape(image)
    image_height, image_width = image_shape[0], image_shape[1]
    offset_height = tf.cast((image_height - image_size)/2, tf.int32)
    offset_width = tf.cast((image_width - image_size)/2, tf.int32)
    image = tf.image.crop_to_bounding_box(
        image, offset_height, offset_width, image_size, image_size)
    image = tf.image.resize_images(image, [output_size, output_size])

    image.set_shape([output_size, output_size, c_dim])

    return image


def get_input_cifar10(filename_queue):
    # See https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py for how to
    # read and parse examples from CIFAR10 data files.
    # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the input format.
    label_bytes = 1     # 2 for CIFAR-100
    image_bytes = 32 * 32 * 3
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes = label_bytes + image_bytes

    # Read a record, getting filenames from the filename_queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key, value = reader.read(filename_queue)
    record = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    # label = tf.cast(record[0], tf.int32)
    label = tf.cast(
        tf.strided_slice(record, [0], [label_bytes]), tf.int32)

    # The remaining bytes after the label represent the image, which we reshape
    # from [depth * height * width] to [depth, height, width].
    # image = tf.reshape(record[label_bytes:label_bytes+image_bytes], [3, 32, 32])
    image = tf.reshape(
        tf.strided_slice(record, [label_bytes], [label_bytes + image_bytes]), [3, 32, 32])
    image = tf.cast(image, tf.float32) / 255.
    # Convert from [depth, height, width] to [height, width, depth].
    image = tf.transpose(image, [1, 2, 0])

    return image, label


def read_and_decode(filename_queue):
    """Return a single example for queue"""
    img_shape = [3, 32, 32]
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        }
    )
    # some essential steps
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, img_shape)  # THIS IS IMPORTANT
    image.set_shape(img_shape)
    image = tf.cast(image, tf.float32) * (1 / 255.0)  # set to [0, 1]
    image = tf.transpose(image, [1, 2, 0])

    sparse_label = tf.cast(features['label'], tf.int32)

    return image, sparse_label
