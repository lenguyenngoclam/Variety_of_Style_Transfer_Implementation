import tensorflow as tf
import config
import numpy as np
import scipy
import helpers


def feature_reconstruction_loss(feature_outputs, feature_targets):
    loss = tf.add_n(
        [
            tf.reduce_mean(tf.square(feature_outputs[i] - feature_targets[i]))
            / (feature_outputs[i].shape[1] * feature_outputs[i].shape[2])
            for i in range(len(feature_outputs))
        ]
    )

    loss *= config.FEATURE_WEIGHT * 1.0 / len(config.CONTENT_LAYERS)

    return loss


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = input_tensor.shape
    num_location = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_location


def style_reconstruction_loss(style_outputs, style_targets):
    loss = tf.add_n(
        [
            tf.reduce_mean(tf.square(style_outputs[i] - style_targets[i]))
            for i in range(len(style_targets))
        ]
    )

    loss *= config.STYLE_WEIGHT * 1.0 / len(config.STYLE_LAYERS)

    return loss


def feature_style_loss(
    input_images, feature_outputs, style_outputs, feature_targets, style_targets
):
    feature_loss = feature_reconstruction_loss(
        feature_outputs=feature_outputs, feature_targets=feature_targets
    )
    style_loss = style_reconstruction_loss(
        style_outputs=style_outputs, style_targets=style_targets
    )

    total_loss = (
        feature_loss
        + style_loss
        + config.TOTAL_VARIATION_WEIGHT
        * tf.add_n(tf.image.total_variation(input_images))
    )

    return total_loss


if __name__ == "__main__":
    x = tf.random.uniform([4, 64, 64])
    print(style_reconstruction_loss([x], [x + tf.random.normal([4, 64, 64])]))
    print(
        feature_reconstruction_loss(
            [tf.random.uniform([64, 64, 256])], [tf.random.uniform([64, 64, 256])]
        )
    )
    print(
        feature_style_loss(
            input_images=tf.random.uniform([4, 256, 256, 3], -1.0, 1.0),
            feature_outputs=[tf.random.uniform([64, 64, 256])],
            style_outputs=[x],
            feature_targets=[tf.random.uniform([64, 64, 256])],
            style_targets=[x + tf.random.normal([4, 64, 64])],
        )
    )
