import tensorflow as tf
import config


def feature_reconstruction_loss(feature_outputs, feature_targets):
    loss = tf.reduce_mean(
        [
            tf.norm(feature_outputs[i] - feature_targets[i], ord="euclidean") ** 2
            / (feature_outputs[i].shape[1] * feature_outputs[i].shape[2])
            for i in range(len(feature_outputs))
        ]
    )
    return loss


def gram_matrix(input_tensor):
    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = input_tensor.shape
    num_location = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_location


def style_reconstruction_loss(style_outputs, style_targets):
    loss = tf.reduce_mean(
        [
            tf.norm(style_outputs[i] - style_targets[i], ord="fro", axis=(1, 2)) ** 2
            for i in range(len(style_targets))
        ]
    )

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
        config.FEATURE_WEIGHT * feature_loss
        + config.STYLE_WEIGHT * style_loss
        + config.TOTAL_VARIATION_WEIGHT * tf.reduce_mean(tf.image.total_variation(input_images))
    )

    return total_loss


if __name__ == "__main__":
    x = tf.random.uniform([4, 64, 64])
    print(style_reconstruction_loss([x], [x + tf.random.normal([4, 64, 64])]))
    print(feature_reconstruction_loss([tf.random.uniform([4, 256, 256, 3])], [tf.random.uniform([4, 256, 256, 3])]))
    print(feature_style_loss(input_images=tf.random.uniform([4, 256, 256, 3], -1.0, 1.0), 
                feature_outputs=[tf.random.uniform([4, 256, 256, 3])],
                style_outputs= [x],
                feature_targets=[tf.random.uniform([4, 256, 256, 3])],
                style_targets=[x + tf.random.normal([4, 64, 64])]))