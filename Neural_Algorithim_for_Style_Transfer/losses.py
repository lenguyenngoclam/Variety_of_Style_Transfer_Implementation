import tensorflow as tf
import config


def gram_matrix(input_tensor):
    """Create gram matrix for creating style iamge

    @param input: Input tensor. Expected shape : [batch_size, h, w, c]

    @return gram_matrix (Shape : [batch_size, c, c])
    """

    result = tf.linalg.einsum("bijc,bijd->bcd", input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)

    return result / (num_locations)


def get_style_loss(style_outputs, style_targets):
    """Calculate style loss

    @param style_outputs: Generate style outputs
    @param style_targets: Original style targets

    @return style_loss: tf.float32
    """

    style_loss = tf.add_n(
        [
            tf.reduce_mean(tf.square(style_outputs[i] - style_targets[i]))
            for i in range(len(style_targets))
        ]
    )

    style_loss *= config.STYLE_WEIGHT * 1.0 / len(config.STYLE_LAYERS)

    return style_loss


def get_content_loss(content_outputs, content_targets):
    """Calculate content loss

    @param content_outputs: Generated content outputs
    @param content_targets: Original content outputs

    @return content_loss: tf.float32
    """
    content_loss = tf.add_n(
        [
            tf.reduce_mean(tf.square(content_outputs[i] - content_targets[i]))
            for i in range(len(content_targets))
        ]
    )

    content_loss *= config.CONTENT_WEIGHT

    return content_loss


def style_content_loss(
    image, style_outputs, content_outputs, style_targets, content_targets
):
    """Calculate style transfer loss

    @Return Total loss : Style transfer loss
    """
    style_loss = get_style_loss(
        style_outputs=style_outputs, style_targets=style_targets
    )

    content_loss = get_content_loss(
        content_outputs=content_outputs, content_targets=content_targets
    )

    loss = (
        style_loss
        + content_loss
        + config.TOTAL_VARIATION_WEIGHT * tf.image.total_variation(image)
    )

    return loss


def total_variation_loss(image):
    """Calculate absolute difference between neiboring pixels for reducing artifact"""
    return tf.reduce_sum(tf.abs(image[:, 1:, :] - image[:, :-1, :])) + tf.reduce_sum(
        tf.abs(image[1:, :, :] - image[:-1, :, :])
    )
